"""
Grid Search for GuidedBox Model - PW + Network Architecture Variations
=======================================================================
Focused grid search with fixed lr=0.0001, ema=0.9, img_size=512
and variable:
  - positive_weight: [1, 1.5, 2, 2.5, 3]
  - network_type:    [resnet50_3layer, resnet34_2layer, resnet50_4layer]

Total: 5 pw x 3 networks = 15 combinations

The 3 network variations:
  1. resnet50_3layer  - Original: ResNet50 backbone, decoder 2048->1024->512->1
  2. resnet34_2layer  - Lightweight: ResNet34 backbone, decoder 512->256->1
  3. resnet50_4layer  - Heavy: ResNet50 backbone, decoder 2048->1024->512->256->1

Usage:
    python grid_search_pw_only.py
    python grid_search_pw_only.py --config guidedbox_config.yaml
    python grid_search_pw_only.py --epochs 20 --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision import models

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import json
import os
import argparse
import logging
import time
from datetime import datetime


# ============================================================
# Logging Setup
# ============================================================
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"grid_search_pw_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return log_file


# ============================================================
# Dataset
# ============================================================
class IJmondBboxDataset(Dataset):
    def __init__(self, records, img_npy_dir, img_size=256):
        self.records = records
        self.img_npy_dir = img_npy_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = os.path.join(self.img_npy_dir, f"{record['id']}.npy")
        image = np.load(img_path)

        h_img, w_img = record['h_image'], record['w_image']
        x_bbox, y_bbox = record['x_bbox'], record['y_bbox']
        w_bbox, h_bbox = record['w_bbox'], record['h_bbox']

        actual_h, actual_w = image.shape[:2]
        scale_x = actual_w / w_img
        scale_y = actual_h / h_img

        x_bbox_scaled = int(x_bbox * scale_x)
        y_bbox_scaled = int(y_bbox * scale_y)
        w_bbox_scaled = int(w_bbox * scale_x)
        h_bbox_scaled = int(h_bbox * scale_y)

        pseudo_mask = np.zeros((actual_h, actual_w), dtype=np.float32)
        x1 = max(0, x_bbox_scaled)
        y1 = max(0, y_bbox_scaled)
        x2 = min(actual_w, x_bbox_scaled + w_bbox_scaled)
        y2 = min(actual_h, y_bbox_scaled + h_bbox_scaled)
        pseudo_mask[y1:y2, x1:x2] = 1.0

        image_pil = Image.fromarray(image).resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_pil = Image.fromarray((pseudo_mask * 255).astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        image_tensor = TF.to_tensor(image_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        box_tensor = torch.tensor([
            x_bbox / w_img, y_bbox / h_img,
            w_bbox / w_img, h_bbox / h_img,
        ], dtype=torch.float32)

        return image_tensor, box_tensor, mask_tensor


# ============================================================
# Model with configurable backbone + decoder
# ============================================================
NETWORK_TYPES = ['resnet50_3layer', 'resnet34_2layer', 'resnet50_4layer']


class GuidedBoxModel(nn.Module):
    """
    Teacher-student model with selectable backbone and decoder depth.

    network_type controls backbone + decoder:
      - resnet50_3layer : ResNet50,  decoder 2048->1024->512->1  (original)
      - resnet34_2layer : ResNet34,  decoder 512->256->1         (lightweight)
      - resnet50_4layer : ResNet50,  decoder 2048->1024->512->256->1 (heavy)
    """

    def __init__(self, config, network_type='resnet50_3layer'):
        super().__init__()
        self.network_type = network_type
        self.alpha = config['ema_alpha']
        num_classes = config['num_classes']

        # ---------- backbone ----------
        if network_type in ('resnet50_3layer', 'resnet50_4layer'):
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif network_type == 'resnet34_2layer':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # ---------- decoder channel list ----------
        if network_type == 'resnet50_3layer':
            ch = [2048, 1024, 512]
        elif network_type == 'resnet34_2layer':
            ch = [512, 256]
        elif network_type == 'resnet50_4layer':
            ch = [2048, 1024, 512, 256]
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        self.teacher = self._build_decoder(ch, num_classes)
        self.student = self._build_decoder(ch, num_classes)

    @staticmethod
    def _build_decoder(channel_list, num_classes):
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i + 1],
                                    kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channel_list[-1], num_classes, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, images, positive_weight, boxes=None, masks=None,
                return_confidence=False):
        features = self.backbone(images)
        teacher_output = self.teacher(features)
        student_output = self.student(features)

        teacher_output = F.interpolate(teacher_output, size=images.shape[2:],
                                       mode='bilinear', align_corners=False)
        student_output = F.interpolate(student_output, size=images.shape[2:],
                                       mode='bilinear', align_corners=False)

        if self.training:
            return self.compute_loss(teacher_output, student_output,
                                     boxes=boxes, masks=masks,
                                     positive_weight=positive_weight)
        else:
            if return_confidence:
                conf = self.compute_confidence_scores(teacher_output, student_output)
                return student_output, conf
            return student_output

    def compute_loss(self, teacher_output, student_output, boxes, masks,
                     positive_weight):
        conf_scores = self.compute_confidence_scores(teacher_output, student_output)
        mask_loss = self.robust_pseudo_mask_loss(student_output, masks, conf_scores,
                                                 positive_weight=positive_weight)
        box_loss = F.mse_loss(teacher_output, student_output)
        return box_loss + mask_loss

    def robust_pseudo_mask_loss(self, preds, pseudo_masks, conf_scores,
                                positive_weight=5):
        pixel_loss = F.binary_cross_entropy_with_logits(
            preds, pseudo_masks,
            pos_weight=torch.tensor(float(positive_weight), device=preds.device),
        )
        affinity_loss = self.enhanced_mask_affinity_loss(preds, pseudo_masks)
        return torch.mean(conf_scores * (0.4 * pixel_loss + 0.1 * affinity_loss))

    def enhanced_mask_affinity_loss(self, preds, pseudo_masks):
        eps = 1e-7
        affinity_loss = 0.0
        for i in range(preds.size(0)):
            mask_pred = torch.sigmoid(preds[i])
            mask_pred = torch.clamp(mask_pred, eps, 1 - eps)
            neighbors = F.max_pool2d(
                mask_pred.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0) > 0.45
            fg_loss = -torch.mean(torch.log(mask_pred[neighbors])) if neighbors.sum() > 0 else 0.0
            bg_loss = -torch.mean(torch.log(1 - mask_pred[~neighbors])) if (~neighbors).sum() > 0 else 0.0
            affinity_loss += fg_loss + bg_loss
        return affinity_loss / preds.size(0)

    def compute_confidence_scores(self, teacher_output, student_output):
        return torch.sigmoid(F.cosine_similarity(teacher_output, student_output))

    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(),
                                     self.student.parameters()):
            t_param.data = self.alpha * t_param.data + (1 - self.alpha) * s_param.data


# ============================================================
# Metrics
# ============================================================
def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()


def calculate_dice(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2 * intersection / (pred_binary.sum() + target.sum() + 1e-8)).item()


# ============================================================
# Data Loading
# ============================================================
def load_records(config):
    with open(config['bbox_labels_path'], 'r') as f:
        bbox_data = json.load(f)

    all_records = bbox_data['data']
    logging.info(f"Total records in JSON: {len(all_records)}")

    npy_ids = set(
        f.replace('.npy', '') for f in os.listdir(config['img_npy_path'])
        if f.endswith('.npy')
    )
    logging.info(f"Total npy images: {len(npy_ids)}")

    matched_records = [r for r in all_records if str(r['id']) in npy_ids]
    logging.info(f"Records with matching npy files: {len(matched_records)}")

    SMOKE_STATES = {3, 4, 9, 10, 11, 13, 15}
    NO_SMOKE_STATES = {5, 12, 14}

    smoke_records = [r for r in matched_records if r['label_state'] in SMOKE_STATES]
    no_smoke_records = [r for r in matched_records if r['label_state'] in NO_SMOKE_STATES]

    logging.info(f"Smoke records: {len(smoke_records)}")
    logging.info(f"No-smoke records: {len(no_smoke_records)}")

    train_records = smoke_records + no_smoke_records
    logging.info(f"Using {len(train_records)} records for training")
    return train_records


# ============================================================
# Single training run
# ============================================================
def train_single_run(params, train_records, config, grid_epochs,
                     grid_save_dir, device):
    net_type = params['network_type']
    pw = params['positive_weight']
    run_name = f"{net_type}_pw{pw}"
    logging.info(f"Starting run: {run_name}")
    start_time = time.time()

    run_dataset = IJmondBboxDataset(train_records, config['img_npy_path'],
                                    img_size=config['img_size'])
    val_size = int(len(run_dataset) * 0.2)
    train_size = len(run_dataset) - val_size
    run_train_ds, run_val_ds = random_split(
        run_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    run_train_loader = DataLoader(run_train_ds, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=2,
                                  pin_memory=(device.type == 'cuda'))
    run_val_loader = DataLoader(run_val_ds, batch_size=config['batch_size'],
                                shuffle=False, num_workers=2,
                                pin_memory=(device.type == 'cuda'))

    run_model = GuidedBoxModel(config, network_type=net_type).to(device)
    n_params = sum(p.numel() for p in run_model.parameters())
    logging.info(f"  Model params: {n_params:,}")
    run_optimizer = torch.optim.Adam(run_model.parameters(),
                                     lr=config['learning_rate'])

    run_train_losses, run_val_losses, run_val_ious = [], [], []
    best_iou = 0.0

    for epoch in range(grid_epochs):
        run_model.train()
        epoch_loss = 0.0
        for images, boxes, masks in run_train_loader:
            images, boxes, masks = (images.to(device), boxes.to(device),
                                    masks.to(device))
            loss = run_model(images, boxes=boxes, masks=masks,
                             positive_weight=pw)
            run_optimizer.zero_grad()
            loss.backward()
            run_optimizer.step()
            run_model.update_teacher()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(run_train_loader)
        run_train_losses.append(avg_train_loss)

        run_model.eval()
        epoch_val_loss = 0.0
        epoch_iou = 0.0
        with torch.no_grad():
            for images, boxes, masks in run_val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = run_model(images, pw)
                val_loss = F.binary_cross_entropy_with_logits(
                    outputs, masks,
                    pos_weight=torch.tensor(float(pw), device=device),
                )
                epoch_val_loss += val_loss.item()
                epoch_iou += calculate_iou(outputs, masks)

        avg_val_loss = epoch_val_loss / len(run_val_loader)
        avg_iou = epoch_iou / len(run_val_loader)
        run_val_losses.append(avg_val_loss)
        run_val_ious.append(avg_iou)
        if avg_iou > best_iou:
            best_iou = avg_iou

        logging.info(
            f"  [{run_name}] Epoch {epoch+1}/{grid_epochs}: "
            f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
            f"IoU={avg_iou:.4f}"
        )

    elapsed = time.time() - start_time

    save_path = os.path.join(grid_save_dir, f'{run_name}.pth')
    torch.save({
        'model_state_dict': run_model.state_dict(),
        'params': params,
        'network_type': net_type,
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'val_ious': run_val_ious,
        'best_iou': best_iou,
        'final_iou': run_val_ious[-1],
        'final_val_loss': run_val_losses[-1],
    }, save_path)

    result = {
        'run_name': run_name,
        'network_type': net_type,
        'positive_weight': pw,
        'n_params': n_params,
        'best_iou': best_iou,
        'final_iou': run_val_ious[-1],
        'final_val_loss': run_val_losses[-1],
        'final_train_loss': run_train_losses[-1],
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'val_ious': run_val_ious,
        'time_seconds': elapsed,
        'save_path': save_path,
    }
    logging.info(f"  [{run_name}] Best IoU: {best_iou:.4f} | "
                 f"Time: {elapsed:.1f}s | Saved: {save_path}")
    return result


# ============================================================
# Plotting
# ============================================================
COLORS = {
    'resnet50_3layer': '#1f77b4',
    'resnet34_2layer': '#2ca02c',
    'resnet50_4layer': '#d62728',
}


def plot_results(grid_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    by_net = {}
    for r in grid_results:
        by_net.setdefault(r['network_type'], []).append(r)

    # IoU curves per network
    fig, axes = plt.subplots(1, len(by_net), figsize=(6 * len(by_net), 5),
                             squeeze=False)
    for col, (net, runs) in enumerate(sorted(by_net.items())):
        ax = axes[0, col]
        for r in sorted(runs, key=lambda x: x['positive_weight']):
            ax.plot(range(1, len(r['val_ious']) + 1), r['val_ious'],
                    label=f"pw={r['positive_weight']}", alpha=0.8,
                    marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation IoU')
        ax.set_title(net, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    plt.suptitle('Validation IoU per Network Architecture', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves_by_network.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Bar + scatter comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pw_vals = sorted(set(r['positive_weight'] for r in grid_results))
    net_types = sorted(by_net.keys())
    bar_w = 0.25

    ax = axes[0]
    for i, net in enumerate(net_types):
        ious = []
        for pw in pw_vals:
            match = [r for r in by_net[net] if r['positive_weight'] == pw]
            ious.append(match[0]['best_iou'] if match else 0)
        positions = [x + i * bar_w for x in range(len(pw_vals))]
        ax.bar(positions, ious, bar_w, label=net, color=COLORS.get(net, 'gray'))
    ax.set_xticks([x + bar_w for x in range(len(pw_vals))])
    ax.set_xticklabels([f"pw={p}" for p in pw_vals])
    ax.set_ylabel('Best IoU')
    ax.set_title('Best IoU by PW and Network')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    for i, net in enumerate(net_types):
        times = [r['time_seconds'] / 60 for r in sorted(by_net[net],
                 key=lambda x: x['positive_weight'])]
        positions = [x + i * bar_w for x in range(len(pw_vals))]
        ax.bar(positions, times, bar_w, label=net, color=COLORS.get(net, 'gray'))
    ax.set_xticks([x + bar_w for x in range(len(pw_vals))])
    ax.set_xticklabels([f"pw={p}" for p in pw_vals])
    ax.set_ylabel('Training Time (min)')
    ax.set_title('Training Time')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    for r in grid_results:
        net = r['network_type']
        ax.scatter(r['time_seconds'] / 60, r['best_iou'], s=120,
                   color=COLORS.get(net, 'gray'), alpha=0.8,
                   edgecolors='black', linewidth=0.5)
        ax.annotate(f"pw={r['positive_weight']}", fontsize=7,
                    xy=(r['time_seconds'] / 60, r['best_iou']),
                    xytext=(4, 4), textcoords='offset points')
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS[n], markersize=10, label=n)
               for n in net_types]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('Training Time (min)')
    ax.set_ylabel('Best IoU')
    ax.set_title('Speed vs Accuracy')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Grid Search Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grid_search_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Plots saved to {output_dir}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='GuidedBox Grid Search - PW + Network Variations')
    parser.add_argument('--config', type=str, default='guidedbox_config.yaml')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='guided_box_ijmond/grid_search_pw')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Fixed hyper-params
    config['learning_rate'] = 0.0001
    config['ema_alpha'] = 0.9
    config['img_size'] = 512

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    PW_VALUES = [1, 1.5, 2, 2.5, 3]
    total_combos = len(NETWORK_TYPES) * len(PW_VALUES)

    logging.info("=" * 70)
    logging.info("Grid Search: Network Variations x Positive Weight")
    logging.info("=" * 70)
    logging.info(f"Fixed -- lr: {config['learning_rate']}, "
                 f"ema: {config['ema_alpha']}, img_size: {config['img_size']}")
    logging.info(f"Networks:  {NETWORK_TYPES}")
    logging.info(f"PW values: {PW_VALUES}")
    logging.info(f"Total combinations: {total_combos}")
    logging.info(f"Epochs per run: {args.epochs}")
    logging.info(f"Device: {device}")
    logging.info(f"Output: {args.output_dir}")
    logging.info("=" * 70)

    train_records = load_records(config)

    grid_results = []
    run_idx = 0
    total_start = time.time()

    for net_type in NETWORK_TYPES:
        for pw in PW_VALUES:
            run_idx += 1
            logging.info(f"\n{'='*70}")
            logging.info(f"[{run_idx}/{total_combos}] "
                         f"network={net_type}  pw={pw}")
            logging.info(f"{'='*70}")

            params = {'network_type': net_type, 'positive_weight': pw}
            result = train_single_run(params, train_records, config,
                                      args.epochs, args.output_dir, device)
            grid_results.append(result)

            with open(os.path.join(args.output_dir, 'grid_results.json'), 'w') as f:
                json.dump(grid_results, f, indent=2)

    total_elapsed = time.time() - total_start

    logging.info(f"\n{'='*70}")
    logging.info(f"GRID SEARCH COMPLETE -- {len(grid_results)} models "
                 f"in {total_elapsed/60:.1f} min")
    logging.info(f"{'='*70}")

    sorted_results = sorted(grid_results, key=lambda r: r['best_iou'],
                            reverse=True)
    logging.info(f"\n{'Rank':<5} {'Run':<35} {'Net':<20} "
                 f"{'PW':<6} {'Params':>12} {'Best IoU':<10} {'Time(s)':<8}")
    logging.info("-" * 100)
    for rank, r in enumerate(sorted_results, 1):
        logging.info(
            f"{rank:<5} {r['run_name']:<35} {r['network_type']:<20} "
            f"{r['positive_weight']:<6} {r['n_params']:>12,} "
            f"{r['best_iou']:<10.4f} {r['time_seconds']:<8.1f}"
        )

    best = sorted_results[0]
    logging.info(f"\n{'='*70}")
    logging.info(f"BEST MODEL: {best['run_name']}")
    logging.info(f"  Network:    {best['network_type']}")
    logging.info(f"  PW:         {best['positive_weight']}")
    logging.info(f"  Parameters: {best['n_params']:,}")
    logging.info(f"  Best IoU:   {best['best_iou']:.4f}")
    logging.info(f"  Saved at:   {best['save_path']}")
    logging.info(f"{'='*70}")

    for net in NETWORK_TYPES:
        net_runs = [r for r in grid_results if r['network_type'] == net]
        avg_iou = np.mean([r['best_iou'] for r in net_runs])
        best_net = max(net_runs, key=lambda r: r['best_iou'])
        avg_time = np.mean([r['time_seconds'] for r in net_runs])
        logging.info(f"\n  {net}:")
        logging.info(f"    Best IoU:  {best_net['best_iou']:.4f} "
                     f"(pw={best_net['positive_weight']})")
        logging.info(f"    Avg IoU:   {avg_iou:.4f}")
        logging.info(f"    Avg Time:  {avg_time:.1f}s ({avg_time/60:.1f} min)")

    plot_results(grid_results, args.output_dir)
    logging.info(f"\nAll outputs in: {args.output_dir}")
    logging.info("Done!")


if __name__ == '__main__':
    main()
