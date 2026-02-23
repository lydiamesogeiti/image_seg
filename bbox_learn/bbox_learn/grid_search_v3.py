"""
Grid Search V3 - Comprehensive GuidedBox Model Search
======================================================
Search dimensions:
  - network_type:    [resnet50_3layer, resnet50_4layer]
  - epochs:          [10, 20, 50, 100]
  - learning_rate:   [0.0001, 0.00005, 0.00001]
  - positive_weight: [1.5, 2.5, 3]

Fixed: ema=0.9, img_size=512

Total: 2 x 4 x 3 x 3 = 72 combinations

Key improvements over v1:
  - Cosine annealing LR scheduler for stability
  - Average confidence score tracked per epoch (teacher-student agreement)
  - Backbone frozen for first 3 epochs to prevent feature destruction
  - Dice score tracked alongside IoU
  - Best model saved based on IoU (early stopping checkpoint)

Usage:
    python grid_search_v3.py --config guidedbox_config.yaml --device cuda
    python grid_search_v3.py --config guidedbox_config.yaml --device cuda --output_dir guided_box_ijmond/grid_search_v3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    log_file = os.path.join(log_dir, f"grid_search_v3_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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

        image_pil = Image.fromarray(image).resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask_pil = Image.fromarray((pseudo_mask * 255).astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST)

        image_tensor = TF.to_tensor(image_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        box_tensor = torch.tensor([
            x_bbox / w_img, y_bbox / h_img,
            w_bbox / w_img, h_bbox / h_img,
        ], dtype=torch.float32)

        return image_tensor, box_tensor, mask_tensor


# ============================================================
# Model - ResNet50 3-layer and 4-layer decoders
# ============================================================
NETWORK_TYPES = ['resnet50_3layer', 'resnet50_4layer']


class GuidedBoxModel(nn.Module):
    """
    Teacher-student model with selectable decoder depth.

    resnet50_3layer : ResNet50, decoder 2048->1024->512->1        (original)
    resnet50_4layer : ResNet50, decoder 2048->1024->512->256->1   (deeper)
    """

    def __init__(self, config, network_type='resnet50_3layer'):
        super().__init__()
        self.network_type = network_type
        self.alpha = config['ema_alpha']
        num_classes = config['num_classes']

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        if network_type == 'resnet50_3layer':
            ch = [2048, 1024, 512]
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
            layers.append(nn.Conv2d(
                channel_list[i], channel_list[i + 1],
                kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channel_list[-1], num_classes, kernel_size=1))
        return nn.Sequential(*layers)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, positive_weight, boxes=None, masks=None,
                return_confidence=False):
        features = self.backbone(images)
        teacher_output = self.teacher(features)
        student_output = self.student(features)

        teacher_output = F.interpolate(
            teacher_output, size=images.shape[2:],
            mode='bilinear', align_corners=False)
        student_output = F.interpolate(
            student_output, size=images.shape[2:],
            mode='bilinear', align_corners=False)

        if self.training:
            return self.compute_loss(
                teacher_output, student_output,
                boxes=boxes, masks=masks,
                positive_weight=positive_weight)
        else:
            conf = self.compute_confidence_scores(
                teacher_output, student_output)
            if return_confidence:
                return student_output, conf
            return student_output, conf

    def compute_loss(self, teacher_output, student_output, boxes, masks,
                     positive_weight):
        conf_scores = self.compute_confidence_scores(
            teacher_output, student_output)
        mask_loss = self.robust_pseudo_mask_loss(
            student_output, masks, conf_scores,
            positive_weight=positive_weight)
        box_loss = F.mse_loss(teacher_output, student_output)
        return box_loss + mask_loss

    def robust_pseudo_mask_loss(self, preds, pseudo_masks, conf_scores,
                                positive_weight=5):
        pixel_loss = F.binary_cross_entropy_with_logits(
            preds, pseudo_masks,
            pos_weight=torch.tensor(
                float(positive_weight), device=preds.device))
        affinity_loss = self.enhanced_mask_affinity_loss(preds, pseudo_masks)
        return torch.mean(conf_scores * (0.4 * pixel_loss + 0.1 * affinity_loss))

    def enhanced_mask_affinity_loss(self, preds, pseudo_masks):
        eps = 1e-7
        affinity_loss = 0.0
        for i in range(preds.size(0)):
            mask_pred = torch.sigmoid(preds[i])
            mask_pred = torch.clamp(mask_pred, eps, 1 - eps)
            neighbors = F.max_pool2d(
                mask_pred.unsqueeze(0), kernel_size=3,
                stride=1, padding=1
            ).squeeze(0) > 0.45
            fg_loss = (-torch.mean(torch.log(mask_pred[neighbors]))
                       if neighbors.sum() > 0 else 0.0)
            bg_loss = (-torch.mean(torch.log(1 - mask_pred[~neighbors]))
                       if (~neighbors).sum() > 0 else 0.0)
            affinity_loss += fg_loss + bg_loss
        return affinity_loss / preds.size(0)

    def compute_confidence_scores(self, teacher_output, student_output):
        return torch.sigmoid(
            F.cosine_similarity(teacher_output, student_output))

    def update_teacher(self):
        for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters()):
            t_param.data = (self.alpha * t_param.data
                            + (1 - self.alpha) * s_param.data)


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
        if f.endswith('.npy'))
    logging.info(f"Total npy images: {len(npy_ids)}")

    matched_records = [r for r in all_records if str(r['id']) in npy_ids]
    logging.info(f"Records with matching npy: {len(matched_records)}")

    SMOKE_STATES = {3, 4, 9, 10, 11, 13, 15}
    NO_SMOKE_STATES = {5, 12, 14}

    smoke_records = [r for r in matched_records
                     if r['label_state'] in SMOKE_STATES]
    no_smoke_records = [r for r in matched_records
                        if r['label_state'] in NO_SMOKE_STATES]

    logging.info(f"Smoke: {len(smoke_records)}, "
                 f"No-smoke: {len(no_smoke_records)}")

    train_records = smoke_records + no_smoke_records
    logging.info(f"Total training records: {len(train_records)}")
    return train_records


# ============================================================
# Single training run
# ============================================================
FREEZE_EPOCHS = 3  # freeze backbone for first N epochs


def train_single_run(params, train_records, config,
                     grid_save_dir, device, args):
    net_type = params['network_type']
    pw = params['positive_weight']
    lr = params['learning_rate']
    n_epochs = params['epochs']
    run_name = f"{net_type}_ep{n_epochs}_lr{lr}_pw{pw}"
    logging.info(f"Starting run: {run_name}")
    start_time = time.time()

    # -- data loaders --
    run_dataset = IJmondBboxDataset(
        train_records, config['img_npy_path'],
        img_size=config['img_size'])
    val_size = int(len(run_dataset) * 0.2)
    train_size = len(run_dataset) - val_size
    run_train_ds, run_val_ds = random_split(
        run_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    run_train_loader = DataLoader(
        run_train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'))
    run_val_loader = DataLoader(
        run_val_ds, batch_size=config['batch_size'],
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'))

    # -- model --
    run_model = GuidedBoxModel(config, network_type=net_type).to(device)
    n_params = sum(p.numel() for p in run_model.parameters())
    logging.info(f"  Params: {n_params:,}  |  Epochs: {n_epochs}  |  "
                 f"LR: {lr}  |  PW: {pw}")

    run_optimizer = torch.optim.Adam(run_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(run_optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # -- tracking --
    run_train_losses = []
    run_val_losses = []
    run_val_ious = []
    run_val_dices = []
    run_val_confidences = []
    run_lr_history = []
    best_iou = 0.0
    best_epoch = 0
    best_state_dict = None

    # -- freeze backbone initially --
    run_model.freeze_backbone()
    backbone_frozen = True

    for epoch in range(n_epochs):
        # Unfreeze backbone after FREEZE_EPOCHS
        if backbone_frozen and epoch >= FREEZE_EPOCHS:
            run_model.unfreeze_backbone()
            backbone_frozen = False
            logging.info(f"  [{run_name}] Backbone unfrozen at epoch {epoch+1}")

        current_lr = run_optimizer.param_groups[0]['lr']
        run_lr_history.append(current_lr)

        # -- train --
        run_model.train()
        epoch_loss = 0.0
        for images, boxes, masks in run_train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            masks = masks.to(device)
            loss = run_model(images, positive_weight=pw,
                             boxes=boxes, masks=masks)
            run_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(run_model.parameters(), max_norm=1.0)
            run_optimizer.step()
            run_model.update_teacher()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(run_train_loader)
        run_train_losses.append(avg_train_loss)

        # -- validate --
        run_model.eval()
        epoch_val_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_conf = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, boxes, masks in run_val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs, conf_scores = run_model(images, pw)
                val_loss = F.binary_cross_entropy_with_logits(
                    outputs, masks,
                    pos_weight=torch.tensor(float(pw), device=device))
                epoch_val_loss += val_loss.item()
                epoch_iou += calculate_iou(outputs, masks)
                epoch_dice += calculate_dice(outputs, masks)
                epoch_conf += conf_scores.mean().item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / n_val_batches
        avg_iou = epoch_iou / n_val_batches
        avg_dice = epoch_dice / n_val_batches
        avg_conf = epoch_conf / n_val_batches

        run_val_losses.append(avg_val_loss)
        run_val_ious.append(avg_iou)
        run_val_dices.append(avg_dice)
        run_val_confidences.append(avg_conf)

        if avg_iou > best_iou:
            best_iou = avg_iou
            best_epoch = epoch + 1
            best_state_dict = {
                k: v.cpu().clone() for k, v in run_model.state_dict().items()
            }

        scheduler.step()

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == 0:
            logging.info(
                f"  [{run_name}] Ep {epoch+1}/{n_epochs}: "
                f"Train={avg_train_loss:.4f} Val={avg_val_loss:.4f} "
                f"IoU={avg_iou:.4f} Dice={avg_dice:.4f} "
                f"Conf={avg_conf:.4f} LR={current_lr:.6f}")

    elapsed = time.time() - start_time

    # -- save best model --
    save_path = os.path.join(grid_save_dir, f'{run_name}.pth')
    torch.save({
        'model_state_dict': best_state_dict,
        'params': params,
        'network_type': net_type,
        'best_epoch': best_epoch,
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'val_ious': run_val_ious,
        'val_dices': run_val_dices,
        'val_confidences': run_val_confidences,
        'lr_history': run_lr_history,
        'best_iou': best_iou,
        'final_iou': run_val_ious[-1],
        'final_val_loss': run_val_losses[-1],
    }, save_path)

    result = {
        'run_name': run_name,
        'network_type': net_type,
        'epochs': n_epochs,
        'learning_rate': lr,
        'positive_weight': pw,
        'n_params': n_params,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'final_iou': run_val_ious[-1],
        'final_dice': run_val_dices[-1],
        'final_confidence': run_val_confidences[-1],
        'best_confidence': max(run_val_confidences),
        'avg_confidence': float(np.mean(run_val_confidences)),
        'final_val_loss': run_val_losses[-1],
        'final_train_loss': run_train_losses[-1],
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'val_ious': run_val_ious,
        'val_dices': run_val_dices,
        'val_confidences': run_val_confidences,
        'lr_history': run_lr_history,
        'time_seconds': elapsed,
        'save_path': save_path,
    }

    logging.info(
        f"  [{run_name}] Done: Best IoU={best_iou:.4f} (ep {best_epoch}) | "
        f"Final Conf={run_val_confidences[-1]:.4f} | "
        f"Time={elapsed:.1f}s | {save_path}")
    return result


# ============================================================
# Plotting
# ============================================================
NET_COLORS = {
    'resnet50_3layer': '#1f77b4',
    'resnet50_4layer': '#d62728',
}
EPOCH_MARKERS = {10: 's', 20: 'o', 50: '^', 100: 'D'}


def plot_results(grid_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. IoU curves grouped by (network, lr) ----
    combos = {}
    for r in grid_results:
        key = (r['network_type'], r['learning_rate'])
        combos.setdefault(key, []).append(r)

    n_combos = len(combos)
    cols = min(4, n_combos)
    rows = (n_combos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    for idx, ((net, lr_val), runs) in enumerate(sorted(combos.items())):
        ax = axes[idx // cols][idx % cols]
        for r in sorted(runs, key=lambda x: (x['epochs'], x['positive_weight'])):
            label = f"ep={r['epochs']},pw={r['positive_weight']}"
            ax.plot(range(1, len(r['val_ious']) + 1), r['val_ious'],
                    label=label, alpha=0.7, marker='o', markersize=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val IoU')
        ax.set_title(f"{net}\nlr={lr_val}", fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    for idx in range(len(combos), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.suptitle('Validation IoU Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 2. Confidence curves grouped same way ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    for idx, ((net, lr_val), runs) in enumerate(sorted(combos.items())):
        ax = axes[idx // cols][idx % cols]
        for r in sorted(runs, key=lambda x: (x['epochs'], x['positive_weight'])):
            label = f"ep={r['epochs']},pw={r['positive_weight']}"
            ax.plot(range(1, len(r['val_confidences']) + 1),
                    r['val_confidences'],
                    label=label, alpha=0.7, marker='o', markersize=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Confidence')
        ax.set_title(f"{net}\nlr={lr_val}", fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    for idx in range(len(combos), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.suptitle('Confidence Score Curves (teacher-student agreement)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 3. Summary: Best IoU bar chart by epochs ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, net in enumerate(NETWORK_TYPES):
        ax = axes[ax_idx]
        net_runs = [r for r in grid_results if r['network_type'] == net]
        epoch_vals = sorted(set(r['epochs'] for r in net_runs))
        lr_vals = sorted(set(r['learning_rate'] for r in net_runs))
        bar_w = 0.8 / len(lr_vals)
        for li, lr_val in enumerate(lr_vals):
            ious = []
            for ep in epoch_vals:
                match = [r for r in net_runs
                         if r['learning_rate'] == lr_val and r['epochs'] == ep]
                best_of = max(match, key=lambda x: x['best_iou']) if match else None
                ious.append(best_of['best_iou'] if best_of else 0)
            positions = [x + li * bar_w for x in range(len(epoch_vals))]
            ax.bar(positions, ious, bar_w, label=f"lr={lr_val}", alpha=0.85)
        ax.set_xticks([x + bar_w * (len(lr_vals) - 1) / 2
                       for x in range(len(epoch_vals))])
        ax.set_xticklabels([f"{ep} ep" for ep in epoch_vals])
        ax.set_ylabel('Best IoU (over pw)')
        ax.set_title(net, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Best IoU by Epochs and Learning Rate',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_iou_by_epochs_lr.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 4. IoU vs Confidence scatter ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in grid_results:
        net = r['network_type']
        ep = r['epochs']
        color = NET_COLORS.get(net, 'gray')
        marker = EPOCH_MARKERS.get(ep, 'o')
        ax.scatter(r['final_confidence'], r['best_iou'],
                   s=80, color=color, marker=marker, alpha=0.7,
                   edgecolors='black', linewidth=0.4)
    net_handles = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=NET_COLORS[n], markersize=10, label=n)
                   for n in NETWORK_TYPES]
    ep_handles = [plt.Line2D([0], [0], marker=EPOCH_MARKERS[ep], color='gray',
                  markersize=8, linestyle='', label=f"{ep} epochs")
                  for ep in sorted(EPOCH_MARKERS.keys())]
    ax.legend(handles=net_handles + ep_handles, fontsize=8, ncol=2)
    ax.set_xlabel('Final Avg Confidence')
    ax.set_ylabel('Best IoU')
    ax.set_title('IoU vs Confidence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_vs_confidence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 5. Speed vs accuracy ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in grid_results:
        net = r['network_type']
        ep = r['epochs']
        color = NET_COLORS.get(net, 'gray')
        marker = EPOCH_MARKERS.get(ep, 'o')
        ax.scatter(r['time_seconds'] / 60, r['best_iou'],
                   s=80, color=color, marker=marker, alpha=0.7,
                   edgecolors='black', linewidth=0.4)
    ax.legend(handles=net_handles + ep_handles, fontsize=8, ncol=2)
    ax.set_xlabel('Training Time (min)')
    ax.set_ylabel('Best IoU')
    ax.set_title('Speed vs Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_vs_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Plots saved to {output_dir}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='GuidedBox Grid Search V3')
    parser.add_argument('--config', type=str,
                        default='guidedbox_config.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='guided_box_ijmond/grid_search_v3')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader num_workers')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Fixed params
    config['ema_alpha'] = 0.9
    config['img_size'] = 512

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    # --- Grid dimensions ---
    EPOCH_VALUES = [10, 20, 50, 100]
    LR_VALUES = [0.0001, 0.00005, 0.00001]
    PW_VALUES = [1.5, 2.5, 3]
    

    total_combos = (len(NETWORK_TYPES) * len(EPOCH_VALUES)
                    * len(LR_VALUES) * len(PW_VALUES))

    logging.info("=" * 70)
    logging.info("Grid Search V3: Comprehensive Search")
    logging.info("=" * 70)
    logging.info(f"Fixed:   ema={config['ema_alpha']}, "
                 f"img_size={config['img_size']}")
    logging.info(f"Networks:  {NETWORK_TYPES}")
    logging.info(f"Epochs:    {EPOCH_VALUES}")
    logging.info(f"LR:        {LR_VALUES}")
    logging.info(f"PW:        {PW_VALUES}")
    logging.info(f"Total combinations: {total_combos}")
    logging.info(f"Backbone freeze: first {FREEZE_EPOCHS} epochs")
    logging.info(f"LR scheduler: CosineAnnealingLR")
    logging.info(f"Gradient clipping: max_norm=1.0")
    logging.info(f"Device: {device}")
    logging.info(f"Output: {args.output_dir}")
    logging.info("=" * 70)

    train_records = load_records(config)
    train_records = train_records[:10]

    # -- check for existing results (crash recovery) --
    results_path = os.path.join(args.output_dir, 'grid_v3_results.json')
    grid_results = []
    completed_runs = set()
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            grid_results = json.load(f)
        completed_runs = {r['run_name'] for r in grid_results}
        logging.info(f"Resuming: {len(completed_runs)} runs already done")

    run_idx = len(completed_runs)
    total_start = time.time()

    for net_type in NETWORK_TYPES:
        for n_epochs in EPOCH_VALUES:
            for lr in LR_VALUES:
                for pw in PW_VALUES:
                    run_name = f"{net_type}_ep{n_epochs}_lr{lr}_pw{pw}"
                    if run_name in completed_runs:
                        continue

                    run_idx += 1
                    logging.info(f"\n{'='*70}")
                    logging.info(
                        f"[{run_idx}/{total_combos}] "
                        f"net={net_type}  ep={n_epochs}  "
                        f"lr={lr}  pw={pw}")
                    logging.info(f"{'='*70}")

                    config['learning_rate'] = lr
                    params = {
                        'network_type': net_type,
                        'epochs': n_epochs,
                        'learning_rate': lr,
                        'positive_weight': pw,
                    }
                    result = train_single_run(
                        params, train_records, config,
                        args.output_dir, device, args)
                    grid_results.append(result)

                    # save after each run (crash recovery)
                    with open(results_path, 'w') as f:
                        json.dump(grid_results, f, indent=2)

    total_elapsed = time.time() - total_start

    # ---- Summary ----
    logging.info(f"\n{'='*70}")
    logging.info(f"GRID SEARCH V3 COMPLETE: {len(grid_results)} models "
                 f"in {total_elapsed/60:.1f} min")
    logging.info(f"{'='*70}")

    sorted_results = sorted(grid_results, key=lambda r: r['best_iou'],
                            reverse=True)

    logging.info(f"\n{'Rank':<5} {'Run':<45} {'Net':<18} {'Ep':<5} "
                 f"{'LR':<10} {'PW':<5} {'BestIoU':<9} {'BestEp':<7} "
                 f"{'AvgConf':<9} {'Time':<8}")
    logging.info("-" * 130)
    for rank, r in enumerate(sorted_results[:20], 1):
        logging.info(
            f"{rank:<5} {r['run_name']:<45} "
            f"{r['network_type']:<18} {r['epochs']:<5} "
            f"{r['learning_rate']:<10} {r['positive_weight']:<5} "
            f"{r['best_iou']:<9.4f} {r['best_epoch']:<7} "
            f"{r['avg_confidence']:<9.4f} "
            f"{r['time_seconds']:<8.1f}")

    best = sorted_results[0]
    logging.info(f"\n{'='*70}")
    logging.info(f"BEST MODEL: {best['run_name']}")
    logging.info(f"  Network:      {best['network_type']}")
    logging.info(f"  Epochs:       {best['epochs']}")
    logging.info(f"  LR:           {best['learning_rate']}")
    logging.info(f"  PW:           {best['positive_weight']}")
    logging.info(f"  Params:       {best['n_params']:,}")
    logging.info(f"  Best IoU:     {best['best_iou']:.4f} "
                 f"(epoch {best['best_epoch']})")
    logging.info(f"  Final Dice:   {best['final_dice']:.4f}")
    logging.info(f"  Avg Conf:     {best['avg_confidence']:.4f}")
    logging.info(f"  Saved at:     {best['save_path']}")
    logging.info(f"{'='*70}")

    # Per-network summary
    for net in NETWORK_TYPES:
        net_runs = [r for r in grid_results if r['network_type'] == net]
        if not net_runs:
            continue
        best_net = max(net_runs, key=lambda r: r['best_iou'])
        avg_iou = np.mean([r['best_iou'] for r in net_runs])
        avg_conf = np.mean([r['avg_confidence'] for r in net_runs])
        avg_time = np.mean([r['time_seconds'] for r in net_runs])
        logging.info(f"\n  {net} ({len(net_runs)} runs):")
        logging.info(f"    Best:     IoU={best_net['best_iou']:.4f} "
                     f"(ep={best_net['epochs']}, lr={best_net['learning_rate']}, "
                     f"pw={best_net['positive_weight']})")
        logging.info(f"    Avg IoU:  {avg_iou:.4f}")
        logging.info(f"    Avg Conf: {avg_conf:.4f}")
        logging.info(f"    Avg Time: {avg_time/60:.1f} min")

    # Per-epoch summary
    for ep in sorted(set(r['epochs'] for r in grid_results)):
        ep_runs = [r for r in grid_results if r['epochs'] == ep]
        avg_iou = np.mean([r['best_iou'] for r in ep_runs])
        best_ep = max(ep_runs, key=lambda r: r['best_iou'])
        logging.info(f"\n  {ep} epochs ({len(ep_runs)} runs):")
        logging.info(f"    Best:    IoU={best_ep['best_iou']:.4f} "
                     f"({best_ep['run_name']})")
        logging.info(f"    Avg IoU: {avg_iou:.4f}")

    plot_results(grid_results, args.output_dir)
    logging.info(f"\nAll outputs in: {args.output_dir}")
    logging.info("Done!")


if __name__ == '__main__':
    main()
