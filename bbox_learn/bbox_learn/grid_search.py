"""
Grid Search for GuidedBox Model on IJmond Bbox Dataset
=======================================================
Standalone script for running on a cluster (GPU or CPU).

Usage:
    python grid_search.py
    python grid_search.py --config guidedbox_config.yaml
    python grid_search.py --epochs 20 --device cuda
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
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import yaml
import json
import os
import argparse
import logging
import time
from itertools import product
from datetime import datetime

# ============================================================
# Logging Setup
# ============================================================
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"grid_search_{timestamp}.log")

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
    """
    Loads .npy images and creates pseudo masks from bounding box annotations.
    """
    def __init__(self, records, img_npy_dir, img_size=256):
        self.records = records
        self.img_npy_dir = img_npy_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load image from .npy file
        img_path = os.path.join(self.img_npy_dir, f"{record['id']}.npy")
        image = np.load(img_path)  # (H, W, 3) uint8

        # Get original image dimensions and bbox
        h_img, w_img = record['h_image'], record['w_image']
        x_bbox, y_bbox = record['x_bbox'], record['y_bbox']
        w_bbox, h_bbox = record['w_bbox'], record['h_bbox']

        # The npy image might be a different size than h_image/w_image
        actual_h, actual_w = image.shape[:2]
        scale_x = actual_w / w_img
        scale_y = actual_h / h_img

        # Scale bbox to actual image coordinates
        x_bbox_scaled = int(x_bbox * scale_x)
        y_bbox_scaled = int(y_bbox * scale_y)
        w_bbox_scaled = int(w_bbox * scale_x)
        h_bbox_scaled = int(h_bbox * scale_y)

        # Create pseudo mask from bounding box (binary: 1 inside box, 0 outside)
        pseudo_mask = np.zeros((actual_h, actual_w), dtype=np.float32)
        x1 = max(0, x_bbox_scaled)
        y1 = max(0, y_bbox_scaled)
        x2 = min(actual_w, x_bbox_scaled + w_bbox_scaled)
        y2 = min(actual_h, y_bbox_scaled + h_bbox_scaled)
        pseudo_mask[y1:y2, x1:x2] = 1.0

        # Resize image and mask
        image_pil = Image.fromarray(image).resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_pil = Image.fromarray((pseudo_mask * 255).astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        # Convert to tensors
        image_tensor = TF.to_tensor(image_pil)  # [3, H, W], [0, 1]
        mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        # Normalize image (ImageNet stats for ResNet50 backbone)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Normalized bbox [x, y, w, h] in [0, 1] range
        box_tensor = torch.tensor([
            x_bbox / w_img,
            y_bbox / h_img,
            w_bbox / w_img,
            h_bbox / h_img,
        ], dtype=torch.float32)

        return image_tensor, box_tensor, mask_tensor


# ============================================================
# Model
# ============================================================
class GuidedBoxModel(nn.Module):
    def __init__(self, config):
        super(GuidedBoxModel, self).__init__()

        # Load ResNet50 and extract only convolutional feature layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # [B, 256, H/4, W/4]
            resnet.layer2,   # [B, 512, H/8, W/8]
            resnet.layer3,   # [B, 1024, H/16, W/16]
            resnet.layer4,   # [B, 2048, H/32, W/32]
        )

        self.teacher = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, config['num_classes'], kernel_size=1),
        )
        self.student = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, config['num_classes'], kernel_size=1),
        )
        self.alpha = config['ema_alpha']

    def forward(self, images, positive_weight, boxes=None, masks=None):
        features = self.backbone(images)
        teacher_output = self.teacher(features)
        student_output = self.student(features)

        # Upsample predictions to original image size
        teacher_output = F.interpolate(teacher_output, size=images.shape[2:], mode='bilinear', align_corners=False)
        student_output = F.interpolate(student_output, size=images.shape[2:], mode='bilinear', align_corners=False)

        if self.training:
            loss = self.compute_loss(teacher_output, student_output, boxes=boxes, masks=masks, positive_weight=positive_weight)
            return loss
        else:
            return student_output

    def compute_loss(self, teacher_output, student_output, boxes, masks, positive_weight):
        conf_scores = self.compute_confidence_scores(teacher_output, student_output)
        mask_loss = self.robust_pseudo_mask_loss(student_output, masks, conf_scores, positive_weight=positive_weight)
        box_loss = F.mse_loss(teacher_output, student_output)
        return box_loss + mask_loss

    def robust_pseudo_mask_loss(self, preds, pseudo_masks, conf_scores, positive_weight=5):
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

            fg_loss = 0.0
            bg_loss = 0.0
            if neighbors.sum() > 0:
                fg_loss = -torch.mean(torch.log(mask_pred[neighbors]))
            if (~neighbors).sum() > 0:
                bg_loss = -torch.mean(torch.log(1 - mask_pred[~neighbors]))

            affinity_loss += fg_loss + bg_loss
        return affinity_loss / preds.size(0)

    def compute_confidence_scores(self, teacher_output, student_output):
        return torch.sigmoid(F.cosine_similarity(teacher_output, student_output))

    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
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
    """Load and filter IJmond bbox labels."""
    with open(config['bbox_labels_path'], 'r') as f:
        bbox_data = json.load(f)

    all_records = bbox_data['data']
    logging.info(f"Total records in JSON: {len(all_records)}")

    # Get available npy image IDs
    npy_ids = set(
        f.replace('.npy', '') for f in os.listdir(config['img_npy_path']) if f.endswith('.npy')
    )
    logging.info(f"Total npy images: {len(npy_ids)}")

    matched_records = [r for r in all_records if str(r['id']) in npy_ids]
    logging.info(f"Records with matching npy files: {len(matched_records)}")

    SMOKE_STATES = {3, 4, 9, 10, 11, 13, 15}
    NO_SMOKE_STATES = {5, 12, 14}

    smoke_records = [r for r in matched_records if r['label_state'] in SMOKE_STATES]
    no_smoke_records = [r for r in matched_records if r['label_state'] in NO_SMOKE_STATES]

    logging.info(f"Smoke records (with bbox): {len(smoke_records)}")
    logging.info(f"No-smoke records: {len(no_smoke_records)}")

    train_records = smoke_records + no_smoke_records
    logging.info(f"Using {len(train_records)} records for training")
    return train_records


# ============================================================
# Single Run
# ============================================================
def train_single_run(params, train_records, config, grid_epochs, grid_save_dir, device):
    """Train a single hyperparameter combination and return results dict."""
    run_name = (
        f"lr{params['learning_rate']}_ema{params['ema_alpha']}"
        f"_img{params['img_size']}_pw{params['positive_weight']}"
    )
    logging.info(f"Starting run: {run_name}")
    start_time = time.time()

    # Build dataset and loaders
    run_dataset = IJmondBboxDataset(train_records, config['img_npy_path'], img_size=params['img_size'])
    run_val_size = int(len(run_dataset) * 0.2)
    run_train_size = len(run_dataset) - run_val_size
    run_train_ds, run_val_ds = random_split(
        run_dataset, [run_train_size, run_val_size],
        generator=torch.Generator().manual_seed(42),
    )
    run_train_loader = DataLoader(run_train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))
    run_val_loader = DataLoader(run_val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

    # Build model
    run_config = config.copy()
    run_config['ema_alpha'] = params['ema_alpha']
    run_model = GuidedBoxModel(run_config).to(device)
    run_optimizer = torch.optim.Adam(run_model.parameters(), lr=params['learning_rate'])

    # Training
    run_train_losses = []
    run_val_losses = []
    run_val_ious = []
    best_iou = 0.0

    for epoch in range(grid_epochs):
        # --- Train ---
        run_model.train()
        epoch_loss = 0.0
        for images, boxes, masks in run_train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            masks = masks.to(device)

            loss = run_model(images, boxes=boxes, masks=masks, positive_weight=params['positive_weight'])
            run_optimizer.zero_grad()
            loss.backward()
            run_optimizer.step()
            run_model.update_teacher()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(run_train_loader)
        run_train_losses.append(avg_train_loss)

        # --- Validate ---
        run_model.eval()
        epoch_val_loss = 0.0
        epoch_iou = 0.0
        with torch.no_grad():
            for images, boxes, masks in run_val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = run_model(images, params['positive_weight'])
                val_loss = F.binary_cross_entropy_with_logits(
                    outputs, masks,
                    pos_weight=torch.tensor(float(params['positive_weight']), device=device),
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
            f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, IoU={avg_iou:.4f}"
        )

    elapsed = time.time() - start_time

    # Save model checkpoint
    save_path = os.path.join(grid_save_dir, f'{run_name}.pth')
    torch.save({
        'model_state_dict': run_model.state_dict(),
        'params': params,
        'train_losses': run_train_losses,
        'val_losses': run_val_losses,
        'val_ious': run_val_ious,
        'best_iou': best_iou,
        'final_iou': run_val_ious[-1],
        'final_val_loss': run_val_losses[-1],
    }, save_path)

    result = {
        'run_name': run_name,
        'learning_rate': params['learning_rate'],
        'ema_alpha': params['ema_alpha'],
        'img_size': params['img_size'],
        'positive_weight': params['positive_weight'],
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

    logging.info(f"  [{run_name}] Best IoU: {best_iou:.4f} | Time: {elapsed:.1f}s | Saved: {save_path}")
    return result


# ============================================================
# Plotting
# ============================================================
def plot_results(grid_results, output_dir):
    """Save comparison plots to disk."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. IoU curves
    ax = axes[0, 0]
    for r in grid_results:
        ax.plot(range(1, len(r['val_ious']) + 1), r['val_ious'], label=r['run_name'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation IoU')
    ax.set_title('Validation IoU Across All Runs')
    ax.legend(fontsize=5, ncol=2, loc='lower right')
    ax.grid(True)

    # 2. Val loss curves
    ax = axes[0, 1]
    for r in grid_results:
        ax.plot(range(1, len(r['val_losses']) + 1), r['val_losses'], label=r['run_name'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Across All Runs')
    ax.legend(fontsize=5, ncol=2, loc='upper right')
    ax.grid(True)

    # 3. Best IoU by learning rate
    ax = axes[1, 0]
    lr_groups = {}
    for r in grid_results:
        lr_groups.setdefault(r['learning_rate'], []).append(r['best_iou'])
    lr_labels = [str(lr) for lr in sorted(lr_groups.keys())]
    lr_means = [np.mean(lr_groups[float(lr)]) for lr in lr_labels]
    lr_maxs = [np.max(lr_groups[float(lr)]) for lr in lr_labels]
    x = range(len(lr_labels))
    ax.bar([i - 0.15 for i in x], lr_means, 0.3, label='Mean IoU', color='steelblue')
    ax.bar([i + 0.15 for i in x], lr_maxs, 0.3, label='Max IoU', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('IoU')
    ax.set_title('Best IoU by Learning Rate')
    ax.legend()
    ax.grid(axis='y')

    # 4. Best IoU by EMA alpha
    ax = axes[1, 1]
    ema_groups = {}
    for r in grid_results:
        ema_groups.setdefault(r['ema_alpha'], []).append(r['best_iou'])
    ema_labels = [str(e) for e in sorted(ema_groups.keys())]
    ema_means = [np.mean(ema_groups[float(e)]) for e in ema_labels]
    ema_maxs = [np.max(ema_groups[float(e)]) for e in ema_labels]
    x = range(len(ema_labels))
    ax.bar([i - 0.15 for i in x], ema_means, 0.3, label='Mean IoU', color='steelblue')
    ax.bar([i + 0.15 for i in x], ema_maxs, 0.3, label='Max IoU', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels(ema_labels)
    ax.set_xlabel('EMA Alpha')
    ax.set_ylabel('IoU')
    ax.set_title('Best IoU by EMA Alpha')
    ax.legend()
    ax.grid(axis='y')

    plt.suptitle('Grid Search Results Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'grid_search_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Comparison plot saved to: {plot_path}")

    # Best model visualization
    best_result = max(grid_results, key=lambda r: r['best_iou'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(best_result['train_losses'], label='Train Loss', marker='o')
    axes[0].plot(best_result['val_losses'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f"Best Model Loss: {best_result['run_name']}")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(best_result['val_ious'], label='Val IoU', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title(f"Best Model IoU (Best: {best_result['best_iou']:.4f})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    best_plot_path = os.path.join(output_dir, 'best_model_curves.png')
    plt.savefig(best_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Best model plot saved to: {best_plot_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GuidedBox Grid Search on IJmond Bbox Dataset")
    parser.add_argument('--config', type=str, default='guidedbox_config.yaml', help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per grid search combination')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda or cpu (auto-detect if omitted)')
    parser.add_argument('--output_dir', type=str, default='guided_box_ijmond/grid_search', help='Directory to save models and plots')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader num_workers')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = str(device)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    logging.info("=" * 70)
    logging.info("GuidedBox Grid Search")
    logging.info("=" * 70)
    logging.info(f"Config: {args.config}")
    logging.info(f"Device: {device}")
    logging.info(f"Epochs per run: {args.epochs}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Log file: {log_file}")

    # Load data
    train_records = load_records(config)

    # Parameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.0002, 0.0005],
        'ema_alpha': [0.85, 0.9, 0.95],
        'img_size': [128, 256, 512],
        'positive_weight': [3, 5, 7, 9],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    logging.info(f"Total combinations: {len(all_combinations)}")
    for i, combo in enumerate(all_combinations):
        logging.info(f"  [{i+1}] {combo}")

    # Run grid search
    grid_results = []
    total_start = time.time()

    for run_idx, params in enumerate(all_combinations):
        logging.info(f"\n{'='*70}")
        logging.info(f"[{run_idx+1}/{len(all_combinations)}]")
        logging.info(f"{'='*70}")

        result = train_single_run(
            params=params,
            train_records=train_records,
            config=config,
            grid_epochs=args.epochs,
            grid_save_dir=args.output_dir,
            device=device,
        )
        grid_results.append(result)

        # Save intermediate results after each run (in case of crash)
        results_path = os.path.join(args.output_dir, 'grid_results.json')
        serializable = []
        for r in grid_results:
            sr = {k: v for k, v in r.items() if k not in ('train_losses', 'val_losses', 'val_ious')}
            sr['train_losses'] = r['train_losses']
            sr['val_losses'] = r['val_losses']
            sr['val_ious'] = r['val_ious']
            serializable.append(sr)
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    total_elapsed = time.time() - total_start

    # Summary
    logging.info(f"\n{'='*70}")
    logging.info(f"Grid search complete! {len(grid_results)} models trained.")
    logging.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logging.info(f"{'='*70}")

    # Sort and print results
    sorted_results = sorted(grid_results, key=lambda r: r['best_iou'], reverse=True)
    logging.info("\nRanked Results (by Best IoU):")
    logging.info(f"{'Rank':<5} {'Run':<55} {'Best IoU':<10} {'Final IoU':<10} {'Time(s)':<8}")
    logging.info("-" * 90)
    for rank, r in enumerate(sorted_results, 1):
        logging.info(
            f"{rank:<5} {r['run_name']:<55} {r['best_iou']:<10.4f} "
            f"{r['final_iou']:<10.4f} {r['time_seconds']:<8.1f}"
        )

    best = sorted_results[0]
    logging.info(f"\n🏆 BEST MODEL: {best['run_name']}")
    logging.info(f"   Best IoU: {best['best_iou']:.4f}")
    logging.info(f"   LR={best['learning_rate']}, EMA={best['ema_alpha']}, "
                 f"Img={best['img_size']}, PW={best['positive_weight']}")
    logging.info(f"   Saved at: {best['save_path']}")

    # Generate plots
    plot_results(grid_results, args.output_dir)

    logging.info(f"\nAll results saved to: {args.output_dir}")
    logging.info("Done!")


if __name__ == '__main__':
    main()
