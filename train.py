"""
Autoresearch training script for Assignment 2 (Landslide Detection).
Task: binary semantic segmentation on 14-channel Landslide4Sense patches.

Usage:
    uv run train.py
"""

import os
import time

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prepare import (
    IN_CHANNELS,
    NUM_CLASSES,
    build_dataloaders,
    evaluate_segmentation,
    find_best_threshold,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 45

LR = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

AMP_ENABLED = torch.cuda.is_available()
THRESHOLD_SEARCH_EVERY = 2
EARLY_STOP_PATIENCE = 10

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "best_model.pth"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg_w = self.mlp(self.avg_pool(x).view(b, c))
        max_w = self.mlp(self.max_pool(x).view(b, c))
        w = self.sigmoid(avg_w + max_w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class LandslideModel(nn.Module):
    """
    U-Net++ with EfficientNet-B4 encoder and CBAM refinement on decoder output.
    Supports 14-channel input and outputs a single segmentation logit map.
    """

    def __init__(self, in_channels=IN_CHANNELS, classes=NUM_CLASSES):
        super().__init__()
        self.core = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_attention_type="scse",
        )

        seg_head = self.core.segmentation_head
        head_in_channels = seg_head[0].in_channels

        self.core.segmentation_head = nn.Identity()
        self.cbam = CBAM(head_in_channels)
        self.seg_head = seg_head

    def forward(self, x):
        features = self.core(x)
        features = self.cbam(features)
        return self.seg_head(features)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = torch.sum(probs * targets)
        denom = probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_weight = torch.where(
            targets == 1.0,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        focal = alpha_weight * (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(
            logits, targets
        ) + self.dice_weight * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED
        ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def run_training():
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    print("=" * 72)
    print("Assignment 2 | Landslide Detection (Binary Segmentation)")
    print("Model: U-Net++ (EfficientNet-B4) + CBAM | Input channels: 14")
    print("Metric focus: F1 and IoU")
    print("=" * 72)

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        valid_batch_size=BATCH_SIZE,
    )

    model = LandslideModel().to(DEVICE)
    n_params = count_trainable_params(model)

    criterion = CombinedLoss(focal_weight=1.0, dice_weight=1.0)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    best_threshold = 0.5
    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    epochs_without_improve = 0

    print(f"Trainable parameters: {n_params:,}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print()

    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        val_metrics = evaluate_segmentation(
            model,
            valid_loader,
            device=DEVICE,
            threshold=best_threshold,
            criterion=criterion,
        )

        if epoch % THRESHOLD_SEARCH_EVERY == 0:
            best_thr_snapshot = find_best_threshold(model, valid_loader, device=DEVICE)
            best_threshold = best_thr_snapshot["threshold"]
            val_metrics.update(best_thr_snapshot)
            val_metrics["threshold"] = best_threshold

        scheduler.step()

        val_f1 = val_metrics["f1"]
        val_iou = val_metrics["iou"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]
        val_loss = val_metrics["loss"]

        score = 0.6 * val_f1 + 0.4 * val_iou
        improved = score > best_score

        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "threshold": float(best_threshold),
                    "val_metrics": {
                        "precision": float(val_prec),
                        "recall": float(val_rec),
                        "f1": float(val_f1),
                        "iou": float(val_iou),
                    },
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improve += 1

        epoch_sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        mark = "*" if improved else " "
        print(
            f"{mark} Ep {epoch:03d}/{EPOCHS} | "
            f"TL {train_loss:.4f} | VL {val_loss:.4f} | "
            f"P {val_prec * 100:.2f} R {val_rec * 100:.2f} F1 {val_f1 * 100:.2f} IoU {val_iou * 100:.2f} | "
            f"thr {best_threshold:.2f} | lr {lr_now:.2e} | {epoch_sec:.1f}s"
        )

        if time.time() - train_start > 280:
            print("Time limit of 280s reached, stopping training.")
            break

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)."
            )
            break

    training_seconds = time.time() - train_start

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError("Training finished without a valid checkpoint.")

    print("\nLoading best checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    best_threshold = float(ckpt["threshold"])

    val_final = evaluate_segmentation(
        model,
        valid_loader,
        device=DEVICE,
        threshold=best_threshold,
        criterion=criterion,
    )
    test_final = evaluate_segmentation(
        model,
        test_loader,
        device=DEVICE,
        threshold=best_threshold,
        criterion=criterion,
    )

    val_z = 0.6 * val_final["f1"] * 100.0 + 0.4 * val_final["iou"] * 100.0
    test_z = 0.6 * test_final["f1"] * 100.0 + 0.4 * test_final["iou"] * 100.0

    print("\n" + "=" * 72)
    print("Best Validation Snapshot")
    print(f"  Epoch      : {best_epoch}")
    print(f"  Threshold  : {best_threshold:.2f}")
    if best_metrics is not None:
        print(f"  Precision  : {best_metrics['precision'] * 100:.2f}%")
        print(f"  Recall     : {best_metrics['recall'] * 100:.2f}%")
        print(f"  F1         : {best_metrics['f1'] * 100:.2f}%")
        print(f"  IoU        : {best_metrics['iou'] * 100:.2f}%")
    print("=" * 72)

    print("Validation Metrics")
    print(f"  Precision  : {val_final['precision'] * 100:.2f}%")
    print(f"  Recall     : {val_final['recall'] * 100:.2f}%")
    print(f"  F1 Score   : {val_final['f1'] * 100:.2f}%")
    print(f"  IoU        : {val_final['iou'] * 100:.2f}%")
    print(f"  Z          : {val_z:.2f}")

    print("Test Metrics")
    print(f"  Precision  : {test_final['precision'] * 100:.2f}%")
    print(f"  Recall     : {test_final['recall'] * 100:.2f}%")
    print(f"  F1 Score   : {test_final['f1'] * 100:.2f}%")
    print(f"  IoU        : {test_final['iou'] * 100:.2f}%")
    print(f"  Z          : {test_z:.2f}")

    print("Run Summary")
    print(f"  Training seconds     : {training_seconds:.1f}")
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Checkpoint           : {CHECKPOINT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    run_training()
