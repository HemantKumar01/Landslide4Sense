"""
One-time data preparation and runtime utilities for Landslide4Sense experiments.

This version assumes dataset files already exist locally at:
    ../data/landSlide4Sense/

Usage:
    python prepare.py
    python prepare.py --force-stats --max-stat-samples 800
"""

import argparse
import os
import random
import re
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 128
IN_CHANNELS = 14
NUM_CLASSES = 1
SEED = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "landSlide4Sense"))

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "TrainData", "img")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "TrainData", "mask")
VALID_IMG_DIR = os.path.join(DATA_ROOT, "ValidData", "img")
VALID_MASK_DIR = os.path.join(DATA_ROOT, "ValidData", "mask")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "TestData", "img")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "TestData", "test")

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_l4s")
STATS_FILE = os.path.join(CACHE_DIR, "channel_stats.npz")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_idx(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Could not extract numeric id from filename: {name}")
    return int(m.group(1))


def _list_h5(folder):
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]
    return sorted(files, key=_extract_idx)


def _match_pairs(img_dir, mask_dir):
    imgs = _list_h5(img_dir)
    masks = _list_h5(mask_dir)
    if not imgs:
        raise ValueError(f"No .h5 image files found in: {img_dir}")
    if not masks:
        raise ValueError(f"No .h5 mask files found in: {mask_dir}")

    img_map = {_extract_idx(p): p for p in imgs}
    mask_map = {_extract_idx(p): p for p in masks}
    common_ids = sorted(set(img_map).intersection(mask_map))
    if not common_ids:
        raise ValueError(f"No overlapping ids between {img_dir} and {mask_dir}")

    return [(img_map[i], mask_map[i]) for i in common_ids]


def _read_h5_array(path, preferred_keys):
    with h5py.File(path, "r") as f:
        for key in preferred_keys:
            if key in f:
                return f[key][()]
        first_key = list(f.keys())[0]
        return f[first_key][()]


def load_image(path):
    arr = _read_h5_array(path, preferred_keys=["img", "image", "data", "x"]).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor in {path}, got shape {arr.shape}")
    # Convert HWC -> CHW if needed.
    if arr.shape[-1] == IN_CHANNELS:
        arr = arr.transpose(2, 0, 1)
    if arr.shape[0] != IN_CHANNELS:
        raise ValueError(f"Expected {IN_CHANNELS} channels in {path}, got shape {arr.shape}")
    return arr


def load_mask(path):
    arr = _read_h5_array(path, preferred_keys=["mask", "label", "gt", "y"]).astype(np.int64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask tensor in {path}, got shape {arr.shape}")
    return (arr > 0).astype(np.int64)


def validate_dataset_layout(verbose=True):
    required_dirs = [
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VALID_IMG_DIR,
        VALID_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
    ]
    for d in required_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required dataset directory not found: {d}")

    train_pairs = _match_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    valid_pairs = _match_pairs(VALID_IMG_DIR, VALID_MASK_DIR)
    test_pairs = _match_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

    if verbose:
        print(f"Dataset root: {DATA_ROOT}")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Valid pairs: {len(valid_pairs)}")
        print(f"Test pairs:  {len(test_pairs)}")

        sample_img, sample_mask = train_pairs[0]
        x = load_image(sample_img)
        y = load_mask(sample_mask)
        print(f"Sample image shape: {x.shape} | dtype={x.dtype}")
        print(f"Sample mask shape:  {y.shape} | positive ratio={float(y.mean()):.4f}")

    return train_pairs, valid_pairs, test_pairs


def compute_channel_stats(train_pairs, max_samples=512, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(STATS_FILE) and not force:
        stats = np.load(STATS_FILE)
        return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

    pairs = train_pairs[: min(max_samples, len(train_pairs))]
    channel_sum = np.zeros(IN_CHANNELS, dtype=np.float64)
    channel_sumsq = np.zeros(IN_CHANNELS, dtype=np.float64)
    total_pixels = 0

    for img_path, _ in pairs:
        x = load_image(img_path)
        flat = x.reshape(IN_CHANNELS, -1)
        channel_sum += flat.sum(axis=1)
        channel_sumsq += (flat * flat).sum(axis=1)
        total_pixels += flat.shape[1]

    mean = (channel_sum / total_pixels).astype(np.float32)
    var = channel_sumsq / total_pixels - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    np.savez(STATS_FILE, mean=mean, std=std)
    return mean, std


@dataclass
class DatasetItem:
    image_path: str
    mask_path: str


class LandslideDataset(Dataset):
    def __init__(self, pairs, mean, std, augment=False):
        self.items = [DatasetItem(image_path=i, mask_path=m) for i, m in pairs]
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _normalize(self, x):
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return np.clip(x, -6.0, 6.0)

    def _augment(self, x, y):
        if random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()
        if random.random() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()

        if random.random() < 0.4:
            gain = np.random.uniform(0.9, 1.1, size=(IN_CHANNELS, 1, 1)).astype(np.float32)
            bias = np.random.uniform(-0.15, 0.15, size=(IN_CHANNELS, 1, 1)).astype(np.float32)
            x = x * gain + bias

        if random.random() < 0.25:
            noise = np.random.normal(0.0, 0.03, size=x.shape).astype(np.float32)
            x = x + noise

        if random.random() < 0.3:
            h, w = y.shape
            ch = random.randint(8, 24)
            cw = random.randint(8, 24)
            cy = random.randint(0, h - ch)
            cx = random.randint(0, w - cw)
            x[:, cy : cy + ch, cx : cx + cw] = 0.0

        return x, y

    def __getitem__(self, idx):
        item = self.items[idx]
        x = load_image(item.image_path)
        y = load_mask(item.mask_path)

        x = self._normalize(x)
        if self.augment:
            x, y = self._augment(x, y)

        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(y.copy()).float(),
            os.path.basename(item.image_path),
        )


def build_dataloaders(batch_size=16, num_workers=4, valid_batch_size=None):
    train_pairs, valid_pairs, test_pairs = validate_dataset_layout(verbose=False)
    mean, std = compute_channel_stats(train_pairs)

    valid_batch_size = valid_batch_size or batch_size
    pin = torch.cuda.is_available()

    train_ds = LandslideDataset(train_pairs, mean, std, augment=True)
    valid_ds = LandslideDataset(valid_pairs, mean, std, augment=False)
    test_ds = LandslideDataset(test_pairs, mean, std, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, valid_loader, test_loader, (mean, std)


# ---------------------------------------------------------------------------
# Evaluation (fixed assignment metrics)
# ---------------------------------------------------------------------------

def metrics_from_counts(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou


@torch.no_grad()
def collect_probs_and_targets(model, loader, device):
    model.eval()
    probs_all = []
    targets_all = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        probs_all.append(torch.sigmoid(logits).reshape(-1).cpu())
        targets_all.append(y.reshape(-1).cpu())

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy().astype(np.uint8)
    return probs, targets


def evaluate_from_probs(probs, targets, threshold=0.5):
    preds = (probs >= threshold).astype(np.uint8)
    tp = float(np.logical_and(preds == 1, targets == 1).sum())
    fp = float(np.logical_and(preds == 1, targets == 0).sum())
    fn = float(np.logical_and(preds == 0, targets == 1).sum())
    return metrics_from_counts(tp, fp, fn)


@torch.no_grad()
def evaluate_segmentation(model, loader, device, threshold=0.5, criterion=None):
    model.eval()
    total_loss = 0.0
    probs, targets = collect_probs_and_targets(model, loader, device)

    if criterion is not None:
        n_batches = 0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            total_loss += float(criterion(logits, y).item())
            n_batches += 1
        total_loss = total_loss / max(1, n_batches)

    precision, recall, f1, iou = evaluate_from_probs(probs, targets, threshold=threshold)
    return {
        "loss": total_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "threshold": threshold,
    }


@torch.no_grad()
def find_best_threshold(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.02)

    probs, targets = collect_probs_and_targets(model, loader, device)
    best = {"threshold": 0.5, "f1": -1.0, "iou": -1.0, "precision": 0.0, "recall": 0.0}

    for t in thresholds:
        p, r, f1, iou = evaluate_from_probs(probs, targets, threshold=float(t))
        if f1 > best["f1"]:
            best.update({
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "iou": float(iou),
            })
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Landslide4Sense setup and precompute normalization stats")
    parser.add_argument("--force-stats", action="store_true", help="Recompute channel statistics")
    parser.add_argument("--max-stat-samples", type=int, default=512, help="Max train samples for channel statistics")
    args = parser.parse_args()

    set_seed(SEED)
    train_pairs, valid_pairs, test_pairs = validate_dataset_layout(verbose=True)
    mean, std = compute_channel_stats(train_pairs, max_samples=args.max_stat_samples, force=args.force_stats)

    print()
    print(f"Stats cache: {STATS_FILE}")
    print(f"Mean (first 5): {mean[:5]}")
    print(f"Std  (first 5): {std[:5]}")
    print(f"Ready. train={len(train_pairs)} valid={len(valid_pairs)} test={len(test_pairs)}")