"""Supervised sanity check for both ConvNet and FilterbankNet backbones.

Tests two tasks:
  1. Majority-vote classification (cross-entropy) — per-motif accuracy
  2. Fractional occupancy regression (MSE) — per-motif R²

This establishes the ceiling for what DINO should be able to achieve.
Expected: ConvNet high on all motifs, FilterbankNet high on frequency motifs
but poor on shape motifs (sawtooth vs sinusoid, sharp wave vs background).

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/supervised_sanity.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.model import ConvNet, ConvNetV2, FilterbankNet
from modules.data import MEGLabeledDataset

# ------------------------------------------------------------------ config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

n_channels = 1
sampling_frequency = 250
n_classes = 5
feat_dim = 256

# Window params (matches DINO eval window for direct comparison)
window_length = 75
eval_stride = 37

# Training
epochs = 50
batch_size = 256
lr = 1e-3

STATE_NAMES = ["theta_sin", "alpha_spindle", "beta_sawtooth", "sharp_wave", "background"]

# Backbone configs to test
BACKBONES = {
    "convnet": lambda: ConvNet(
        in_channels=n_channels, feat_dim=feat_dim,
        kernel_sizes=[25, 9, 5], strides=[2, 2, 2],
    ),
    "convnet_v2": lambda: ConvNetV2(
        in_channels=n_channels, feat_dim=feat_dim,
        stem_channels=32, stem_kernel_sizes=(7, 15, 31),
        block_channels=(128, 256), block_kernel_sizes=(9, 5),
        attn_hidden=64,
    ),
    "filterbank": lambda: FilterbankNet(
        in_channels=n_channels, feat_dim=feat_dim,
        n_filters=8, filter_length=65, hidden_dim=128, n_queries=4,
    ),
}


class OccupancyDataset(Dataset):
    """Windows with fractional occupancy targets."""

    def __init__(self, X, Y, window_length, stride):
        self.X = X
        self.Y = Y
        n_samples = X.shape[-1]
        self.windows = []
        for start in range(0, n_samples - window_length + 1, stride):
            self.windows.append(start)
        self.window_length = window_length

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        x = self.X[..., start:start + self.window_length]
        y = self.Y[start:start + self.window_length]

        occupancy = np.zeros(n_classes, dtype=np.float32)
        for c in range(n_classes):
            occupancy[c] = (y == c).mean()

        return torch.from_numpy(x.copy()), torch.from_numpy(occupancy)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------------------------------------------ load data
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))
print(f"  X_train: {X_train.shape}, X_eval: {X_eval.shape}")

# ------------------------------------------------------------------ run for each backbone
for backbone_name, backbone_fn in BACKBONES.items():
    print(f"\n{'=' * 70}")
    print(f"BACKBONE: {backbone_name}")
    print(f"{'=' * 70}")

    # ================================================================
    # Task 1: Majority-vote classification
    # ================================================================
    print(f"\n--- Task 1: Majority-vote classification ---")

    train_ds = MEGLabeledDataset(X_train, Y_train, window_length, eval_stride)
    eval_ds = MEGLabeledDataset(X_eval, Y_eval, window_length, eval_stride)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    backbone_cls = backbone_fn().to(device)
    classifier = nn.Linear(feat_dim, n_classes).to(device)

    params = list(backbone_cls.parameters()) + list(classifier.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in backbone_cls.parameters())
    print(f"  Backbone params: {n_params:,}")
    print(f"  Training classification for {epochs} epochs (window={window_length})...")

    for epoch in range(epochs):
        backbone_cls.train()
        classifier.train()
        for windows, labels in train_dl:
            windows = windows.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            logits = classifier(backbone_cls(windows))
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    # Evaluate
    backbone_cls.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for windows, labels in eval_dl:
            windows = windows.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            preds = classifier(backbone_cls(windows)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    eval_acc = (all_preds == all_labels).float().mean().item()

    print(f"\n  Eval accuracy: {eval_acc*100:.1f}%")
    print(f"  Per-motif eval accuracy:")
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).float().mean().item()
            print(f"    {STATE_NAMES[c]:>18}: {acc*100:.1f}%  (n={mask.sum().item()})")

    # ================================================================
    # Task 2: Fractional occupancy regression
    # ================================================================
    print(f"\n--- Task 2: Fractional occupancy regression ---")

    train_occ_ds = OccupancyDataset(X_train, Y_train, window_length, eval_stride)
    eval_occ_ds = OccupancyDataset(X_eval, Y_eval, window_length, eval_stride)
    train_occ_dl = DataLoader(train_occ_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_occ_dl = DataLoader(eval_occ_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    backbone_reg = backbone_fn().to(device)
    regressor = nn.Linear(feat_dim, n_classes).to(device)

    params = list(backbone_reg.parameters()) + list(regressor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  Training regression for {epochs} epochs (window={window_length})...")
    for epoch in range(epochs):
        backbone_reg.train()
        regressor.train()
        epoch_loss = 0.0
        n_batches = 0
        for windows, targets in train_occ_dl:
            windows = windows.to(device)
            targets = targets.to(device)
            preds = regressor(backbone_reg(windows))
            loss = nn.functional.mse_loss(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train MSE = {epoch_loss / n_batches:.4f}")

    # Evaluate
    backbone_reg.eval()
    regressor.eval()
    all_preds_reg, all_targets_reg = [], []
    with torch.no_grad():
        for windows, targets in eval_occ_dl:
            windows = windows.to(device)
            preds = regressor(backbone_reg(windows))
            all_preds_reg.append(preds.cpu())
            all_targets_reg.append(targets)

    all_preds_reg = torch.cat(all_preds_reg).numpy()
    all_targets_reg = torch.cat(all_targets_reg).numpy()

    print(f"\n  Per-motif R²:")
    r2_values = []
    for c in range(n_classes):
        y_true = all_targets_reg[:, c]
        y_pred = all_preds_reg[:, c]
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_values.append(r2)
        print(f"    {STATE_NAMES[c]:>18}: R² = {r2:.3f}  (mean occ = {y_true.mean():.3f})")

    mean_r2 = np.mean(r2_values)
    print(f"\n  Mean R²: {mean_r2:.3f}")

print("\nDone.")
