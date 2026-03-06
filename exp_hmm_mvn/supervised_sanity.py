"""Supervised sanity check for the FilterbankNet backbone.

Tests two tasks:
  1. Majority-vote classification (cross-entropy) — per-state accuracy
  2. Fractional occupancy regression (MSE) — per-state R²

This establishes the ceiling for what DINO should be able to achieve.
If supervised R² is high but DINO R² is low, the bottleneck is the
self-supervised objective, not the architecture.

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_hmm_mvn/supervised_sanity.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.model import FilterbankNet
from modules.data import MEGLabeledDataset

# ------------------------------------------------------------------ config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

n_channels = 52
sampling_frequency = 250
n_classes = 5
feat_dim = 256

# FilterbankNet params (must match train.py)
fb_n_filters = 8
fb_filter_length = 65
fb_hidden_dim = 128

# Window params (matches DINO eval window for direct comparison)
window_length = 200
eval_stride = 100

# Training
epochs = 50
batch_size = 128
lr = 1e-3

# Bandpass (must match train.py)
bandpass = (3, 45)

STATE_NAMES = ["theta", "alpha", "beta", "low_gamma", "background"]


def bandpass_filter(X, fs, low, high):
    sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    X_filt = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    mean = X_filt.mean(axis=-1, keepdims=True)
    std = X_filt.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (X_filt - mean) / std


class OccupancyDataset(Dataset):
    """Windows with fractional occupancy targets."""

    def __init__(self, X, Y, window_length, stride):
        self.X = X
        self.Y = Y
        n_samples = X.shape[-1]
        self.windows = []
        for start in range(0, n_samples - window_length + 1, stride):
            self.windows.append(start)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        x = self.X[:, start:start + window_length]
        y = self.Y[start:start + window_length]

        # Fractional occupancy: proportion of time in each state
        occupancy = np.zeros(n_classes, dtype=np.float32)
        for c in range(n_classes):
            occupancy[c] = (y == c).mean()

        return torch.from_numpy(x), torch.from_numpy(occupancy)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------------------------------------------ load data
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))

if bandpass is not None:
    print(f"  Bandpass filtering {bandpass[0]}-{bandpass[1]} Hz...")
    X_train = bandpass_filter(X_train, sampling_frequency, *bandpass)
    X_eval = bandpass_filter(X_eval, sampling_frequency, *bandpass)

window_length_val = window_length  # store for use below

# ================================================================
# Task 1: Majority-vote classification
# ================================================================
print("\n" + "=" * 70)
print("Task 1: Majority-vote classification")
print("=" * 70)

train_ds = MEGLabeledDataset(X_train, Y_train, window_length, eval_stride)
eval_ds = MEGLabeledDataset(X_eval, Y_eval, window_length, eval_stride)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4)

backbone_cls = FilterbankNet(
    in_channels=n_channels, feat_dim=feat_dim,
    n_filters=fb_n_filters, filter_length=fb_filter_length,
    hidden_dim=fb_hidden_dim,
).to(device)
classifier = nn.Linear(feat_dim, n_classes).to(device)

params = list(backbone_cls.parameters()) + list(classifier.parameters())
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
loss_fn = nn.CrossEntropyLoss()

print(f"Training classification for {epochs} epochs (window={window_length})...")
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

# Evaluate classification
backbone_cls.eval()
classifier.eval()
all_preds, all_labels = [], []
train_correct, train_total = 0, 0
with torch.no_grad():
    for windows, labels in train_dl:
        windows = windows.to(device)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
        preds = classifier(backbone_cls(windows)).argmax(dim=1).cpu()
        train_correct += (preds == labels.cpu()).sum().item()
        train_total += len(labels)

    for windows, labels in eval_dl:
        windows = windows.to(device)
        labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
        preds = classifier(backbone_cls(windows)).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
train_acc = train_correct / train_total
eval_acc = (all_preds == all_labels).float().mean().item()

print(f"\n  Train accuracy: {train_acc*100:.1f}%")
print(f"  Eval accuracy:  {eval_acc*100:.1f}%")
print(f"  Per-state eval accuracy:")
for c in range(n_classes):
    mask = all_labels == c
    if mask.sum() > 0:
        acc = (all_preds[mask] == c).float().mean().item()
        print(f"    {STATE_NAMES[c]:>12}: {acc*100:.1f}%  (n={mask.sum().item()})")

# ================================================================
# Task 2: Fractional occupancy regression
# ================================================================
print("\n" + "=" * 70)
print("Task 2: Fractional occupancy regression")
print("=" * 70)

train_occ_ds = OccupancyDataset(X_train, Y_train, window_length, eval_stride)
eval_occ_ds = OccupancyDataset(X_eval, Y_eval, window_length, eval_stride)
train_occ_dl = DataLoader(train_occ_ds, batch_size=batch_size, shuffle=True, num_workers=4)
eval_occ_dl = DataLoader(eval_occ_ds, batch_size=batch_size, shuffle=False, num_workers=4)

backbone_reg = FilterbankNet(
    in_channels=n_channels, feat_dim=feat_dim,
    n_filters=fb_n_filters, filter_length=fb_filter_length,
    hidden_dim=fb_hidden_dim,
).to(device)
regressor = nn.Linear(feat_dim, n_classes).to(device)

params = list(backbone_reg.parameters()) + list(regressor.parameters())
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

print(f"Training regression for {epochs} epochs (window={window_length})...")
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
        print(f"  Epoch {epoch+1}: train MSE = {epoch_loss / n_batches:.4f}")

# Evaluate regression
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

# Per-state R²
print(f"\n  Per-state R²:")
r2_values = []
for c in range(n_classes):
    y_true = all_targets_reg[:, c]
    y_pred = all_preds_reg[:, c]
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_values.append(r2)
    print(f"    {STATE_NAMES[c]:>12}: R² = {r2:.3f}  (mean occ = {y_true.mean():.3f})")

mean_r2 = np.mean(r2_values)
print(f"\n  Mean R²: {mean_r2:.3f}")

print("\nDone.")
