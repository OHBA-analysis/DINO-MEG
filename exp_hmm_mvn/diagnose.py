"""Diagnostic analysis of the trained oscillatory burst DINO backbone.

Produces:
  figures/training_curves.png    – 2×2: loss, grad norm, kNN top-1, LP top-1
  figures/confusion_matrix.png   – 5×5 k-NN confusion matrix
  figures/per_state_accuracy.png – per-state top-1 accuracy bar chart
  figures/tsne.png               – t-SNE of backbone features coloured by state

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_hmm_mvn/diagnose.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet, ConvNet2D, ViT1D, ViT2D, FilterbankNet
from modules.data import MEGLabeledDataset, compute_amplitude_envelopes

# ------------------------------------------------------------------ config
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "backbone_final.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "checkpoints", "metrics.json")
os.makedirs(OUT_DIR, exist_ok=True)

n_channels = 52
feat_dim = 256
knn_k = 10
eval_window_length = 200
eval_stride = 100
n_classes = 5
sampling_frequency = 250

# Input representation (must match train.py)
use_tf = False
tf_freqs = [4, 6, 8, 10, 13, 17, 22, 28, 35, 43]

# Backbone type (must match train.py)
backbone_type = "filterbank"  # "convnet", "vit", or "filterbank"
vit_patch_size = 5               # 1D: temporal patch size
vit_patch_size_2d = (5, 25)      # 2D: (freq, time) patch size
vit_d_model = 192
vit_n_heads = 4
vit_n_layers = 4
vit_dropout = 0.05
use_mask_token = False  # must match train.py (use_masked_prediction)

# Filterbank backbone (must match train.py)
fb_n_filters = 8
fb_filter_length = 65
fb_hidden_dim = 128

STATE_NAMES = ["theta", "alpha", "beta", "low_gamma", "background"]
STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#cccccc"]

# Bandpass filter (must match train.py)
bandpass = (3, 45)  # Hz, or None to skip


def bandpass_filter(X, fs, low, high):
    """Bandpass filter and re-z-score. X: (C, T), returns (C, T) float32."""
    sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    X_filt = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    mean = X_filt.mean(axis=-1, keepdims=True)
    std = X_filt.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (X_filt - mean) / std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ training curves
print("\n--- Training curves ---")
with open(METRICS_FILE) as f:
    metrics = json.load(f)

epochs = [m["epoch"] for m in metrics]
losses = [m["loss"] for m in metrics]
grad_norms = [m["grad_norm"] for m in metrics]
knn_epochs = [m["epoch"] for m in metrics if "knn_top1" in m]
knn_top1 = [m["knn_top1"] for m in metrics if "knn_top1" in m]
lp_epochs = [m["epoch"] for m in metrics if "lp_top1" in m]
lp_top1 = [m["lp_top1"] for m in metrics if "lp_top1" in m]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training curves", fontsize=13)

axes[0, 0].plot(epochs, losses, linewidth=0.8, color="steelblue")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("DINO loss")

axes[0, 1].plot(epochs, grad_norms, linewidth=0.8, color="darkorange")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Gradient norm")
axes[0, 1].set_title("Mean gradient norm")

if knn_top1:
    axes[1, 0].plot(knn_epochs, [v * 100 for v in knn_top1], "o-", markersize=3, color="green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_title("k-NN top-1 accuracy")
else:
    axes[1, 0].text(0.5, 0.5, "No k-NN data", ha="center", va="center", transform=axes[1, 0].transAxes)

if lp_top1:
    axes[1, 1].plot(lp_epochs, [v * 100 for v in lp_top1], "o-", markersize=3, color="purple")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Linear probe top-1 accuracy")
else:
    axes[1, 1].text(0.5, 0.5, "No LP data", ha="center", va="center", transform=axes[1, 1].transAxes)

fig.tight_layout()
savefig(fig, "training_curves.png")


# ------------------------------------------------------------------ load model
print("\n--- Loading backbone ---")
if use_tf:
    if backbone_type == "convnet":
        backbone = ConvNet2D(in_channels=n_channels, feat_dim=feat_dim).to(device)
    elif backbone_type == "vit":
        backbone = ViT2D(
            in_channels=n_channels, feat_dim=feat_dim,
            patch_size=vit_patch_size_2d, d_model=vit_d_model,
            n_heads=vit_n_heads, n_layers=vit_n_layers, dropout=vit_dropout,
        ).to(device)
    elif backbone_type == "filterbank":
        raise ValueError("FilterbankNet is a 1D backbone — set use_tf=False")
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")
else:
    if backbone_type == "convnet":
        backbone = ConvNet(in_channels=n_channels, feat_dim=feat_dim, kernel_sizes=[25, 9, 5]).to(device)
    elif backbone_type == "vit":
        backbone = ViT1D(
            in_channels=n_channels, feat_dim=feat_dim,
            patch_size=vit_patch_size, d_model=vit_d_model,
            n_heads=vit_n_heads, n_layers=vit_n_layers, dropout=vit_dropout,
            use_mask_token=use_mask_token,
        ).to(device)
    elif backbone_type == "filterbank":
        backbone = FilterbankNet(
            in_channels=n_channels, feat_dim=feat_dim,
            n_filters=fb_n_filters, filter_length=fb_filter_length,
            hidden_dim=fb_hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

backbone.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
backbone.eval()
print("Backbone loaded.")


# ------------------------------------------------------------------ extract features
print("Loading eval data...")
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

if bandpass is not None:
    print(f"  Bandpass filtering {bandpass[0]}-{bandpass[1]} Hz...")
    X_eval = bandpass_filter(X_eval, sampling_frequency, *bandpass)
    X_train = bandpass_filter(X_train, sampling_frequency, *bandpass)

if use_tf:
    print(f"  Computing TF amplitude envelopes ({len(tf_freqs)} freqs)...")
    X_eval_input = compute_amplitude_envelopes(X_eval, sampling_frequency, tf_freqs,
                                                log_transform=True, standardize=True)
    X_train_input = compute_amplitude_envelopes(X_train, sampling_frequency, tf_freqs,
                                                 log_transform=True, standardize=True)
else:
    X_eval_input = X_eval
    X_train_input = X_train

train_ds = MEGLabeledDataset(X_train_input, Y_train, eval_window_length, eval_stride)
eval_ds = MEGLabeledDataset(X_eval_input, Y_eval, eval_window_length, eval_stride)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4)


def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for windows, lbls in loader:
            f = backbone(windows.to(device))
            f = F.normalize(f, dim=1).cpu()
            feats.append(f)
            labels.append(torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls)
    return torch.cat(feats), torch.cat(labels)


print("Extracting train features...")
train_feats, train_labels = extract(train_dl)
print("Extracting eval features...")
eval_feats, eval_labels = extract(eval_dl)
print(f"  Train: {train_feats.shape}  Eval: {eval_feats.shape}")


# ------------------------------------------------------------------ k-NN predictions
print("Running k-NN...")
sims = eval_feats @ train_feats.t()
topk = sims.topk(knn_k, dim=1).indices
neighbor_labels = train_labels[topk]

preds = torch.tensor([
    torch.bincount(neighbor_labels[i], minlength=n_classes).argmax().item()
    for i in range(len(eval_feats))
])
overall_top1 = (preds == eval_labels).float().mean().item()
print(f"Overall k-NN top-1: {overall_top1*100:.2f}%")


# ------------------------------------------------------------------ confusion matrix
conf = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(eval_labels.numpy(), preds.numpy()):
    conf[true, pred] += 1

per_class_acc = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(conf, cmap="Blues")
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"k-NN Confusion Matrix (top-1 = {overall_top1*100:.1f}%)")
for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if conf[i, j] > conf[i].max() * 0.5 else "black"
        ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                fontsize=8, color=color)
plt.colorbar(im, ax=ax)
fig.tight_layout()
savefig(fig, "confusion_matrix.png")


# ------------------------------------------------------------------ per-state accuracy bar
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(n_classes), per_class_acc * 100,
              color=STATE_COLORS, edgecolor="white")
ax.axhline(overall_top1 * 100, ls="--", color="red",
           label=f"Mean {overall_top1*100:.1f}%")
ax.set_xticks(range(n_classes))
ax.set_xticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("State")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Per-state k-NN top-1 accuracy")
ax.set_ylim(0, 105)
ax.legend()
for i, v in enumerate(per_class_acc):
    ax.text(i, v * 100 + 1, f"{v*100:.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
savefig(fig, "per_state_accuracy.png")

print("\nPer-state accuracy:")
for c in range(n_classes):
    print(f"  {STATE_NAMES[c]}: {per_class_acc[c]*100:.1f}%  (n={conf[c].sum()})")


# ------------------------------------------------------------------ t-SNE
n_tsne = min(5000, len(eval_feats))
print(f"\nRunning t-SNE on {n_tsne} eval windows...")
idx = np.random.RandomState(42).choice(len(eval_feats), size=n_tsne, replace=False)
feats_sub = eval_feats[idx].numpy()
labels_sub = eval_labels[idx].numpy()

tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000)
embed = tsne.fit_transform(feats_sub)

fig, ax = plt.subplots(figsize=(8, 7))
for c in range(n_classes):
    mask = labels_sub == c
    ax.scatter(embed[mask, 0], embed[mask, 1], s=4, alpha=0.6,
               color=STATE_COLORS[c], label=STATE_NAMES[c])
ax.legend(title="State", markerscale=3, fontsize=9, loc="best")
ax.set_title(f"t-SNE of backbone features (eval, {n_tsne} windows)")
ax.axis("off")
fig.tight_layout()
savefig(fig, "tsne.png")

print("\nDone.")
