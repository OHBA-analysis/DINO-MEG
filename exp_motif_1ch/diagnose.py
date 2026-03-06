"""Diagnostic analysis of the trained single-channel motif DINO backbone.

Produces:
  figures/{backbone_type}/training_curves.png    – 2x2: loss, grad norm, kNN top-1, LP top-1
  figures/{backbone_type}/confusion_matrix.png   – 5x5 k-NN confusion matrix
  figures/{backbone_type}/per_motif_accuracy.png – per-motif top-1 accuracy bar chart
  figures/{backbone_type}/tsne.png               – t-SNE of backbone features coloured by motif

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/diagnose.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet, ConvNetV2, FilterbankNet
from modules.data import MEGLabeledDataset

# ------------------------------------------------------------------ config
backbone_type = "convnet_v2"  # "convnet", "convnet_v2", or "filterbank"

CKPT = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}", "backbone_final.pt")
METRICS_FILE = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}", "metrics.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", backbone_type)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

n_channels = 1
feat_dim = 256
knn_k = 10
eval_window_length = 75
eval_stride = 37
n_classes = 5
sampling_frequency = 250

# ConvNet params
convnet_kernel_sizes = [25, 9, 5]
convnet_strides = [2, 2, 2]

# FilterbankNet params
fb_n_filters = 8
fb_filter_length = 65
fb_hidden_dim = 128
fb_n_queries = 4

STATE_NAMES = ["theta_sin", "alpha_spindle", "beta_sawtooth", "sharp_wave", "background"]
STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#cccccc"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Backbone type: {backbone_type}")


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ training curves
print("\n--- Training curves ---")
with open(METRICS_FILE) as f:
    metrics = json.load(f)

epochs_list = [m["epoch"] for m in metrics]
losses = [m["loss"] for m in metrics]
grad_norms = [m["grad_norm"] for m in metrics]
knn_epochs = [m["epoch"] for m in metrics if "knn_top1" in m]
knn_top1 = [m["knn_top1"] for m in metrics if "knn_top1" in m]
lp_epochs = [m["epoch"] for m in metrics if "lp_top1" in m]
lp_top1 = [m["lp_top1"] for m in metrics if "lp_top1" in m]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Training curves ({backbone_type})", fontsize=13)

axes[0, 0].plot(epochs_list, losses, linewidth=0.8, color="steelblue")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("DINO loss")

axes[0, 1].plot(epochs_list, grad_norms, linewidth=0.8, color="darkorange")
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
if backbone_type == "convnet":
    backbone = ConvNet(
        in_channels=n_channels, feat_dim=feat_dim,
        kernel_sizes=convnet_kernel_sizes, strides=convnet_strides,
    ).to(device)
elif backbone_type == "convnet_v2":
    backbone = ConvNetV2(
        in_channels=n_channels, feat_dim=feat_dim,
        stem_channels=32, stem_kernel_sizes=(7, 15, 31),
        block_channels=(128, 256), block_kernel_sizes=(9, 5),
        attn_hidden=64,
    ).to(device)
elif backbone_type == "filterbank":
    backbone = FilterbankNet(
        in_channels=n_channels, feat_dim=feat_dim,
        n_filters=fb_n_filters, filter_length=fb_filter_length,
        hidden_dim=fb_hidden_dim, n_queries=fb_n_queries,
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

train_ds = MEGLabeledDataset(X_train, Y_train, eval_window_length, eval_stride)
eval_ds = MEGLabeledDataset(X_eval, Y_eval, eval_window_length, eval_stride)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=4)


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
ax.set_title(f"k-NN Confusion Matrix ({backbone_type}, top-1 = {overall_top1*100:.1f}%)")
for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if conf[i, j] > conf[i].max() * 0.5 else "black"
        ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                fontsize=8, color=color)
plt.colorbar(im, ax=ax)
fig.tight_layout()
savefig(fig, "confusion_matrix.png")


# ------------------------------------------------------------------ per-motif accuracy bar
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(n_classes), per_class_acc * 100,
              color=STATE_COLORS, edgecolor="white")
ax.axhline(overall_top1 * 100, ls="--", color="red",
           label=f"Mean {overall_top1*100:.1f}%")
ax.set_xticks(range(n_classes))
ax.set_xticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("Motif")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title(f"Per-motif k-NN top-1 accuracy ({backbone_type})")
ax.set_ylim(0, 105)
ax.legend()
for i, v in enumerate(per_class_acc):
    ax.text(i, v * 100 + 1, f"{v*100:.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
savefig(fig, "per_motif_accuracy.png")

print("\nPer-motif accuracy:")
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
ax.legend(title="Motif", markerscale=3, fontsize=9, loc="best")
ax.set_title(f"t-SNE of backbone features ({backbone_type}, eval, {n_tsne} windows)")
ax.axis("off")
fig.tight_layout()
savefig(fig, "tsne.png")

print("\nDone.")
