"""Diagnostic analysis of the trained MNIST DINO backbone.

Produces:
  figures/confusion_matrix.png   – per-class k-NN confusion matrix
  figures/tsne.png               – t-SNE of backbone features (colour = class)
  figures/augmentation_grid.png  – sample global/local crops after augmentation
  figures/knn_per_class.png      – per-class top-1 accuracy bar chart

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_mnist/diagnose.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet2D

# ------------------------------------------------------------------ config
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "backbone_final.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

feat_dim = 256
knn_k = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------------------------------------------------------------------ load model
backbone = ConvNet2D(feat_dim=feat_dim).to(device)
backbone.load_state_dict(torch.load(CKPT, map_location=device))
backbone.eval()
print("Backbone loaded.")


# ------------------------------------------------------------------ extract features
_norm = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=_norm)
test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_norm)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4)
test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=4)


def extract(loader):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            f = backbone(imgs.to(device))
            f = F.normalize(f, dim=1).cpu()
            feats.append(f)
            labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


print("Extracting train features...")
train_feats, train_labels = extract(train_dl)
print("Extracting test features...")
test_feats,  test_labels  = extract(test_dl)
print(f"  Train: {train_feats.shape}  Test: {test_feats.shape}")


# ------------------------------------------------------------------ k-NN predictions
print("Running k-NN...")
sims = test_feats @ train_feats.t()          # (N_test, N_train)
topk = sims.topk(knn_k, dim=1).indices      # (N_test, k)
neighbor_labels = train_labels[topk]         # (N_test, k)

preds = torch.tensor([
    torch.bincount(neighbor_labels[i]).argmax().item()
    for i in range(len(test_feats))
])
overall_top1 = (preds == test_labels).float().mean().item()
print(f"Overall k-NN top-1: {overall_top1*100:.2f}%")


# ------------------------------------------------------------------ confusion matrix
n_classes = 10
conf = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(test_labels.numpy(), preds.numpy()):
    conf[true, pred] += 1

per_class_acc = conf.diagonal() / conf.sum(axis=1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(conf, cmap="Blues")
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"k-NN Confusion Matrix (top-1 = {overall_top1*100:.1f}%)")
for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if conf[i, j] > conf[i].max() * 0.5 else "black"
        ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                fontsize=7, color=color)
plt.colorbar(im, ax=ax)
fig.tight_layout()
path = os.path.join(OUT_DIR, "confusion_matrix.png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"Saved {path}")


# ------------------------------------------------------------------ per-class accuracy bar
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(n_classes), per_class_acc * 100, color="steelblue", edgecolor="white")
ax.axhline(overall_top1 * 100, ls="--", color="red", label=f"Mean {overall_top1*100:.1f}%")
ax.set_xticks(range(n_classes))
ax.set_xlabel("Digit class"); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Per-class k-NN top-1 accuracy")
ax.set_ylim(0, 105)
ax.legend()
for i, v in enumerate(per_class_acc):
    ax.text(i, v * 100 + 1, f"{v*100:.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
path = os.path.join(OUT_DIR, "knn_per_class.png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"Saved {path}")
print("\nPer-class accuracy:")
for c in range(n_classes):
    print(f"  {c}: {per_class_acc[c]*100:.1f}%  (n={conf[c].sum()})")


# ------------------------------------------------------------------ t-SNE (subsample)
print("\nRunning t-SNE on 5000 test samples...")
idx = np.random.choice(len(test_feats), size=5000, replace=False)
feats_sub = test_feats[idx].numpy()
labels_sub = test_labels[idx].numpy()

tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000, verbose=1)
embed = tsne.fit_transform(feats_sub)

cmap = cm.get_cmap("tab10", 10)
fig, ax = plt.subplots(figsize=(8, 7))
for c in range(n_classes):
    mask = labels_sub == c
    ax.scatter(embed[mask, 0], embed[mask, 1], s=4, alpha=0.6,
               color=cmap(c), label=str(c))
ax.legend(title="Digit", markerscale=3, fontsize=9, loc="best")
ax.set_title("t-SNE of backbone features (test set, 5000 samples)")
ax.axis("off")
fig.tight_layout()
path = os.path.join(OUT_DIR, "tsne.png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"Saved {path}")


# ------------------------------------------------------------------ augmentation visualisation
# Show 4 MNIST digits with all 4 views side-by-side
_global_crop = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
_local_crop = transforms.Compose([
    transforms.RandomResizedCrop(14, scale=(0.2, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

raw_ds = datasets.MNIST(DATA_DIR, train=True, transform=transforms.ToTensor())
n_show = 8
view_labels = ["global 1 (28×28)", "global 2 (28×28)", "local 1 (14×14)", "local 2 (14×14)"]

fig, axes = plt.subplots(n_show, 5, figsize=(10, 2 * n_show))
fig.suptitle("Augmented views per sample (columns: original | g1 | g2 | l1 | l2)",
             fontsize=11)

_denorm = lambda t: t * 0.3081 + 0.1307  # undo normalisation for display

for row in range(n_show):
    img_pil, label = raw_ds[row]
    img_pil_pil = transforms.ToPILImage()(img_pil)
    views = [_global_crop(img_pil_pil), _global_crop(img_pil_pil),
             _local_crop(img_pil_pil), _local_crop(img_pil_pil)]

    # original
    axes[row, 0].imshow(img_pil.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"label={label}", fontsize=7) if row == 0 else None
    axes[row, 0].axis("off")

    for col, v in enumerate(views):
        axes[row, col + 1].imshow(_denorm(v).squeeze().numpy(),
                                  cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, col + 1].set_title(view_labels[col], fontsize=7)
        axes[row, col + 1].axis("off")

fig.tight_layout()
path = os.path.join(OUT_DIR, "augmentation_grid.png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"Saved {path}")

print("\nDone.")
