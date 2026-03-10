"""Diagnostic analysis of the trained real MEG DINO backbone.

Produces figures in exp_real_meg/figures/:
  training_curves.png      – 2x2: loss, grad norm, center norm, feature rank
  pca_by_channel.png       – PCA scatter coloured by parcel channel
  pca_by_subject.png       – PCA scatter coloured by subject
  pca_by_cluster.png       – PCA scatter coloured by k-means cluster
  tsne_by_channel.png      – t-SNE coloured by parcel (52 channels)
  tsne_by_subject.png      – t-SNE coloured by subject
  tsne_by_cluster.png      – t-SNE coloured by k-means cluster
  pca_eigenspectrum.png    – PCA variance explained
  embedding_viz.png        – learned channel & subject embeddings (PCA)

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_real_meg/diagnose.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNetV2

# ------------------------------------------------------------------ config
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CKPT = os.path.join(CKPT_DIR, "backbone_final.pt")
METRICS_FILE = os.path.join(CKPT_DIR, "metrics.json")
META_FILE = os.path.join(CKPT_DIR, "metadata.json")
DATA_ROOT = "/well/win-camcan/shared/spring23/src"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

feat_dim = 256
emb_dim = 32
n_parcels = 52
sampling_frequency = 250
eval_window_length = 112
eval_stride = 56
n_tsne = 8000
n_clusters = 20

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

epochs_list = [m["epoch"] for m in metrics]
losses = [m["loss"] for m in metrics]
grad_norms = [m["grad_norm"] for m in metrics]
center_norms = [m["center_norm"] for m in metrics]
feat_ranks = [m.get("feat_rank", 0) for m in metrics]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Real MEG DINO training curves", fontsize=13)

axes[0, 0].plot(epochs_list, losses, linewidth=0.8, color="steelblue")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("DINO loss")

# Replace Inf with NaN for plotting
grad_norms_plot = [g if np.isfinite(g) else np.nan for g in grad_norms]
axes[0, 1].plot(epochs_list, grad_norms_plot, linewidth=0.8, color="darkorange")
# Mark epochs with Inf gradient
inf_epochs = [e for e, g in zip(epochs_list, grad_norms) if not np.isfinite(g)]
if inf_epochs:
    axes[0, 1].axhspan(axes[0, 1].get_ylim()[0], axes[0, 1].get_ylim()[1],
                        alpha=0.0)  # force axis range
    for ie in inf_epochs:
        axes[0, 1].axvline(ie, color="red", alpha=0.15, linewidth=1)
    axes[0, 1].set_title(f"Mean gradient norm ({len(inf_epochs)} epochs had Inf)")
else:
    axes[0, 1].set_title("Mean gradient norm")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Gradient norm")

axes[1, 0].plot(epochs_list, center_norms, linewidth=0.8, color="green")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Center norm")
axes[1, 0].set_title("Center vector norm")

axes[1, 1].plot(epochs_list, feat_ranks, linewidth=0.8, color="purple")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Effective rank")
axes[1, 1].set_title("Feature effective rank")

fig.tight_layout()
savefig(fig, "training_curves.png")


# ------------------------------------------------------------------ load metadata
print("\n--- Loading metadata ---")
with open(META_FILE) as f:
    meta = json.load(f)

subject_names = meta["subject_names"]
n_subjects = meta["n_subjects"]
file_metadata = {int(k): v for k, v in meta["metadata"].items()}
print(f"  {n_subjects} subjects, {meta['n_files']} channel-streams")


# ------------------------------------------------------------------ load model
print("\n--- Loading backbone ---")
backbone = ConvNetV2(
    in_channels=1, feat_dim=feat_dim,
    stem_channels=32, stem_kernel_sizes=(7, 15, 31),
    block_channels=(128, 256), block_kernel_sizes=(9, 5),
    attn_hidden=64,
).to(device)

state = torch.load(CKPT, map_location=device, weights_only=True)
backbone.load_state_dict(state["backbone"])
backbone.eval()

channel_emb_layer = nn.Embedding(n_parcels, emb_dim).to(device)
subject_emb_layer = nn.Embedding(n_subjects, emb_dim).to(device)
channel_emb_layer.load_state_dict(state["channel_emb"])
subject_emb_layer.load_state_dict(state["subject_emb"])
print("Backbone + embeddings loaded.")


# ------------------------------------------------------------------ extract features
print("\n--- Extracting features ---")

import mne
import glob
mne.set_log_level("WARNING")

max_subjects_eval = min(20, n_subjects)
fif_files = sorted(glob.glob(os.path.join(DATA_ROOT, "sub-*/sflip_parc-raw.fif")))
fif_files = fif_files[:max_subjects_eval]

all_feats = []
all_channel_ids = []
all_subject_ids = []

for subj_idx, fif_path in enumerate(fif_files):
    subj_name = os.path.basename(os.path.dirname(fif_path))
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.pick("misc")
    data = raw.get_data()

    for ch_idx in range(data.shape[0]):
        ch_data = data[ch_idx]
        mu, std = ch_data.mean(), ch_data.std()
        if std > 0:
            ch_data = (ch_data - mu) / std
        arr = ch_data[np.newaxis, :].astype(np.float32)

        T = arr.shape[-1]
        starts = list(range(0, T - eval_window_length + 1, eval_stride))
        windows = np.stack([arr[:, s:s + eval_window_length] for s in starts])
        windows_t = torch.from_numpy(windows).to(device)

        with torch.no_grad():
            feats = []
            for batch_start in range(0, len(windows_t), 256):
                batch = windows_t[batch_start:batch_start + 256]
                f = backbone(batch)
                f = F.normalize(f, dim=1)
                feats.append(f.cpu())
            feats = torch.cat(feats)

        all_feats.append(feats)
        all_channel_ids.extend([ch_idx] * len(feats))
        all_subject_ids.extend([subj_idx] * len(feats))

    print(f"  {subj_name}: {data.shape[0]} channels, {len(starts)} windows/ch")

all_feats = torch.cat(all_feats)
all_channel_ids = np.array(all_channel_ids)
all_subject_ids = np.array(all_subject_ids)
print(f"Total features: {all_feats.shape}")


# ------------------------------------------------------------------ subsample
rng = np.random.RandomState(42)
n_total = len(all_feats)
n_use = min(n_tsne, n_total)
idx = rng.choice(n_total, size=n_use, replace=False)
feats_sub = all_feats[idx].numpy()
ch_ids_sub = all_channel_ids[idx]
subj_ids_sub = all_subject_ids[idx]


# ------------------------------------------------------------------ PCA scatter plots
print("\n--- PCA scatter plots ---")
pca_2d = PCA(n_components=2)
pca_embed = pca_2d.fit_transform(feats_sub)
pca_var = pca_2d.explained_variance_ratio_

# PCA by channel
fig, ax = plt.subplots(figsize=(10, 9))
cmap = plt.colormaps["tab20"]
for ch in range(n_parcels):
    mask = ch_ids_sub == ch
    if mask.sum() == 0:
        continue
    ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1], s=3, alpha=0.4,
               color=cmap(ch % 20), label=f"ch{ch}" if ch < 20 else None)
ax.legend(title="Channel", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
ax.set_title(f"PCA by parcel channel ({n_use} windows, {max_subjects_eval} subjects)")
fig.tight_layout()
savefig(fig, "pca_by_channel.png")

# PCA by subject
fig, ax = plt.subplots(figsize=(10, 9))
cmap_subj = plt.colormaps["tab20"]
for si in range(max_subjects_eval):
    mask = subj_ids_sub == si
    if mask.sum() == 0:
        continue
    ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1], s=3, alpha=0.4,
               color=cmap_subj(si % 20),
               label=subject_names[si] if si < 20 else None)
ax.legend(title="Subject", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
ax.set_title(f"PCA by subject ({n_use} windows)")
fig.tight_layout()
savefig(fig, "pca_by_subject.png")

# PCA by k-means cluster
print(f"  Running k-means (k={n_clusters})...")
km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
cluster_labels = km.fit_predict(feats_sub)

fig, ax = plt.subplots(figsize=(10, 9))
cmap_k = plt.colormaps["tab20"]
for k in range(n_clusters):
    mask = cluster_labels == k
    ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1], s=3, alpha=0.4,
               color=cmap_k(k % 20), label=f"k{k}")
ax.legend(title="Cluster", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
ax.set_title(f"PCA by k-means cluster (k={n_clusters})")
fig.tight_layout()
savefig(fig, "pca_by_cluster.png")


# ------------------------------------------------------------------ t-SNE scatter plots
print(f"\nRunning t-SNE on {n_use} samples...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000)
embed = tsne.fit_transform(feats_sub)

# t-SNE by channel
fig, ax = plt.subplots(figsize=(10, 9))
for ch in range(n_parcels):
    mask = ch_ids_sub == ch
    if mask.sum() == 0:
        continue
    ax.scatter(embed[mask, 0], embed[mask, 1], s=3, alpha=0.4,
               color=cmap(ch % 20), label=f"ch{ch}" if ch < 20 else None)
ax.legend(title="Channel", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_title(f"t-SNE by parcel channel ({n_use} windows, {max_subjects_eval} subjects)")
ax.axis("off")
fig.tight_layout()
savefig(fig, "tsne_by_channel.png")

# t-SNE by subject
fig, ax = plt.subplots(figsize=(10, 9))
for si in range(max_subjects_eval):
    mask = subj_ids_sub == si
    if mask.sum() == 0:
        continue
    ax.scatter(embed[mask, 0], embed[mask, 1], s=3, alpha=0.4,
               color=cmap_subj(si % 20),
               label=subject_names[si] if si < 20 else None)
ax.legend(title="Subject", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_title(f"t-SNE by subject ({n_use} windows)")
ax.axis("off")
fig.tight_layout()
savefig(fig, "tsne_by_subject.png")

# t-SNE by k-means cluster
fig, ax = plt.subplots(figsize=(10, 9))
for k in range(n_clusters):
    mask = cluster_labels == k
    ax.scatter(embed[mask, 0], embed[mask, 1], s=3, alpha=0.4,
               color=cmap_k(k % 20), label=f"k{k}")
ax.legend(title="Cluster", markerscale=3, fontsize=6, loc="best", ncol=2)
ax.set_title(f"t-SNE by k-means cluster (k={n_clusters})")
ax.axis("off")
fig.tight_layout()
savefig(fig, "tsne_by_cluster.png")


# ------------------------------------------------------------------ PCA eigenspectrum
print("\n--- PCA eigenspectrum ---")
n_pca = min(50, feat_dim)
pca_full = PCA(n_components=n_pca)
pca_full.fit(all_feats.numpy())

var_ratio = pca_full.explained_variance_ratio_
cumvar = np.cumsum(var_ratio)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(range(n_pca), var_ratio * 100, color="steelblue", alpha=0.8, label="Individual")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance (%)")
ax1.set_title(f"PCA of real MEG backbone features ({feat_dim}-dim)")

ax2 = ax1.twinx()
ax2.plot(range(n_pca), cumvar * 100, color="red", marker="o", markersize=3, label="Cumulative")
ax2.set_ylabel("Cumulative variance (%)")

for thresh, ls in [(0.90, "--"), (0.95, ":")]:
    n_needed = np.searchsorted(cumvar, thresh) + 1
    ax2.axhline(thresh * 100, color="gray", linestyle=ls, alpha=0.5)
    if n_needed <= n_pca:
        ax2.annotate(f"{thresh*100:.0f}% at PC{n_needed}",
                     xy=(n_needed, thresh * 100), fontsize=9,
                     xytext=(n_needed + 3, thresh * 100 - 3),
                     arrowprops=dict(arrowstyle="->", color="gray"))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
fig.tight_layout()
savefig(fig, "pca_eigenspectrum.png")


# ------------------------------------------------------------------ embedding visualization
print("\n--- Embedding visualization ---")

with torch.no_grad():
    ch_emb_weights = channel_emb_layer.weight.cpu().numpy()  # (52, emb_dim)
    subj_emb_weights = subject_emb_layer.weight.cpu().numpy()  # (N, emb_dim)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Channel embeddings via PCA
if n_parcels > 3:
    pca_ch = PCA(n_components=2)
    ch_2d = pca_ch.fit_transform(ch_emb_weights)
    axes[0].scatter(ch_2d[:, 0], ch_2d[:, 1], c=range(n_parcels),
                    cmap="viridis", s=40, edgecolors="black", linewidth=0.5)
    for i in range(n_parcels):
        axes[0].annotate(str(i), (ch_2d[i, 0], ch_2d[i, 1]),
                         fontsize=6, ha="center", va="bottom")
    axes[0].set_title(f"Channel embeddings (PCA, {n_parcels} parcels)")
    axes[0].set_xlabel(f"PC1 ({pca_ch.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca_ch.explained_variance_ratio_[1]*100:.1f}%)")
else:
    axes[0].text(0.5, 0.5, "Too few channels for PCA",
                 ha="center", va="center", transform=axes[0].transAxes)

# Subject embeddings via PCA
if n_subjects > 3:
    pca_subj = PCA(n_components=2)
    subj_2d = pca_subj.fit_transform(subj_emb_weights)
    axes[1].scatter(subj_2d[:, 0], subj_2d[:, 1], c=range(n_subjects),
                    cmap="plasma", s=40, edgecolors="black", linewidth=0.5)
    for i in range(min(n_subjects, 30)):
        axes[1].annotate(subject_names[i][-6:], (subj_2d[i, 0], subj_2d[i, 1]),
                         fontsize=5, ha="center", va="bottom")
    axes[1].set_title(f"Subject embeddings (PCA, {n_subjects} subjects)")
    axes[1].set_xlabel(f"PC1 ({pca_subj.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca_subj.explained_variance_ratio_[1]*100:.1f}%)")
else:
    axes[1].text(0.5, 0.5, "Too few subjects for visualization",
                 ha="center", va="center", transform=axes[1].transAxes)

fig.tight_layout()
savefig(fig, "embedding_viz.png")

print(f"\nAll figures saved to {OUT_DIR}")
