"""Deep post-hoc analysis of real MEG DINO representations.

Produces figures in exp_real_meg/figures/:
  attention_by_region.png          – ConvNetV2 temporal attention profiles by brain region
  nn_retrieval.png                 – query + 8 nearest neighbours with (subject, channel) labels
  nn_cross_subject.png             – nearest neighbours across different subjects
  cluster_composition.png          – k-means cluster composition by channel/subject
  channel_similarity.png           – 52x52 cosine similarity + hierarchical clustering
  subject_similarity.png           – NxN subject cosine similarity
  cross_subject_consistency.png    – within-channel vs between-channel similarity
  motif_waveforms.png              – average waveforms per k-means cluster (motif)
  motif_examples.png               – example windows per motif cluster
  motif_frequency_per_subject.png  – heatmap of motif proportions per subject
  motif_age_correlation.png        – motif frequency vs age scatter plots
  feature_age_correlation.png      – PCA feature components vs age

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_real_meg/analyse.py
"""

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNetV2

# ------------------------------------------------------------------ config
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CKPT = os.path.join(CKPT_DIR, "backbone_final.pt")
META_FILE = os.path.join(CKPT_DIR, "metadata.json")
DATA_ROOT = "/well/win-camcan/shared/spring23/src"
PARTICIPANTS_FILE = "/well/win-camcan/shared/participants.tsv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

feat_dim = 256
n_parcels = 52
sampling_frequency = 250
eval_window_length = 112
eval_stride = 56
max_subjects_eval = 20
n_clusters = 20
n_neighbors = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ load metadata + model
print("\n--- Loading metadata + model ---")
with open(META_FILE) as f:
    meta = json.load(f)
subject_names = meta["subject_names"]
n_subjects = meta["n_subjects"]

backbone = ConvNetV2(
    in_channels=1, feat_dim=feat_dim,
    stem_channels=32, stem_kernel_sizes=(7, 15, 31),
    block_channels=(128, 256), block_kernel_sizes=(9, 5),
    attn_hidden=64,
).to(device)

state = torch.load(CKPT, map_location=device, weights_only=True)
backbone.load_state_dict(state["backbone"])
backbone.eval()
print("Backbone loaded.")


# ------------------------------------------------------------------ load age data
print("\n--- Loading participant age data ---")
age_map = {}  # subject_name -> age
if os.path.exists(PARTICIPANTS_FILE):
    with open(PARTICIPANTS_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid = parts[0]  # e.g. "sub-CC110033"
                try:
                    age = float(parts[1])
                    age_map[pid] = age
                except ValueError:
                    pass
    print(f"  Loaded ages for {len(age_map)} participants")
else:
    print(f"  WARNING: {PARTICIPANTS_FILE} not found, age analyses will be skipped")


# ------------------------------------------------------------------ extract features
print("\n--- Extracting features ---")
import mne
mne.set_log_level("WARNING")

fif_files = sorted(glob.glob(os.path.join(DATA_ROOT, "sub-*/sflip_parc-raw.fif")))
fif_files = fif_files[:max_subjects_eval]

all_feats = []
all_channel_ids = []
all_subject_ids = []
all_windows = []  # raw windows for visualization
eval_subject_names = []  # names of subjects actually loaded

for subj_idx, fif_path in enumerate(fif_files):
    subj_name = os.path.basename(os.path.dirname(fif_path))
    eval_subject_names.append(subj_name)
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
            for bs in range(0, len(windows_t), 256):
                batch = windows_t[bs:bs + 256]
                f = backbone(batch)
                f = F.normalize(f, dim=1)
                feats.append(f.cpu())
            feats = torch.cat(feats)

        all_feats.append(feats)
        all_channel_ids.extend([ch_idx] * len(feats))
        all_subject_ids.extend([subj_idx] * len(feats))
        all_windows.append(torch.from_numpy(windows))

    print(f"  {subj_name}: done")

all_feats = torch.cat(all_feats)
all_channel_ids = np.array(all_channel_ids)
all_subject_ids = np.array(all_subject_ids)
all_windows = torch.cat(all_windows)
print(f"Total: {all_feats.shape[0]} windows from {len(eval_subject_names)} subjects")


# ------------------------------------------------------------------ k-means clustering (used by multiple analyses)
print(f"\n--- K-means clustering (k={n_clusters}) ---")
n_km = min(100000, len(all_feats))
rng = np.random.RandomState(0)
km_idx = rng.choice(len(all_feats), size=n_km, replace=False)
km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
km.fit(all_feats[km_idx].numpy())

# Assign ALL windows to clusters (in chunks to save memory)
print("  Assigning all windows to clusters...")
all_cluster_labels = np.empty(len(all_feats), dtype=np.int32)
chunk_size = 50000
for start in range(0, len(all_feats), chunk_size):
    end = min(start + chunk_size, len(all_feats))
    all_cluster_labels[start:end] = km.predict(all_feats[start:end].numpy())
print(f"  Cluster sizes: min={np.bincount(all_cluster_labels).min()}, "
      f"max={np.bincount(all_cluster_labels).max()}")


# ======================================================================
# Analysis 1: Temporal attention profiles by brain region
# ======================================================================
print("\n--- Analysis 1: Attention by region ---")

# Group parcels into broad regions (approximate Glasser groupings)
region_size = n_parcels // 6
region_names = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Other"]
region_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def get_region(ch_idx):
    r = min(ch_idx // region_size, len(region_names) - 1)
    return r


# Collect attention weights per region
n_sample = min(5000, len(all_windows))
sample_idx = rng.choice(len(all_windows), size=n_sample, replace=False)

sample_windows = all_windows[sample_idx].to(device)
sample_ch = all_channel_ids[sample_idx]

with torch.no_grad():
    attn_list = []
    for bs in range(0, n_sample, 256):
        batch = sample_windows[bs:bs + 256]
        _, attn_w = backbone(batch, return_attention=True)
        attn_list.append(attn_w.cpu().numpy())
attn_all = np.concatenate(attn_list)  # (n_sample, T')
T_prime = attn_all.shape[1]

fig, ax = plt.subplots(figsize=(10, 5))
t_prime = np.arange(T_prime)
for r in range(len(region_names)):
    mask = np.array([get_region(ch) == r for ch in sample_ch])
    if mask.sum() == 0:
        continue
    avg_attn = attn_all[mask].mean(axis=0)
    ax.plot(t_prime, avg_attn, label=region_names[r], color=region_colors[r], linewidth=1.5)

ax.set_xlabel("Temporal position (downsampled)")
ax.set_ylabel("Attention weight")
ax.set_title("Average temporal attention by brain region")
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "attention_by_region.png")


# ======================================================================
# Analysis 2: NN retrieval
# ======================================================================
print("\n--- Analysis 2: NN retrieval ---")

# Pick diverse query windows (one per region)
n_queries = min(6, len(region_names))
queries = []
for r in range(n_queries):
    mask = np.array([get_region(ch) == r for ch in all_channel_ids])
    idx_r = np.where(mask)[0]
    if len(idx_r) == 0:
        continue
    feats_r = all_feats[idx_r]
    centroid = F.normalize(feats_r.mean(dim=0, keepdim=True), dim=1)
    sims = (feats_r @ centroid.t()).squeeze()
    queries.append(idx_r[sims.argmax().item()])


def find_nn(query_feat, all_feats, exclude_mask=None, k=8):
    """Find k nearest neighbours by cosine similarity, computed in chunks."""
    best_sims = torch.full((k,), -2.0)
    best_idx = torch.full((k,), -1, dtype=torch.long)
    chunk_size = 100000
    for start in range(0, len(all_feats), chunk_size):
        chunk = all_feats[start:start + chunk_size]
        sims = (chunk @ query_feat).squeeze()
        if exclude_mask is not None:
            sims[exclude_mask[start:start + chunk_size]] = -2.0
        combined_sims = torch.cat([best_sims, sims])
        combined_idx = torch.cat([best_idx, torch.arange(start, start + len(chunk))])
        topk = combined_sims.topk(k)
        best_sims = topk.values
        best_idx = combined_idx[topk.indices]
    return best_idx


fig, axes = plt.subplots(len(queries), n_neighbors + 1,
                          figsize=(2.2 * (n_neighbors + 1), 2.2 * len(queries)))
fig.suptitle("NN retrieval: query + nearest neighbours", fontsize=12, y=1.01)

for row, qi in enumerate(queries):
    q_ch = all_channel_ids[qi]
    q_subj = all_subject_ids[qi]
    q_feat = all_feats[qi:qi + 1].t()

    ax = axes[row, 0]
    signal = all_windows[qi, 0].numpy()
    t = np.arange(len(signal)) / sampling_frequency
    ax.plot(t, signal, color="black", linewidth=0.8)
    ax.set_title(f"Q: s{q_subj} ch{q_ch}", fontsize=7, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])

    exclude = torch.zeros(len(all_feats), dtype=torch.bool)
    exclude[qi] = True
    nn_idx = find_nn(q_feat, all_feats, exclude_mask=exclude, k=n_neighbors)

    for col, ni in enumerate(nn_idx):
        ni = ni.item()
        n_ch = all_channel_ids[ni]
        n_subj = all_subject_ids[ni]
        ax = axes[row, col + 1]
        signal = all_windows[ni, 0].numpy()
        t = np.arange(len(signal)) / sampling_frequency
        ax.plot(t, signal, color="black", linewidth=0.8)
        same_ch = n_ch == q_ch
        same_subj = n_subj == q_subj
        color = "green" if same_ch else ("orange" if same_subj else "red")
        ax.set_title(f"s{n_subj} ch{n_ch}", fontsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()
savefig(fig, "nn_retrieval.png")


# ======================================================================
# Analysis 3: Cross-subject NN retrieval
# ======================================================================
print("\n--- Analysis 3: Cross-subject NN ---")

fig, axes = plt.subplots(len(queries), n_neighbors + 1,
                          figsize=(2.2 * (n_neighbors + 1), 2.2 * len(queries)))
fig.suptitle("Cross-subject NN retrieval", fontsize=12, y=1.01)

for row, qi in enumerate(queries):
    q_ch = all_channel_ids[qi]
    q_subj = all_subject_ids[qi]
    q_feat = all_feats[qi:qi + 1].t()

    ax = axes[row, 0]
    signal = all_windows[qi, 0].numpy()
    t = np.arange(len(signal)) / sampling_frequency
    ax.plot(t, signal, color="black", linewidth=0.8)
    ax.set_title(f"Q: s{q_subj} ch{q_ch}", fontsize=7, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])

    exclude = torch.from_numpy(all_subject_ids == q_subj)
    nn_idx = find_nn(q_feat, all_feats, exclude_mask=exclude, k=n_neighbors)

    for col, ni in enumerate(nn_idx):
        ni = ni.item()
        n_ch = all_channel_ids[ni]
        n_subj = all_subject_ids[ni]
        ax = axes[row, col + 1]
        signal = all_windows[ni, 0].numpy()
        t = np.arange(len(signal)) / sampling_frequency
        ax.plot(t, signal, color="black", linewidth=0.8)
        color = "green" if n_ch == q_ch else "gray"
        ax.set_title(f"s{n_subj} ch{n_ch}", fontsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            spine.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()
savefig(fig, "nn_cross_subject.png")


# ======================================================================
# Analysis 4: K-means cluster composition
# ======================================================================
print(f"\n--- Analysis 4: Cluster composition (k={n_clusters}) ---")

km_ch_ids = all_channel_ids  # use full assignment now

# Channel composition per cluster
ch_comp = np.zeros((n_clusters, n_parcels))
for k in range(n_clusters):
    mask = all_cluster_labels == k
    for ch in range(n_parcels):
        ch_comp[k, ch] = (km_ch_ids[mask] == ch).sum()
    ch_comp[k] /= max(1, mask.sum())

fig, ax = plt.subplots(figsize=(14, 6))
bottom = np.zeros(n_clusters)
cmap = plt.colormaps["tab20"]
for ch in range(n_parcels):
    ax.bar(range(n_clusters), ch_comp[:, ch], bottom=bottom,
           color=cmap(ch % 20), width=0.8)
    bottom += ch_comp[:, ch]
ax.set_xlabel("Cluster")
ax.set_ylabel("Proportion")
ax.set_title(f"Channel composition per k-means cluster (k={n_clusters})")
ax.set_xticks(range(n_clusters))
fig.tight_layout()
savefig(fig, "cluster_composition.png")


# ======================================================================
# Analysis 5: Channel similarity matrix
# ======================================================================
print("\n--- Analysis 5: Channel similarity ---")

ch_centroids = torch.zeros(n_parcels, feat_dim)
for ch in range(n_parcels):
    mask = all_channel_ids == ch
    if mask.sum() > 0:
        ch_centroids[ch] = all_feats[mask].mean(dim=0)
ch_centroids = F.normalize(ch_centroids, dim=1)

ch_sim = (ch_centroids @ ch_centroids.t()).numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im = axes[0].imshow(ch_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
axes[0].set_xlabel("Channel")
axes[0].set_ylabel("Channel")
axes[0].set_title("Channel-channel cosine similarity")
plt.colorbar(im, ax=axes[0], shrink=0.8)

cos_dist = 1.0 - ch_sim
np.fill_diagonal(cos_dist, 0)
cos_dist = (cos_dist + cos_dist.T) / 2
cos_dist = np.clip(cos_dist, 0, None)
condensed = squareform(cos_dist)
Z = linkage(condensed, method="average")
dendrogram(Z, labels=[str(i) for i in range(n_parcels)],
           ax=axes[1], leaf_font_size=7, color_threshold=0.5)
axes[1].set_ylabel("Cosine distance")
axes[1].set_title("Hierarchical clustering of parcels")

fig.tight_layout()
savefig(fig, "channel_similarity.png")


# ======================================================================
# Analysis 6: Subject similarity matrix
# ======================================================================
print("\n--- Analysis 6: Subject similarity ---")

n_eval_subjects = len(eval_subject_names)
subj_centroids = torch.zeros(n_eval_subjects, feat_dim)
for si in range(n_eval_subjects):
    mask = all_subject_ids == si
    if mask.sum() > 0:
        subj_centroids[si] = all_feats[mask].mean(dim=0)
subj_centroids = F.normalize(subj_centroids, dim=1)

subj_sim = (subj_centroids @ subj_centroids.t()).numpy()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(subj_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
ax.set_xlabel("Subject")
ax.set_ylabel("Subject")
ax.set_title(f"Subject-subject cosine similarity ({n_eval_subjects} subjects)")
short_names = [s[-6:] for s in eval_subject_names]
ax.set_xticks(range(n_eval_subjects))
ax.set_xticklabels(short_names, rotation=90, fontsize=7)
ax.set_yticks(range(n_eval_subjects))
ax.set_yticklabels(short_names, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
savefig(fig, "subject_similarity.png")


# ======================================================================
# Analysis 7: Cross-subject consistency
# ======================================================================
print("\n--- Analysis 7: Cross-subject consistency ---")

within_ch_sims = []
between_ch_sims = []

for ch in range(n_parcels):
    per_subj = []
    for si in range(n_eval_subjects):
        mask = (all_channel_ids == ch) & (all_subject_ids == si)
        if mask.sum() > 5:
            cent = F.normalize(all_feats[mask].mean(dim=0, keepdim=True), dim=1)
            per_subj.append((si, cent))

    if len(per_subj) < 2:
        continue

    for i in range(len(per_subj)):
        for j in range(i + 1, len(per_subj)):
            sim = (per_subj[i][1] @ per_subj[j][1].t()).item()
            within_ch_sims.append(sim)

    for other_ch in range(n_parcels):
        if other_ch == ch:
            continue
        other_per_subj = []
        for si in range(n_eval_subjects):
            mask = (all_channel_ids == other_ch) & (all_subject_ids == si)
            if mask.sum() > 5:
                cent = F.normalize(all_feats[mask].mean(dim=0, keepdim=True), dim=1)
                other_per_subj.append(cent)
        if len(other_per_subj) == 0:
            continue
        for ps in per_subj:
            for ops in other_per_subj:
                sim = (ps[1] @ ops.t()).item()
                between_ch_sims.append(sim)

within_ch_sims = np.array(within_ch_sims)
between_ch_sims = np.array(between_ch_sims)

if len(between_ch_sims) > 50000:
    between_ch_sims_plot = rng.choice(between_ch_sims, size=50000, replace=False)
else:
    between_ch_sims_plot = between_ch_sims

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(within_ch_sims, bins=50, alpha=0.7, color="steelblue",
             label=f"Within-channel (n={len(within_ch_sims)})", density=True)
axes[0].hist(between_ch_sims_plot, bins=50, alpha=0.5, color="salmon",
             label=f"Between-channel (n={len(between_ch_sims)})", density=True)
axes[0].axvline(within_ch_sims.mean(), color="blue", linestyle="--", linewidth=1.5,
                label=f"Within mean: {within_ch_sims.mean():.3f}")
axes[0].axvline(between_ch_sims_plot.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Between mean: {between_ch_sims_plot.mean():.3f}")
axes[0].set_xlabel("Cosine similarity")
axes[0].set_ylabel("Density")
axes[0].set_title("Cross-subject consistency")
axes[0].legend(fontsize=8)

per_ch_within = []
for ch in range(n_parcels):
    per_subj = []
    for si in range(n_eval_subjects):
        mask = (all_channel_ids == ch) & (all_subject_ids == si)
        if mask.sum() > 5:
            cent = F.normalize(all_feats[mask].mean(dim=0, keepdim=True), dim=1)
            per_subj.append(cent)
    if len(per_subj) < 2:
        per_ch_within.append(0.0)
        continue
    stacked = torch.cat(per_subj)
    sim_mat = (stacked @ stacked.t()).numpy()
    n = len(per_subj)
    triu = sim_mat[np.triu_indices(n, k=1)]
    per_ch_within.append(triu.mean())

axes[1].bar(range(n_parcels), per_ch_within, color="steelblue", alpha=0.8)
axes[1].axhline(np.mean(per_ch_within), color="red", linestyle="--",
                label=f"Mean: {np.mean(per_ch_within):.3f}")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Mean within-channel similarity")
axes[1].set_title("Cross-subject consistency per channel")
axes[1].legend()

fig.tight_layout()
savefig(fig, "cross_subject_consistency.png")


# ======================================================================
# Analysis 8: Motif waveforms (average waveform per cluster)
# ======================================================================
print(f"\n--- Analysis 8: Motif waveforms (k={n_clusters}) ---")

# Compute average waveform per cluster
t_axis = np.arange(eval_window_length) / sampling_frequency

# Sort clusters by size (largest first)
cluster_sizes = np.bincount(all_cluster_labels, minlength=n_clusters)
cluster_order = np.argsort(-cluster_sizes)

n_cols = 5
n_rows = (n_clusters + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
axes = axes.flatten()

for plot_idx, k in enumerate(cluster_order):
    ax = axes[plot_idx]
    mask = all_cluster_labels == k
    n_in_cluster = mask.sum()

    # Subsample if too many windows
    cluster_idx = np.where(mask)[0]
    if len(cluster_idx) > 5000:
        sub_idx = rng.choice(cluster_idx, size=5000, replace=False)
    else:
        sub_idx = cluster_idx

    waveforms = all_windows[sub_idx, 0].numpy()  # (N, T)
    mean_waveform = waveforms.mean(axis=0)
    std_waveform = waveforms.std(axis=0)

    ax.fill_between(t_axis, mean_waveform - std_waveform, mean_waveform + std_waveform,
                     alpha=0.2, color="steelblue")
    ax.plot(t_axis, mean_waveform, color="steelblue", linewidth=1.5)
    ax.set_title(f"Motif {k} (n={n_in_cluster})", fontsize=8)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.tick_params(labelsize=6)
    if plot_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Time (s)", fontsize=7)

# Hide unused axes
for i in range(n_clusters, len(axes)):
    axes[i].set_visible(False)

fig.suptitle(f"Average waveform per motif cluster (k={n_clusters})", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "motif_waveforms.png")


# ======================================================================
# Analysis 9: Motif examples (random example windows per cluster)
# ======================================================================
print(f"\n--- Analysis 9: Motif examples ---")

n_examples = 8
fig, axes = plt.subplots(n_clusters, n_examples + 1,
                          figsize=(2 * (n_examples + 1), 1.8 * n_clusters))
fig.suptitle("Motif examples: mean + 8 random windows per cluster", fontsize=12, y=1.01)

for row, k in enumerate(cluster_order):
    mask = all_cluster_labels == k
    cluster_idx = np.where(mask)[0]

    # Mean waveform
    ax = axes[row, 0]
    if len(cluster_idx) > 5000:
        sub_idx = rng.choice(cluster_idx, size=5000, replace=False)
    else:
        sub_idx = cluster_idx
    mean_wf = all_windows[sub_idx, 0].numpy().mean(axis=0)
    ax.plot(t_axis, mean_wf, color="steelblue", linewidth=1.2)
    ax.set_ylabel(f"M{k}", fontsize=7, rotation=0, labelpad=15)
    for spine in ax.spines.values():
        spine.set_edgecolor("steelblue")
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Example windows
    example_idx = rng.choice(cluster_idx, size=min(n_examples, len(cluster_idx)), replace=False)
    for col, ei in enumerate(example_idx):
        ax = axes[row, col + 1]
        signal = all_windows[ei, 0].numpy()
        ax.plot(t_axis, signal, color="black", linewidth=0.6)
        ch = all_channel_ids[ei]
        subj = all_subject_ids[ei]
        ax.set_title(f"s{subj} ch{ch}", fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused columns
    for col in range(len(example_idx) + 1, n_examples + 1):
        axes[row, col].set_visible(False)

fig.tight_layout()
savefig(fig, "motif_examples.png")


# ======================================================================
# Analysis 10: Motif frequency per subject
# ======================================================================
print("\n--- Analysis 10: Motif frequency per subject ---")

# Build motif frequency matrix: (n_eval_subjects, n_clusters)
motif_freq = np.zeros((n_eval_subjects, n_clusters))
for si in range(n_eval_subjects):
    mask = all_subject_ids == si
    labels_si = all_cluster_labels[mask]
    counts = np.bincount(labels_si, minlength=n_clusters).astype(float)
    motif_freq[si] = counts / max(1, counts.sum())

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(motif_freq[:, cluster_order].T, aspect="auto", cmap="YlOrRd")
ax.set_xlabel("Subject")
ax.set_ylabel("Motif cluster")
ax.set_title("Motif frequency per subject (proportion of windows)")
ax.set_xticks(range(n_eval_subjects))
ax.set_xticklabels([s[-6:] for s in eval_subject_names], rotation=90, fontsize=7)
ax.set_yticks(range(n_clusters))
ax.set_yticklabels([f"M{k}" for k in cluster_order], fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")
fig.tight_layout()
savefig(fig, "motif_frequency_per_subject.png")


# ======================================================================
# Analysis 11: Motif frequency vs age
# ======================================================================
if age_map:
    print("\n--- Analysis 11: Motif frequency vs age ---")

    # Get ages for eval subjects
    eval_ages = []
    age_valid = []
    for si, sname in enumerate(eval_subject_names):
        if sname in age_map:
            eval_ages.append(age_map[sname])
            age_valid.append(si)
    eval_ages = np.array(eval_ages)
    age_valid = np.array(age_valid)

    if len(eval_ages) >= 5:
        print(f"  {len(eval_ages)} subjects with age data "
              f"(range: {eval_ages.min():.1f} - {eval_ages.max():.1f})")

        # Find motifs with strongest age correlations
        correlations = []
        for k in range(n_clusters):
            freq_k = motif_freq[age_valid, k]
            r, p = spearmanr(eval_ages, freq_k)
            correlations.append({"cluster": k, "r": r, "p": p})

        correlations.sort(key=lambda x: abs(x["r"]), reverse=True)

        # Plot top 8 motifs by |correlation|
        n_plot = min(8, n_clusters)
        n_cols_age = 4
        n_rows_age = (n_plot + n_cols_age - 1) // n_cols_age
        fig, axes = plt.subplots(n_rows_age, n_cols_age,
                                  figsize=(4 * n_cols_age, 3.5 * n_rows_age))
        axes = axes.flatten()

        for i in range(n_plot):
            ax = axes[i]
            k = correlations[i]["cluster"]
            r_val = correlations[i]["r"]
            p_val = correlations[i]["p"]
            freq_k = motif_freq[age_valid, k]

            ax.scatter(eval_ages, freq_k, s=30, alpha=0.7, color="steelblue",
                       edgecolors="black", linewidth=0.3)

            # Trend line
            z = np.polyfit(eval_ages, freq_k, 1)
            age_range = np.linspace(eval_ages.min(), eval_ages.max(), 50)
            ax.plot(age_range, np.polyval(z, age_range), color="red",
                    linewidth=1.5, linestyle="--")

            sig = "*" if p_val < 0.05 else ""
            ax.set_title(f"Motif {k}: r={r_val:.2f}, p={p_val:.3f}{sig}", fontsize=9)
            ax.set_xlabel("Age (years)", fontsize=8)
            ax.set_ylabel("Motif proportion", fontsize=8)
            ax.tick_params(labelsize=7)

        for i in range(n_plot, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle("Motif frequency vs age (Spearman correlation)", fontsize=13, y=1.01)
        fig.tight_layout()
        savefig(fig, "motif_age_correlation.png")

        # Print correlation summary
        print("  Top age-correlated motifs:")
        for c in correlations[:5]:
            sig = "*" if c["p"] < 0.05 else ""
            print(f"    Motif {c['cluster']}: r={c['r']:.3f}, p={c['p']:.4f}{sig}")

        # ======================================================================
        # Analysis 12: Feature PCA vs age
        # ======================================================================
        print("\n--- Analysis 12: Feature PCA vs age ---")

        # Compute per-subject mean feature vector
        subj_mean_feats = np.zeros((n_eval_subjects, feat_dim))
        for si in range(n_eval_subjects):
            mask = all_subject_ids == si
            if mask.sum() > 0:
                subj_mean_feats[si] = all_feats[mask].mean(dim=0).numpy()

        pca = PCA(n_components=5)
        subj_pca = pca.fit_transform(subj_mean_feats)

        # Plot top 4 PCs vs age
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for pc in range(4):
            ax = axes[pc]
            pc_vals = subj_pca[age_valid, pc]
            ax.scatter(eval_ages, pc_vals, s=30, alpha=0.7, color="steelblue",
                       edgecolors="black", linewidth=0.3)

            r_val, p_val = spearmanr(eval_ages, pc_vals)
            z = np.polyfit(eval_ages, pc_vals, 1)
            age_range = np.linspace(eval_ages.min(), eval_ages.max(), 50)
            ax.plot(age_range, np.polyval(z, age_range), color="red",
                    linewidth=1.5, linestyle="--")

            var_pct = pca.explained_variance_ratio_[pc] * 100
            sig = "*" if p_val < 0.05 else ""
            ax.set_title(f"PC{pc+1} ({var_pct:.1f}%): r={r_val:.2f}{sig}", fontsize=9)
            ax.set_xlabel("Age (years)", fontsize=8)
            ax.set_ylabel(f"PC{pc+1}", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle("Subject mean feature PCs vs age", fontsize=13, y=1.02)
        fig.tight_layout()
        savefig(fig, "feature_age_correlation.png")

    else:
        print(f"  Only {len(eval_ages)} subjects with age data, skipping age analyses")
else:
    print("\n--- Skipping age analyses (no participant data) ---")


# ======================================================================
print(f"\nAll figures saved to {OUT_DIR}")
