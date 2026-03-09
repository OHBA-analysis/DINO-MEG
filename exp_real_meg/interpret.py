"""Motif interpretability analysis for real MEG DINO representations.

Produces figures in exp_real_meg/figures/:
  stem_filters_waveforms.png           – 96 learned conv filters as temporal waveforms
  stem_filters_frequency.png           – FFT magnitude (frequency response) of each filter
  maximally_activating_windows.png     – top-8 filters x 8 maximally activating input windows
  motif_psd.png                        – per-motif Welch PSD grid (20 panels)
  motif_psd_overlay.png                – top 10 motifs PSD on same axes
  optimal_input_waveforms.png          – gradient-ascent synthesised ideal waveform per motif
  optimal_input_spectra.png            – FFT of synthesised waveforms
  motif_frequency_saliency.png         – frequency-domain input gradient saliency per motif
  stem_branch_ablation.png             – feature disruption from zeroing each stem branch
  stem_branch_per_motif.png            – per-branch mean |activation| per motif
  stem_branch_per_motif_normalized.png – normalised stacked bar of branch proportions
  feature_frequency_heatmap.png        – feature dim x band-power Pearson correlation
  feature_frequency_top_neurons.png    – scatter plots for top band-selective neurons
  motif_temporal_context.png           – eventplot timelines + burstiness per motif

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_real_meg/interpret.py
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
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNetV2

# ------------------------------------------------------------------ config
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CKPT = os.path.join(CKPT_DIR, "backbone_final.pt")
META_FILE = os.path.join(CKPT_DIR, "metadata.json")
DATA_ROOT = "/well/win-camcan/shared/spring23/src"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

feat_dim = 256
n_parcels = 52
sampling_frequency = 250
eval_window_length = 112
eval_stride = 56
max_subjects_eval = 20
n_clusters = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

rng = np.random.RandomState(0)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ load model
print("\n--- Loading metadata + model ---")
with open(META_FILE) as f:
    meta = json.load(f)
subject_names = meta["subject_names"]

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


# ------------------------------------------------------------------ extract features
print("\n--- Extracting features ---")
import mne
mne.set_log_level("WARNING")

fif_files = sorted(glob.glob(os.path.join(DATA_ROOT, "sub-*/sflip_parc-raw.fif")))
fif_files = fif_files[:max_subjects_eval]

all_feats = []
all_channel_ids = []
all_subject_ids = []
all_windows = []
all_time_offsets = []
eval_subject_names = []

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
        all_time_offsets.extend(starts)

    print(f"  {subj_name}: done")

all_feats = torch.cat(all_feats)
all_channel_ids = np.array(all_channel_ids)
all_subject_ids = np.array(all_subject_ids)
all_windows = torch.cat(all_windows)
all_time_offsets = np.array(all_time_offsets)
print(f"Total: {all_feats.shape[0]} windows from {len(eval_subject_names)} subjects")


# ------------------------------------------------------------------ k-means
print(f"\n--- K-means clustering (k={n_clusters}) ---")
n_km = min(100000, len(all_feats))
km_idx = rng.choice(len(all_feats), size=n_km, replace=False)
km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
km.fit(all_feats[km_idx].numpy())

all_cluster_labels = np.empty(len(all_feats), dtype=np.int32)
chunk_size = 50000
for start in range(0, len(all_feats), chunk_size):
    end = min(start + chunk_size, len(all_feats))
    all_cluster_labels[start:end] = km.predict(all_feats[start:end].numpy())

cluster_sizes = np.bincount(all_cluster_labels, minlength=n_clusters)
cluster_order = np.argsort(-cluster_sizes)
print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}")

# Compute cluster centroids from features
centroids = torch.zeros(n_clusters, feat_dim)
for k in range(n_clusters):
    mask = all_cluster_labels == k
    centroids[k] = all_feats[mask].mean(dim=0)
centroids = F.normalize(centroids, dim=1)


# ======================================================================
# Analysis 1: Stem Filter Visualization
# ======================================================================
print("\n--- Analysis 1: Stem Filter Visualization ---")

stem_kernel_sizes = [7, 15, 31]
branch_colors = ["#4c72b0", "#dd8452", "#55a868"]
branch_labels = [f"k={k}" for k in stem_kernel_sizes]

# Extract filter weights
all_filters = []
for i, branch in enumerate(backbone.stem_branches):
    w = branch[0].weight.detach().cpu().numpy()  # (32, 1, kernel_size)
    all_filters.append(w)

# Figure: stem_filters_waveforms.png
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle("Stem filter waveforms (3 branches x 32 filters)", fontsize=13, y=1.01)

for b, (w, ks) in enumerate(zip(all_filters, stem_kernel_sizes)):
    ax = axes[b]
    n_f = w.shape[0]
    t = np.arange(ks) / sampling_frequency * 1000  # ms
    for fi in range(n_f):
        offset = fi * 1.5
        filt = w[fi, 0]
        filt_norm = filt / (np.abs(filt).max() + 1e-8)
        ax.plot(t, filt_norm + offset, color=branch_colors[b], linewidth=0.8)
    ax.set_title(f"Branch {b}: kernel={ks} ({ks/sampling_frequency*1000:.0f} ms)",
                 fontsize=10)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Filter index (offset)")
    ax.set_yticks(np.arange(0, n_f * 1.5, 1.5 * 4))
    ax.set_yticklabels(range(0, n_f, 4))

fig.tight_layout()
savefig(fig, "stem_filters_waveforms.png")

# Figure: stem_filters_frequency.png
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle("Stem filter frequency responses", fontsize=13, y=1.01)

for b, (w, ks) in enumerate(zip(all_filters, stem_kernel_sizes)):
    ax = axes[b]
    n_f = w.shape[0]
    for fi in range(n_f):
        filt = w[fi, 0]
        fft_mag = np.abs(np.fft.rfft(filt, n=256))
        freqs = np.fft.rfftfreq(256, d=1.0 / sampling_frequency)
        fft_mag /= fft_mag.max() + 1e-8
        offset = fi * 1.2
        ax.plot(freqs, fft_mag + offset, color=branch_colors[b], linewidth=0.8)
    ax.set_title(f"Branch {b}: kernel={ks}", fontsize=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Filter index (offset)")
    ax.set_xlim(0, 125)
    ax.set_yticks(np.arange(0, n_f * 1.2, 1.2 * 4))
    ax.set_yticklabels(range(0, n_f, 4))

fig.tight_layout()
savefig(fig, "stem_filters_frequency.png")


# ======================================================================
# Analysis 2: Maximally Activating Windows
# ======================================================================
print("\n--- Analysis 2: Maximally Activating Windows ---")

# Run all windows through stem branches in chunks, record max activation per filter
n_total_filters = sum(w.shape[0] for w in all_filters)
n_windows = len(all_windows)

# Compute max activations per filter per window
max_acts = np.zeros((n_windows, n_total_filters), dtype=np.float32)

for start in range(0, n_windows, 512):
    end = min(start + 512, n_windows)
    batch = all_windows[start:end].to(device)
    with torch.no_grad():
        fi_offset = 0
        for branch in backbone.stem_branches:
            act = branch(batch)  # (B, 32, T)
            # max activation over time per filter
            max_per_filter = act.max(dim=2).values.cpu().numpy()  # (B, 32)
            max_acts[start:end, fi_offset:fi_offset + act.shape[1]] = max_per_filter
            fi_offset += act.shape[1]

# Select top-8 filters by activation variance (most discriminative)
filter_var = max_acts.var(axis=0)
top_filter_idx = np.argsort(-filter_var)[:8]

# Map filter index to branch/filter
def filter_to_branch(fi):
    offset = 0
    for b, w in enumerate(all_filters):
        if fi < offset + w.shape[0]:
            return b, fi - offset
        offset += w.shape[0]
    return -1, -1

fig, axes = plt.subplots(8, 9, figsize=(22, 20))
fig.suptitle("Maximally activating windows for top-8 discriminative filters", fontsize=13, y=1.01)

for row, fi in enumerate(top_filter_idx):
    b_idx, f_idx = filter_to_branch(fi)
    ks = stem_kernel_sizes[b_idx]

    # Show filter waveform
    ax = axes[row, 0]
    filt = all_filters[b_idx][f_idx, 0]
    t_f = np.arange(len(filt)) / sampling_frequency * 1000
    ax.plot(t_f, filt, color=branch_colors[b_idx], linewidth=1.5)
    ax.set_title(f"B{b_idx} F{f_idx} (k={ks})", fontsize=7, fontweight="bold")
    ax.set_xlabel("ms", fontsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor(branch_colors[b_idx])
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax.tick_params(labelsize=5)

    # Show 8 maximally activating windows
    top_windows = np.argsort(-max_acts[:, fi])[:8]
    for col, wi in enumerate(top_windows):
        ax = axes[row, col + 1]
        signal = all_windows[wi, 0].numpy()
        t = np.arange(len(signal)) / sampling_frequency
        ax.plot(t, signal, color="black", linewidth=0.6)
        ax.set_title(f"act={max_acts[wi, fi]:.1f}", fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()
savefig(fig, "maximally_activating_windows.png")

del max_acts  # free memory


# ======================================================================
# Analysis 3: Per-Motif Spectral Analysis
# ======================================================================
print("\n--- Analysis 3: Per-Motif Spectral Analysis ---")

t_axis = np.arange(eval_window_length) / sampling_frequency

# Compute Welch PSD per cluster
n_cols = 5
n_rows = (n_clusters + n_cols - 1) // n_cols

# Pre-compute PSD params
nperseg = min(64, eval_window_length)
psd_freqs = None
motif_psds_mean = {}
motif_psds_std = {}

for k in range(n_clusters):
    mask = all_cluster_labels == k
    cluster_idx = np.where(mask)[0]
    n_sub = min(2000, len(cluster_idx))
    sub_idx = rng.choice(cluster_idx, size=n_sub, replace=False)

    psds = []
    for wi in sub_idx:
        signal = all_windows[wi, 0].numpy()
        f, pxx = welch(signal, fs=sampling_frequency, nperseg=nperseg)
        psds.append(pxx)
        if psd_freqs is None:
            psd_freqs = f

    psds = np.array(psds)
    motif_psds_mean[k] = psds.mean(axis=0)
    motif_psds_std[k] = psds.std(axis=0)

# Figure: motif_psd.png — grid of 20 panels
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
axes_flat = axes.flatten()

for plot_idx, k in enumerate(cluster_order):
    ax = axes_flat[plot_idx]
    mean_psd = motif_psds_mean[k]
    std_psd = motif_psds_std[k]
    ax.semilogy(psd_freqs, mean_psd, color="steelblue", linewidth=1.2)
    ax.fill_between(psd_freqs, mean_psd - std_psd, mean_psd + std_psd,
                    alpha=0.2, color="steelblue")
    ax.set_title(f"Motif {k} (n={cluster_sizes[k]})", fontsize=8)
    ax.set_xlim(0, 80)
    ax.tick_params(labelsize=6)
    if plot_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Freq (Hz)", fontsize=7)
    if plot_idx % n_cols == 0:
        ax.set_ylabel("PSD", fontsize=7)

for i in range(n_clusters, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Power spectral density per motif cluster", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "motif_psd.png")

# Figure: motif_psd_overlay.png — top 10 motifs on same axes
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.colormaps["tab20"]
for i, k in enumerate(cluster_order[:10]):
    ax.semilogy(psd_freqs, motif_psds_mean[k], color=cmap(i), linewidth=1.5,
                label=f"Motif {k} (n={cluster_sizes[k]})")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (log scale)")
ax.set_title("PSD comparison: top 10 motifs by cluster size")
ax.set_xlim(0, 80)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "motif_psd_overlay.png")


# ======================================================================
# Analysis 4: Optimal Input Synthesis
# ======================================================================
print("\n--- Analysis 4: Optimal Input Synthesis ---")

synth_waveforms = []
synth_spectra = []

for k in range(n_clusters):
    centroid_k = centroids[k].to(device)

    x = torch.randn(1, 1, eval_window_length, device=device) * 0.1
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.01)

    for step_i in range(500):
        opt.zero_grad()
        feat = backbone(x)
        feat = F.normalize(feat, dim=1)
        cos_sim = (feat * centroid_k.unsqueeze(0)).sum()
        loss = -cos_sim + 0.001 * x.pow(2).sum()  # L2 regularisation
        loss.backward()
        opt.step()

    synth = x.detach().cpu().squeeze().numpy()
    synth_waveforms.append(synth)
    fft_mag = np.abs(np.fft.rfft(synth))
    synth_spectra.append(fft_mag)

    if (k + 1) % 5 == 0:
        print(f"  Synthesised {k + 1}/{n_clusters} motifs")

# Figure: optimal_input_waveforms.png
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
axes_flat = axes.flatten()

for plot_idx, k in enumerate(cluster_order):
    ax = axes_flat[plot_idx]
    t = np.arange(eval_window_length) / sampling_frequency
    ax.plot(t, synth_waveforms[k], color="steelblue", linewidth=1.2)
    ax.set_title(f"Motif {k}", fontsize=8)
    ax.tick_params(labelsize=6)
    if plot_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Time (s)", fontsize=7)

for i in range(n_clusters, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Optimal input waveforms (gradient ascent)", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "optimal_input_waveforms.png")

# Figure: optimal_input_spectra.png
fft_freqs = np.fft.rfftfreq(eval_window_length, d=1.0 / sampling_frequency)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
axes_flat = axes.flatten()

for plot_idx, k in enumerate(cluster_order):
    ax = axes_flat[plot_idx]
    ax.plot(fft_freqs, synth_spectra[k], color="darkorange", linewidth=1.2)
    ax.set_title(f"Motif {k}", fontsize=8)
    ax.set_xlim(0, 80)
    ax.tick_params(labelsize=6)
    if plot_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Freq (Hz)", fontsize=7)

for i in range(n_clusters, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Optimal input spectra (FFT magnitude)", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "optimal_input_spectra.png")


# ======================================================================
# Analysis 4b: Input Gradient Saliency in Frequency Domain
# ======================================================================
print("\n--- Analysis 4b: Frequency-domain gradient saliency ---")

n_saliency_windows = 500

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
axes_flat = axes.flatten()

for plot_idx, k in enumerate(cluster_order):
    mask = all_cluster_labels == k
    cluster_idx = np.where(mask)[0]
    n_use = min(n_saliency_windows, len(cluster_idx))
    sub_idx = rng.choice(cluster_idx, size=n_use, replace=False)

    centroid_k = centroids[k].to(device)
    grad_ffts = []

    for bs in range(0, n_use, 64):
        batch_idx = sub_idx[bs:bs + 64]
        x = all_windows[batch_idx].to(device).requires_grad_(True)
        feat = backbone(x)
        feat = F.normalize(feat, dim=1)
        cos_sim = (feat * centroid_k.unsqueeze(0)).sum(dim=1).sum()
        cos_sim.backward()

        grad = x.grad.detach().cpu().numpy()[:, 0, :]  # (B, T)
        for g in grad:
            grad_ffts.append(np.abs(np.fft.rfft(g)))

    grad_ffts = np.array(grad_ffts)
    mean_saliency = grad_ffts.mean(axis=0)

    ax = axes_flat[plot_idx]
    ax.plot(fft_freqs, mean_saliency, color="darkred", linewidth=1.2)
    ax.set_title(f"Motif {k}", fontsize=8)
    ax.set_xlim(0, 80)
    ax.tick_params(labelsize=6)
    if plot_idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Freq (Hz)", fontsize=7)

for i in range(n_clusters, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Frequency-domain input gradient saliency per motif", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "motif_frequency_saliency.png")


# ======================================================================
# Analysis 4c: Feature Ablation by Stem Branch
# ======================================================================
print("\n--- Analysis 4c: Stem Branch Ablation ---")

# Subsample windows for ablation study
n_ablation = min(10000, len(all_windows))
abl_idx = rng.choice(len(all_windows), size=n_ablation, replace=False)
abl_labels = all_cluster_labels[abl_idx]

# Compute baseline features
with torch.no_grad():
    baseline_feats = []
    for bs in range(0, n_ablation, 256):
        batch = all_windows[abl_idx[bs:bs + 256]].to(device)
        f = backbone(batch)
        f = F.normalize(f, dim=1)
        baseline_feats.append(f.cpu())
    baseline_feats = torch.cat(baseline_feats)

# Ablate each branch
n_branches = len(backbone.stem_branches)
ablation_dists = np.zeros((n_ablation, n_branches))

for b in range(n_branches):
    with torch.no_grad():
        ablated_feats = []
        for bs in range(0, n_ablation, 256):
            batch = all_windows[abl_idx[bs:bs + 256]].to(device)
            # Run stem with one branch zeroed
            branches = []
            for bi, branch in enumerate(backbone.stem_branches):
                if bi == b:
                    act = torch.zeros_like(branch(batch))
                else:
                    act = branch(batch)
                branches.append(act)
            x = torch.cat(branches, dim=1)
            for block in backbone.blocks:
                x = block(x)
            x_t = x.transpose(1, 2)
            attn_logits = backbone.attn_net(x_t).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=-1)
            pooled = (x_t * attn_weights.unsqueeze(-1)).sum(dim=1)
            feat = backbone.head(pooled)
            feat = F.normalize(feat, dim=1)
            ablated_feats.append(feat.cpu())
        ablated_feats = torch.cat(ablated_feats)

    dist = (baseline_feats - ablated_feats).pow(2).sum(dim=1).sqrt().numpy()
    ablation_dists[:, b] = dist

# Per-motif ablation: mean distance per cluster per branch
motif_ablation = np.zeros((n_clusters, n_branches))
motif_ablation_sem = np.zeros((n_clusters, n_branches))
for k in range(n_clusters):
    mask = abl_labels == k
    if mask.sum() == 0:
        continue
    for b in range(n_branches):
        vals = ablation_dists[mask, b]
        motif_ablation[k, b] = vals.mean()
        motif_ablation_sem[k, b] = vals.std() / np.sqrt(len(vals))

# Figure: stem_branch_ablation.png
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(n_clusters)
width = 0.8 / n_branches
branch_colors_abl = ["#4c72b0", "#dd8452", "#55a868"]

for b in range(n_branches):
    offset = (b - n_branches / 2 + 0.5) * width
    ax.bar(x_pos + offset, motif_ablation[cluster_order, b], width,
           yerr=motif_ablation_sem[cluster_order, b],
           label=branch_labels[b], color=branch_colors_abl[b], alpha=0.85, capsize=2)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"M{k}" for k in cluster_order], fontsize=8)
ax.set_xlabel("Motif cluster")
ax.set_ylabel("Feature L2 distance from baseline")
ax.set_title("Feature disruption from ablating each stem branch")
ax.legend(title="Branch", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig(fig, "stem_branch_ablation.png")


# ======================================================================
# Analysis 5: Stem Branch Activation per Motif
# ======================================================================
print("\n--- Analysis 5: Stem Branch Activation per Motif ---")

# Compute per-branch mean |activation| on subsampled windows
n_stem = min(20000, len(all_windows))
stem_idx = rng.choice(len(all_windows), size=n_stem, replace=False)
stem_labels = all_cluster_labels[stem_idx]

branch_means = np.zeros((n_stem, n_branches))
with torch.no_grad():
    for bs in range(0, n_stem, 256):
        batch = all_windows[stem_idx[bs:bs + 256]].to(device)
        for b, branch in enumerate(backbone.stem_branches):
            act = branch(batch)
            branch_means[bs:bs + len(batch), b] = act.abs().mean(dim=(1, 2)).cpu().numpy()

# Grouped bar chart
motif_branch_mean = np.zeros((n_clusters, n_branches))
motif_branch_sem = np.zeros((n_clusters, n_branches))
for k in range(n_clusters):
    mask = stem_labels == k
    if mask.sum() == 0:
        continue
    for b in range(n_branches):
        vals = branch_means[mask, b]
        motif_branch_mean[k, b] = vals.mean()
        motif_branch_sem[k, b] = vals.std() / np.sqrt(len(vals))

# Figure: stem_branch_per_motif.png
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(n_clusters)
width = 0.8 / n_branches

for b in range(n_branches):
    offset = (b - n_branches / 2 + 0.5) * width
    ax.bar(x_pos + offset, motif_branch_mean[cluster_order, b], width,
           yerr=motif_branch_sem[cluster_order, b],
           label=branch_labels[b], color=branch_colors[b], alpha=0.85, capsize=2)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"M{k}" for k in cluster_order], fontsize=8)
ax.set_xlabel("Motif cluster")
ax.set_ylabel("Mean |activation|")
ax.set_title("Stem branch activation by motif cluster")
ax.legend(title="Branch kernel", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig(fig, "stem_branch_per_motif.png")

# Figure: stem_branch_per_motif_normalized.png — stacked proportions
branch_total = motif_branch_mean.sum(axis=1, keepdims=True)
branch_total[branch_total == 0] = 1
branch_prop = motif_branch_mean / branch_total

fig, ax = plt.subplots(figsize=(14, 6))
bottom = np.zeros(n_clusters)
for b in range(n_branches):
    ax.bar(x_pos, branch_prop[cluster_order, b], width=0.8, bottom=bottom,
           label=branch_labels[b], color=branch_colors[b], alpha=0.85)
    bottom += branch_prop[cluster_order, b]

ax.set_xticks(x_pos)
ax.set_xticklabels([f"M{k}" for k in cluster_order], fontsize=8)
ax.set_xlabel("Motif cluster")
ax.set_ylabel("Proportion of total activation")
ax.set_title("Normalised stem branch activation per motif")
ax.legend(title="Branch kernel", fontsize=9)
ax.set_ylim(0, 1.05)
fig.tight_layout()
savefig(fig, "stem_branch_per_motif_normalized.png")


# ======================================================================
# Analysis 6: Feature-Frequency Correlation
# ======================================================================
print("\n--- Analysis 6: Feature-Frequency Correlation ---")

BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (15, 30),
    "gamma": (30, 80),
}
band_names = list(BAND_RANGES.keys())
n_bands = len(band_names)

# Subsample 50K windows
n_ff = min(50000, len(all_windows))
ff_idx = rng.choice(len(all_windows), size=n_ff, replace=False)

print("  Computing band power per window...")
band_powers = np.zeros((n_ff, n_bands))
for i, wi in enumerate(ff_idx):
    signal = all_windows[wi, 0].numpy()
    f, pxx = welch(signal, fs=sampling_frequency, nperseg=nperseg)
    freq_res = f[1] - f[0]
    for b_i, (bname, (flo, fhi)) in enumerate(BAND_RANGES.items()):
        freq_mask = (f >= flo) & (f <= fhi)
        band_powers[i, b_i] = pxx[freq_mask].sum() * freq_res
    if (i + 1) % 10000 == 0:
        print(f"    {i + 1}/{n_ff}")

# Correlate features with band power
feats_sub = all_feats[ff_idx].numpy()
corr_matrix = np.zeros((feat_dim, n_bands))
for d in range(feat_dim):
    for b_i in range(n_bands):
        corr_matrix[d, b_i], _ = pearsonr(feats_sub[:, d], band_powers[:, b_i])

# Figure: feature_frequency_heatmap.png
max_abs_corr = np.abs(corr_matrix).max(axis=1)
sort_idx = np.argsort(max_abs_corr)[::-1]
corr_sorted = corr_matrix[sort_idx]

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(corr_sorted, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(n_bands))
ax.set_xticklabels(band_names, fontsize=9)
ax.set_ylabel("Feature dimension (sorted by max |r|)")
ax.set_title("Feature-frequency band correlation")
plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
fig.tight_layout()
savefig(fig, "feature_frequency_heatmap.png")

# Figure: feature_frequency_top_neurons.png
n_top = 5
top_neurons = sort_idx[:n_top]
band_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(n_top, n_bands, figsize=(4 * n_bands, 3 * n_top))
fig.suptitle("Top band-selective neurons", fontsize=13, y=1.01)

n_scatter = min(2000, n_ff)
scatter_idx = rng.choice(n_ff, size=n_scatter, replace=False)

for row_i, neuron_idx in enumerate(top_neurons):
    for col_i, bname in enumerate(band_names):
        ax = axes[row_i, col_i]
        r_val = corr_matrix[neuron_idx, col_i]
        ax.scatter(band_powers[scatter_idx, col_i], feats_sub[scatter_idx, neuron_idx],
                   s=2, alpha=0.3, color=band_colors[col_i])
        ax.set_title(f"Feat {neuron_idx} vs {bname} (r={r_val:.2f})", fontsize=8)
        if row_i == n_top - 1:
            ax.set_xlabel("Band power")
        if col_i == 0:
            ax.set_ylabel(f"Feature {neuron_idx}")

fig.tight_layout()
savefig(fig, "feature_frequency_top_neurons.png")


# ======================================================================
# Analysis 7: Motif Temporal Context
# ======================================================================
print("\n--- Analysis 7: Motif Temporal Context ---")

# Pick top 6 motifs by cluster size and 3 subjects
top_motifs = cluster_order[:6]
show_subjects = list(range(min(3, len(eval_subject_names))))

fig, axes = plt.subplots(len(top_motifs), len(show_subjects) + 1,
                          figsize=(5 * (len(show_subjects) + 1), 1.8 * len(top_motifs)))
fig.suptitle("Motif temporal context: occurrence timelines", fontsize=13, y=1.01)

burstiness = {}

for row, k in enumerate(top_motifs):
    mask_k = all_cluster_labels == k
    # Compute burstiness across all data
    k_offsets = all_time_offsets[mask_k] / sampling_frequency  # seconds
    k_subj = all_subject_ids[mask_k]

    # Per-subject timelines
    for col, si in enumerate(show_subjects):
        ax = axes[row, col]
        s_mask = k_subj == si
        if s_mask.sum() > 0:
            times = np.sort(k_offsets[s_mask])
            ax.eventplot([times], lineoffsets=0, linelengths=1, colors="steelblue",
                         linewidths=0.5)
        ax.set_xlim(0, k_offsets.max())
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f"Subject {si}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"M{k}", fontsize=9, rotation=0, labelpad=20)
        if row == len(top_motifs) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=6)

    # Burstiness: CV of inter-motif intervals (per subject, averaged)
    cvs = []
    for si in range(len(eval_subject_names)):
        s_mask = k_subj == si
        if s_mask.sum() > 5:
            times = np.sort(k_offsets[s_mask])
            intervals = np.diff(times)
            if len(intervals) > 1 and intervals.mean() > 0:
                cvs.append(intervals.std() / intervals.mean())
    mean_cv = np.mean(cvs) if cvs else 0.0
    burstiness[k] = mean_cv

    ax = axes[row, len(show_subjects)]
    ax.text(0.5, 0.5, f"CV={mean_cv:.2f}", ha="center", va="center",
            fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title("Burstiness" if row == 0 else "", fontsize=9)
    ax.axis("off")

fig.tight_layout()
savefig(fig, "motif_temporal_context.png")


# ======================================================================
print(f"\nAll figures saved to {OUT_DIR}")
print("\nBurstiness (CV of inter-motif intervals) per top motif:")
for k in top_motifs:
    print(f"  Motif {k}: CV = {burstiness[k]:.3f}")
