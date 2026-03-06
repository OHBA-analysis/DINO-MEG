"""Post-hoc analysis of oscillatory burst DINO representations.

Produces 12 figures in exp_hmm_mvn/figures/:
  gradcam_spectrograms.png         – GradCAM overlaid on spectrograms per state
  gradcam_average.png              – average GradCAM temporal profile per state
  channel_attribution_brains.png   – learned vs ground-truth spatial maps on glass brain
  channel_attribution_correlation.png – scatter of learned vs ground-truth weights
  pca_variance.png                 – PCA eigenspectrum + cumulative
  pca_extremes.png                 – spectrograms at low/high extremes of top PCs
  tsne_by_pc.png                   – t-SNE coloured by class + PC1/PC2/PC3
  nn_retrieval.png                 – nearest-neighbor retrieval with spectrograms
  feature_frequency_heatmap.png    – feature × band-power correlation heatmap
  feature_frequency_top_neurons.png – top selective neurons scatter plots
  interstate_similarity.png        – cosine similarity of state centroids
  interstate_dendrogram.png        – hierarchical clustering dendrogram

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_hmm_mvn/analyse.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import butter, sosfiltfilt, welch, spectrogram as sp_spectrogram
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nibabel as nib
from nilearn import plotting as ni_plotting

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet, ConvNet2D, ViT1D, ViT2D, FilterbankNet
from modules.data import MEGLabeledDataset, compute_amplitude_envelopes

# Import ground-truth spatial weights from simulation
from simulate_data import SPATIAL_WEIGHTS, STATES, N_CHANNELS as SIM_N_CHANNELS

# ------------------------------------------------------------------ config
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "backbone_final.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
ATLAS_PATH = os.path.join(
    os.path.dirname(__file__),
    "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
)
os.makedirs(OUT_DIR, exist_ok=True)

n_channels = 52
feat_dim = 256
n_classes = 5
FS = 250
analysis_window_length = 375  # 1.5 s — matches global crop length
eval_window_length = 200
eval_stride = 100

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
BAND_RANGES = {
    "theta": (4, 8),
    "alpha": (7.5, 12.5),
    "beta": (15, 25),
    "low_gamma": (28, 42),
}

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


def _parcel_values_to_nifti(values, atlas_img):
    """Map a length-52 vector of parcel values into a 3D NIfTI stat image."""
    atlas_data = atlas_img.get_fdata()
    vol = np.zeros(atlas_data.shape[:3])
    for i, v in enumerate(values):
        mask = atlas_data[..., i] > 0
        vol[mask] = v
    return nib.Nifti1Image(vol, atlas_img.affine)


# ------------------------------------------------------------------ load model
print("Loading backbone...")
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

# Split backbone for GradCAM (ConvNet only)
if isinstance(backbone, ConvNet):
    spatial_layers = backbone.net[:9]
    head_layers = backbone.net[9:]

# ------------------------------------------------------------------ load data & extract features
print("Loading eval data...")
X_eval_raw = np.load(os.path.join(DATA_DIR, "X_eval.npy"))  # (C, T) — always needed for spectrograms
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))  # (T,)
X_train_raw = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

# Bandpass filter for backbone input (raw spectrograms stay unfiltered)
if bandpass is not None:
    print(f"  Bandpass filtering {bandpass[0]}-{bandpass[1]} Hz...")
    X_eval_filt = bandpass_filter(X_eval_raw, FS, *bandpass)
    X_train_filt = bandpass_filter(X_train_raw, FS, *bandpass)
else:
    X_eval_filt = X_eval_raw
    X_train_filt = X_train_raw

if use_tf:
    print(f"  Computing TF amplitude envelopes ({len(tf_freqs)} freqs)...")
    X_eval = compute_amplitude_envelopes(X_eval_filt, FS, tf_freqs,
                                          log_transform=True, standardize=True)
    X_train = compute_amplitude_envelopes(X_train_filt, FS, tf_freqs,
                                           log_transform=True, standardize=True)
else:
    X_eval = X_eval_filt
    X_train = X_train_filt

# Standard eval-sized windows for feature extraction (uses TF data if use_tf)
eval_ds = MEGLabeledDataset(X_eval, Y_eval, eval_window_length, eval_stride)
eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4)

# Raw eval windows for spectrograms/PSD (always raw data)
eval_ds_raw = MEGLabeledDataset(X_eval_raw, Y_eval, eval_window_length, eval_stride)
eval_dl_raw = torch.utils.data.DataLoader(eval_ds_raw, batch_size=128, shuffle=False, num_workers=4)

print("Extracting features...")
all_feats, all_labels, all_windows, all_windows_raw = [], [], [], []
with torch.no_grad():
    for (windows, lbls), (windows_raw, _) in zip(eval_dl, eval_dl_raw):
        f = backbone(windows.to(device))
        f = F.normalize(f, dim=1).cpu()
        all_feats.append(f)
        all_labels.append(torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls)
        all_windows.append(windows)
        all_windows_raw.append(windows_raw)

feats = torch.cat(all_feats)            # (N, feat_dim)
labels = torch.cat(all_labels)          # (N,)
windows_all = torch.cat(all_windows)    # (N, n_channels_input, eval_window_length)
windows_all_raw = torch.cat(all_windows_raw)  # (N, 52, eval_window_length) — always raw
print(f"  Features: {feats.shape}, Labels: {labels.shape}")

# Class centroids
centroids = torch.stack([feats[labels == c].mean(dim=0) for c in range(n_classes)])
centroids = F.normalize(centroids, dim=1)  # (5, feat_dim)

# Longer windows for spectrogram analyses — always raw for plotting
analysis_ds_raw = MEGLabeledDataset(X_eval_raw, Y_eval, analysis_window_length, eval_stride)
analysis_dl_raw = torch.utils.data.DataLoader(analysis_ds_raw, batch_size=128, shuffle=False, num_workers=4)

# Analysis-length windows for backbone forward pass (TF if use_tf)
analysis_ds = MEGLabeledDataset(X_eval, Y_eval, analysis_window_length, eval_stride)
analysis_dl = torch.utils.data.DataLoader(analysis_ds, batch_size=128, shuffle=False, num_workers=4)

print("Extracting analysis-length windows...")
analysis_windows_list, analysis_windows_raw_list, analysis_labels_list = [], [], []
with torch.no_grad():
    for (windows, lbls), (windows_raw, _) in zip(analysis_dl, analysis_dl_raw):
        analysis_windows_list.append(windows)
        analysis_windows_raw_list.append(windows_raw)
        analysis_labels_list.append(torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls)
analysis_windows = torch.cat(analysis_windows_list)       # (N2, n_channels_input, 750)
analysis_windows_raw = torch.cat(analysis_windows_raw_list)  # (N2, 52, 750) — always raw
analysis_labels = torch.cat(analysis_labels_list)


# ------------------------------------------------------------------ helpers
def plot_spectrogram(ax, window, fs=FS, channel=0, cmap="viridis"):
    """Plot spectrogram of a single window on given axes."""
    signal = window[channel] if window.ndim == 2 else window
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    freqs, times, Sxx = sp_spectrogram(signal, fs=fs, nperseg=128, noverlap=120)
    ax.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap=cmap)
    ax.set_ylim(0, 60)
    return freqs, times, Sxx


# ======================================================================
# Analysis 1: Temporal GradCAM (ConvNet only)
# ======================================================================
if isinstance(backbone, ConvNet):
    print("\n--- Analysis 1: Temporal GradCAM ---")

    def compute_gradcam_1d(window_batch, target_direction):
        """Compute 1D GradCAM for a batch of MEG windows.

        target_direction: (feat_dim,) unit vector.
        Returns cam of shape (B, input_length).
        """
        window_batch = window_batch.to(device)
        target_direction = target_direction.to(device)

        with torch.no_grad():
            spatial_out = spatial_layers(window_batch)  # (B, 256, T')

        activations = spatial_out.detach().requires_grad_(True)

        pooled = head_layers[0](activations)   # AdaptiveAvgPool1d → (B, 256, 1)
        flat = head_layers[1](pooled)          # Flatten → (B, 256)
        features = head_layers[2](flat)        # Linear → (B, feat_dim)

        score = (features * target_direction.unsqueeze(0)).sum(dim=1)
        score.sum().backward()

        grads = activations.grad                           # (B, 256, T')
        weights = grads.mean(dim=2, keepdim=True)          # (B, 256, 1)
        cam = (weights * activations).sum(dim=1)           # (B, T')
        cam = torch.relu(cam)

        # Interpolate to input length
        input_len = window_batch.shape[2]
        cam = F.interpolate(cam.unsqueeze(1), size=input_len, mode="linear",
                            align_corners=False).squeeze(1)

        # Normalise each to [0, 1]
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy()

    # Figure: gradcam_spectrograms.png — 5 rows (states) × 3 cols
    n_examples = 3
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(4 * n_examples, 3 * n_classes))
    fig.suptitle("GradCAM overlaid on spectrograms", fontsize=13, y=1.01)

    for c in range(n_classes):
        mask_c = (analysis_labels == c).numpy()
        idx_c = np.where(mask_c)[0]
        if len(idx_c) == 0:
            continue

        # Pick 3 examples evenly spaced
        picks = idx_c[np.linspace(0, len(idx_c) - 1, n_examples, dtype=int)]
        direction = centroids[c]

        for j, pi in enumerate(picks):
            win = analysis_windows[pi:pi + 1]  # (1, C, T)
            cam = compute_gradcam_1d(win, direction)[0]  # (T,)

            ax = axes[c, j]
            # Always use raw windows for spectrogram display
            signal = analysis_windows_raw[pi, 0].numpy()
            freqs, times, Sxx = sp_spectrogram(signal, fs=FS, nperseg=128, noverlap=120)
            ax.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12),
                          shading="gouraud", cmap="viridis")
            ax.set_ylim(0, 60)

            # Overlay GradCAM as semi-transparent red
            t_cam = np.linspace(0, times[-1], len(cam))
            ax.fill_between(t_cam, 0, 60, alpha=cam * 0.4, color="red")

            if j == 0:
                ax.set_ylabel(STATE_NAMES[c], fontsize=10)
            if c == n_classes - 1:
                ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_title(f"Example {j + 1}", fontsize=9)

    fig.tight_layout()
    savefig(fig, "gradcam_spectrograms.png")

    # Figure: gradcam_average.png — average GradCAM per state as line plots
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 2 * n_classes), sharex=True)
    fig.suptitle("Average GradCAM temporal profile per state", fontsize=13, y=1.01)

    for c in range(n_classes):
        mask_c = (labels == c).numpy()
        idx_c = np.where(mask_c)[0]
        if len(idx_c) == 0:
            continue

        direction = centroids[c]
        # Process in batches
        cams = []
        batch_size = 256
        # Use up to 500 windows for averaging
        sample_idx = idx_c[:500]
        for start in range(0, len(sample_idx), batch_size):
            batch_idx = sample_idx[start:start + batch_size]
            batch = windows_all[batch_idx]
            cams.append(compute_gradcam_1d(batch, direction))
        cam_all = np.concatenate(cams, axis=0)
        avg_cam = cam_all.mean(axis=0)

        t = np.arange(len(avg_cam)) / FS
        axes[c].plot(t, avg_cam, color=STATE_COLORS[c], linewidth=1.5)
        axes[c].fill_between(t, avg_cam, alpha=0.3, color=STATE_COLORS[c])
        axes[c].set_ylabel(STATE_NAMES[c], fontsize=10)
        axes[c].set_ylim(0, 1)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "gradcam_average.png")
else:
    print("\n--- Analysis 1: GradCAM skipped (only implemented for 1D ConvNet) ---")


# ======================================================================
# Analysis 2: Channel Attribution on Glass Brain
# ======================================================================
print("\n--- Analysis 2: Channel Attribution ---")


def compute_channel_attribution(window_batch, target_direction):
    """Compute channel-level attribution via input gradient × input.

    For raw input (B, C, T): returns (B, C) attribution.
    For TF input (B, C, F, T): sums over freq and time → (B, C).

    Returns (B, n_channels) attribution scores (always 52-d).
    """
    window_batch = window_batch.to(device).requires_grad_(True)
    target_direction = target_direction.to(device)

    features = backbone(window_batch)
    score = (features * target_direction.unsqueeze(0)).sum(dim=1)
    score.sum().backward()

    grad = window_batch.grad
    attr = (grad * window_batch).abs()

    if use_tf:
        # (B, C, F, T) → mean over T, sum over F → (B, C)
        attribution = attr.mean(dim=3).sum(dim=2)
    else:
        # (B, C, T) → mean over T → (B, C)
        attribution = attr.mean(dim=2)

    return attribution.detach().cpu().numpy()


# Compute per-state attribution
state_attributions = {}
for c in range(n_classes):
    mask_c = (labels == c).numpy()
    idx_c = np.where(mask_c)[0]
    if len(idx_c) == 0:
        state_attributions[c] = np.zeros(n_channels)
        continue

    direction = centroids[c]
    attrs = []
    sample_idx = idx_c[:500]
    batch_size = 128
    for start in range(0, len(sample_idx), batch_size):
        batch_idx = sample_idx[start:start + batch_size]
        batch = windows_all[batch_idx].clone()
        attrs.append(compute_channel_attribution(batch, direction))
    attr_all = np.concatenate(attrs, axis=0)
    state_attributions[c] = attr_all.mean(axis=0)  # (52,)

# Build ground-truth spatial weight vectors (52,) per oscillatory state
gt_weights = {}
for k, (name, center, bw) in enumerate(STATES):
    if center is None:
        continue
    w = np.zeros(n_channels)
    for hemi_idx, val in SPATIAL_WEIGHTS[name].items():
        w[hemi_idx] = val
        w[hemi_idx + 26] = val
    # Normalise to unit norm (same as simulation)
    norm = np.linalg.norm(w)
    if norm > 0:
        w /= norm
    gt_weights[k] = w

# Figure: channel_attribution_brains.png — oscillatory states only (4 rows × 2 cols)
osc_states = [k for k in range(n_classes) if STATES[k][1] is not None]
atlas_img = nib.load(ATLAS_PATH)

fig, axes = plt.subplots(len(osc_states), 2, figsize=(14, 3.5 * len(osc_states)))
fig.suptitle("Channel attribution: learned (left) vs ground truth (right)", fontsize=13, y=1.01)

for row, k in enumerate(osc_states):
    learned = state_attributions[k]
    gt = gt_weights[k]

    # Normalize both to [0, 1] for display
    learned_norm = learned / (learned.max() + 1e-8)
    gt_norm = gt / (gt.max() + 1e-8)

    for col, (vals, title) in enumerate([(learned_norm, "Learned"), (gt_norm, "Ground truth")]):
        stat_img = _parcel_values_to_nifti(vals, atlas_img)
        ni_plotting.plot_glass_brain(
            stat_img,
            display_mode="lyrz",
            colorbar=True,
            cmap="Reds",
            title=f"{STATE_NAMES[k]} — {title}",
            axes=axes[row, col],
            plot_abs=False,
            vmax=1.0,
        )

fig.tight_layout()
savefig(fig, "channel_attribution_brains.png")

# Figure: channel_attribution_correlation.png — scatter per state
fig, axes = plt.subplots(1, len(osc_states), figsize=(4 * len(osc_states), 4))
fig.suptitle("Channel attribution: learned vs ground truth", fontsize=13)

for j, k in enumerate(osc_states):
    learned = state_attributions[k]
    gt = gt_weights[k]

    # Normalise for comparison
    learned_norm = learned / (learned.max() + 1e-8)
    gt_norm = gt / (gt.max() + 1e-8)

    r, p = pearsonr(learned_norm, gt_norm)

    ax = axes[j]
    ax.scatter(gt_norm, learned_norm, s=15, alpha=0.7, color=STATE_COLORS[k])
    ax.set_xlabel("Ground truth weight")
    ax.set_ylabel("Learned attribution")
    ax.set_title(f"{STATE_NAMES[k]} (r={r:.2f}, p={p:.1e})", fontsize=10)

    # Fit line
    m, b = np.polyfit(gt_norm, learned_norm, 1)
    x_line = np.linspace(0, gt_norm.max(), 50)
    ax.plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)

fig.tight_layout()
savefig(fig, "channel_attribution_correlation.png")


# ======================================================================
# Analysis 3: PCA
# ======================================================================
print("\n--- Analysis 3: PCA ---")

n_pca = min(50, feat_dim)
pca = PCA(n_components=n_pca)
feats_pca = pca.fit_transform(feats.numpy())

# Figure: pca_variance.png
var_ratio = pca.explained_variance_ratio_
cumvar = np.cumsum(var_ratio)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(range(n_pca), var_ratio * 100, color="steelblue", alpha=0.8, label="Individual")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance (%)")
ax1.set_title(f"PCA of backbone features ({feat_dim}-dim → {n_pca} PCs)")

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
savefig(fig, "pca_variance.png")

# Figure: pca_extremes.png — top 5 PCs, spectrograms at low/high extremes
# Use analysis-length windows for better spectrograms
analysis_feats = []
with torch.no_grad():
    for start in range(0, len(analysis_windows), 128):
        batch = analysis_windows[start:start + 128].to(device)
        f = backbone(batch)
        f = F.normalize(f, dim=1).cpu()
        analysis_feats.append(f)
analysis_feats = torch.cat(analysis_feats)
analysis_pca = pca.transform(analysis_feats.numpy())

n_show_pcs = 5
n_extremes = 3
fig, axes = plt.subplots(n_show_pcs, 2 * n_extremes + 1,
                          figsize=(3 * (2 * n_extremes + 1), 2.5 * n_show_pcs))
fig.suptitle("PCA extremes: spectrograms at low ← PC → high", fontsize=13, y=1.01)

for pc in range(n_show_pcs):
    scores = analysis_pca[:, pc]
    low_idx = np.argsort(scores)[:n_extremes]
    high_idx = np.argsort(scores)[-n_extremes:][::-1]

    for j, idx in enumerate(low_idx):
        ax = axes[pc, j]
        plot_spectrogram(ax, analysis_windows_raw[idx], channel=0)
        if pc == 0:
            ax.set_title(f"low {j+1}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    mid = n_extremes
    axes[pc, mid].axis("off")
    axes[pc, mid].text(0.5, 0.5, f"PC{pc+1}\n({var_ratio[pc]*100:.1f}%)",
                       ha="center", va="center", fontsize=10, fontweight="bold",
                       transform=axes[pc, mid].transAxes)

    for j, idx in enumerate(high_idx):
        col = mid + 1 + j
        ax = axes[pc, col]
        plot_spectrogram(ax, analysis_windows_raw[idx], channel=0)
        if pc == 0:
            ax.set_title(f"high {j+1}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[pc, 0].set_ylabel(f"PC{pc+1}", fontsize=10, rotation=0, labelpad=20)

fig.tight_layout()
savefig(fig, "pca_extremes.png")

# Figure: tsne_by_pc.png — 4 panels: class + PC1/PC2/PC3
print("  Running t-SNE for PCA overlay (5000 samples)...")
n_tsne = min(5000, len(feats))
tsne_idx = np.random.RandomState(42).choice(len(feats), size=n_tsne, replace=False)
feats_sub = feats[tsne_idx].numpy()
labels_sub = labels[tsne_idx].numpy()
pca_sub = feats_pca[tsne_idx]

tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000)
embed = tsne.fit_transform(feats_sub)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("t-SNE coloured by state and top PCs", fontsize=13)

for c in range(n_classes):
    m = labels_sub == c
    axes[0].scatter(embed[m, 0], embed[m, 1], s=3, alpha=0.6,
                    color=STATE_COLORS[c], label=STATE_NAMES[c])
axes[0].legend(title="State", markerscale=4, fontsize=8, loc="best")
axes[0].set_title("State labels")
axes[0].axis("off")

for i in range(3):
    sc = axes[i + 1].scatter(embed[:, 0], embed[:, 1], s=3, alpha=0.6,
                              c=pca_sub[:, i], cmap="coolwarm")
    axes[i + 1].set_title(f"PC{i+1} value")
    axes[i + 1].axis("off")
    plt.colorbar(sc, ax=axes[i + 1], shrink=0.7)

fig.tight_layout()
savefig(fig, "tsne_by_pc.png")


# ======================================================================
# Analysis 4: NN Retrieval
# ======================================================================
print("\n--- Analysis 4: NN Retrieval ---")

# Use analysis-length windows for spectrograms
sim_matrix = analysis_feats @ analysis_feats.t()
sim_matrix.fill_diagonal_(0)

n_queries = max(n_classes, 5)
n_neighbors = 8
queries = []
for c in range(n_classes):
    mask_c = (analysis_labels == c).numpy()
    idx_c = np.where(mask_c)[0]
    if len(idx_c) == 0:
        continue
    # Pick sample closest to centroid
    class_feats = analysis_feats[idx_c]
    sims_c = (class_feats @ centroids[c]).numpy()
    queries.append(idx_c[np.argmax(sims_c)])

fig, axes = plt.subplots(len(queries), n_neighbors + 1,
                          figsize=(2.5 * (n_neighbors + 1), 3 * len(queries)))
fig.suptitle("NN retrieval: query spectrogram + nearest neighbors (green=same, red=different)",
             fontsize=12, y=1.01)

for row, qi in enumerate(queries):
    q_label = analysis_labels[qi].item()

    # Query spectrogram (always use raw windows)
    ax = axes[row, 0]
    plot_spectrogram(ax, analysis_windows_raw[qi], channel=0)
    ax.set_title(f"Query: {STATE_NAMES[q_label]}", fontsize=8, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Neighbors
    nn_idx = sim_matrix[qi].topk(n_neighbors).indices
    for col, ni in enumerate(nn_idx):
        ni = ni.item()
        n_label = analysis_labels[ni].item()
        ax = axes[row, col + 1]
        plot_spectrogram(ax, analysis_windows_raw[ni], channel=0)
        ax.set_title(STATE_NAMES[n_label], fontsize=7)
        color = "green" if n_label == q_label else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

fig.tight_layout()
savefig(fig, "nn_retrieval.png")


# ======================================================================
# Analysis 5: Feature–Frequency Correlation
# ======================================================================
print("\n--- Analysis 5: Feature–Frequency Correlation ---")

# Compute band power via Welch for each eval window
band_names = list(BAND_RANGES.keys())
n_bands = len(band_names)

print("  Computing band power per window...")
band_powers = np.zeros((len(windows_all_raw), n_bands))
for i in range(len(windows_all_raw)):
    win_np = windows_all_raw[i].numpy()  # (52, T) — always raw
    freqs_w, psd_w = welch(win_np, fs=FS, nperseg=min(256, eval_window_length), axis=1)
    psd_mean = psd_w.mean(axis=0)  # average across channels
    freq_res = freqs_w[1] - freqs_w[0]
    for b, (bname, (flo, fhi)) in enumerate(BAND_RANGES.items()):
        freq_mask = (freqs_w >= flo) & (freqs_w <= fhi)
        band_powers[i, b] = psd_mean[freq_mask].sum() * freq_res

# Correlate each feature dimension with each band
feats_np = feats.numpy()
corr_matrix = np.zeros((feat_dim, n_bands))
for d in range(feat_dim):
    for b in range(n_bands):
        corr_matrix[d, b], _ = pearsonr(feats_np[:, d], band_powers[:, b])

# Figure: feature_frequency_heatmap.png
# Sort features by max absolute correlation
max_abs_corr = np.abs(corr_matrix).max(axis=1)
sort_idx = np.argsort(max_abs_corr)[::-1]
corr_sorted = corr_matrix[sort_idx]

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(corr_sorted, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(n_bands))
ax.set_xticklabels(band_names, fontsize=9)
ax.set_ylabel("Feature dimension (sorted by max |r|)")
ax.set_title("Feature–frequency band correlation")
plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
fig.tight_layout()
savefig(fig, "feature_frequency_heatmap.png")

# Figure: feature_frequency_top_neurons.png — top 5 most selective neurons
n_top = 5
top_neurons = sort_idx[:n_top]
fig, axes = plt.subplots(n_top, n_bands, figsize=(4 * n_bands, 3 * n_top))
fig.suptitle("Top band-selective neurons: activation vs band power", fontsize=13, y=1.01)

# Subsample for scatter
n_scatter = min(2000, len(feats_np))
scatter_idx = np.random.RandomState(0).choice(len(feats_np), size=n_scatter, replace=False)

for row, neuron_idx in enumerate(top_neurons):
    for col, bname in enumerate(band_names):
        ax = axes[row, col]
        r_val = corr_matrix[neuron_idx, col]
        ax.scatter(band_powers[scatter_idx, col], feats_np[scatter_idx, neuron_idx],
                   s=2, alpha=0.3, color=STATE_COLORS[col])
        ax.set_title(f"Feat {neuron_idx} vs {bname} (r={r_val:.2f})", fontsize=8)
        if row == n_top - 1:
            ax.set_xlabel("Band power")
        if col == 0:
            ax.set_ylabel(f"Feature {neuron_idx}")

fig.tight_layout()
savefig(fig, "feature_frequency_top_neurons.png")


# ======================================================================
# Analysis 6: Inter-State Similarity
# ======================================================================
print("\n--- Analysis 6: Inter-state similarity ---")

centroid_sim = (centroids @ centroids.t()).numpy()

# Figure: interstate_similarity.png
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(centroid_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("State")
ax.set_ylabel("State")
ax.set_title("Inter-state cosine similarity (feature centroids)")

for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if centroid_sim[i, j] > 0.5 else "black"
        ax.text(j, i, f"{centroid_sim[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color=color)

plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
savefig(fig, "interstate_similarity.png")

# Figure: interstate_dendrogram.png
cos_dist = 1.0 - centroid_sim
np.fill_diagonal(cos_dist, 0)
cos_dist = (cos_dist + cos_dist.T) / 2
condensed = squareform(cos_dist)

Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(Z, labels=STATE_NAMES, ax=ax, leaf_font_size=11, color_threshold=0.5)
ax.set_ylabel("Cosine distance")
ax.set_title("Hierarchical clustering of oscillatory states (average linkage)")
fig.tight_layout()
savefig(fig, "interstate_dendrogram.png")


# ======================================================================
print(f"\nAll 12 figures saved to {OUT_DIR}")
