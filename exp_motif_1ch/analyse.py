"""Post-hoc analysis of single-channel motif DINO representations.

Produces figures in exp_motif_1ch/figures/{backbone_type}/:
  gradcam_waveforms.png              – GradCAM overlaid on raw waveforms per motif (ConvNet/V2)
  gradcam_average.png                – average GradCAM temporal profile per motif (ConvNet/V2)
  attention_weights.png              – temporal attention weights per motif (ConvNetV2/FilterbankNet)
  attention_individual.png           – attention overlaid on individual windows (ConvNetV2)
  stem_branch_activation.png         – which stem branch activates per motif (ConvNetV2)
  filter_visualization.png           – learned FIR filters + frequency responses (FilterbankNet)
  pca_variance.png                   – PCA eigenspectrum + cumulative
  pca_extremes.png                   – waveforms at low/high extremes of top PCs
  tsne_by_pc.png                     – t-SNE coloured by class + PC1/PC2/PC3
  nn_retrieval.png                   – nearest-neighbor retrieval with raw waveforms
  feature_frequency_heatmap.png      – feature x band-power correlation heatmap
  feature_frequency_top_neurons.png  – top selective neurons scatter plots
  interstate_similarity.png          – cosine similarity of motif centroids
  interstate_dendrogram.png          – hierarchical clustering dendrogram

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/analyse.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, freqz
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet, ConvNetV2, FilterbankNet
from modules.data import MEGLabeledDataset

# ------------------------------------------------------------------ config
backbone_type = "convnet_v2"  # "convnet", "convnet_v2", or "filterbank"

CKPT = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}", "backbone_final.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", backbone_type)
os.makedirs(OUT_DIR, exist_ok=True)

n_channels = 1
feat_dim = 256
n_classes = 5
FS = 250
analysis_window_length = 112  # matches global crop length
eval_window_length = 75
eval_stride = 37

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
BAND_RANGES = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (15, 25),
    "broadband": (1, 50),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Backbone type: {backbone_type}")


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ load model
print("Loading backbone...")
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

# Split backbone for GradCAM (ConvNet only)
if isinstance(backbone, ConvNet):
    spatial_layers = backbone.net[:9]
    head_layers = backbone.net[9:]

# ------------------------------------------------------------------ load data & extract features
print("Loading eval data...")
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))  # (1, T)
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))  # (T,)
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

# Standard eval-sized windows for feature extraction
eval_ds = MEGLabeledDataset(X_eval, Y_eval, eval_window_length, eval_stride)
eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=4)

print("Extracting features...")
all_feats, all_labels, all_windows = [], [], []
with torch.no_grad():
    for windows, lbls in eval_dl:
        f = backbone(windows.to(device))
        f = F.normalize(f, dim=1).cpu()
        all_feats.append(f)
        all_labels.append(torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls)
        all_windows.append(windows)

feats = torch.cat(all_feats)          # (N, feat_dim)
labels = torch.cat(all_labels)        # (N,)
windows_all = torch.cat(all_windows)  # (N, 1, eval_window_length)
print(f"  Features: {feats.shape}, Labels: {labels.shape}")

# Class centroids
centroids = torch.stack([feats[labels == c].mean(dim=0) for c in range(n_classes)])
centroids = F.normalize(centroids, dim=1)  # (5, feat_dim)

# Analysis-length windows for detailed plots
analysis_ds = MEGLabeledDataset(X_eval, Y_eval, analysis_window_length, eval_stride)
analysis_dl = torch.utils.data.DataLoader(analysis_ds, batch_size=256, shuffle=False, num_workers=4)

print("Extracting analysis-length features...")
analysis_feats_list, analysis_windows_list, analysis_labels_list = [], [], []
with torch.no_grad():
    for windows, lbls in analysis_dl:
        f = backbone(windows.to(device))
        f = F.normalize(f, dim=1).cpu()
        analysis_feats_list.append(f)
        analysis_windows_list.append(windows)
        analysis_labels_list.append(torch.tensor(lbls) if not isinstance(lbls, torch.Tensor) else lbls)
analysis_feats = torch.cat(analysis_feats_list)
analysis_windows = torch.cat(analysis_windows_list)
analysis_labels = torch.cat(analysis_labels_list)


# ------------------------------------------------------------------ helpers
def plot_waveform(ax, window, fs=FS, color="black", alpha=1.0):
    """Plot raw waveform of a single-channel window."""
    signal = window[0] if window.ndim == 2 else window
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, color=color, linewidth=0.8, alpha=alpha)
    ax.set_xlim(0, t[-1])


# ======================================================================
# Analysis 1: Temporal GradCAM (ConvNet only) — overlay on raw waveforms
# ======================================================================
if isinstance(backbone, (ConvNet, ConvNetV2)):
    print("\n--- Analysis 1: Temporal GradCAM ---")

    if isinstance(backbone, ConvNet):
        def compute_gradcam_1d(window_batch, target_direction):
            """Compute 1D GradCAM for ConvNet."""
            window_batch = window_batch.to(device)
            target_direction = target_direction.to(device)

            with torch.no_grad():
                spatial_out = spatial_layers(window_batch)

            activations = spatial_out.detach().requires_grad_(True)

            pooled = head_layers[0](activations)
            flat = head_layers[1](pooled)
            features = head_layers[2](flat)

            score = (features * target_direction.unsqueeze(0)).sum(dim=1)
            score.sum().backward()

            grads = activations.grad
            weights = grads.mean(dim=2, keepdim=True)
            cam = (weights * activations).sum(dim=1)
            cam = torch.relu(cam)

            input_len = window_batch.shape[2]
            cam = F.interpolate(cam.unsqueeze(1), size=input_len, mode="linear",
                                align_corners=False).squeeze(1)

            cam_min = cam.min(dim=1, keepdim=True).values
            cam_max = cam.max(dim=1, keepdim=True).values
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam.detach().cpu().numpy()

    else:  # ConvNetV2
        def compute_gradcam_1d(window_batch, target_direction):
            """Compute 1D GradCAM for ConvNetV2 on last residual block output."""
            window_batch = window_batch.to(device)
            target_direction = target_direction.to(device)

            with torch.no_grad():
                branches = [branch(window_batch) for branch in backbone.stem_branches]
                x = torch.cat(branches, dim=1)
                for block in backbone.blocks:
                    x = block(x)

            activations = x.detach().requires_grad_(True)

            x_t = activations.transpose(1, 2)
            attn_logits = backbone.attn_net(x_t).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=-1)
            pooled = (x_t * attn_weights.unsqueeze(-1)).sum(dim=1)
            features = backbone.head(pooled)

            score = (features * target_direction.unsqueeze(0)).sum(dim=1)
            score.sum().backward()

            grads = activations.grad
            weights = grads.mean(dim=2, keepdim=True)
            cam = (weights * activations).sum(dim=1)
            cam = torch.relu(cam)

            input_len = window_batch.shape[2]
            cam = F.interpolate(cam.unsqueeze(1), size=input_len, mode="linear",
                                align_corners=False).squeeze(1)

            cam_min = cam.min(dim=1, keepdim=True).values
            cam_max = cam.max(dim=1, keepdim=True).values
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam.detach().cpu().numpy()

    # Figure: gradcam_waveforms.png — 5 rows (motifs) x 3 cols, GradCAM on raw waveforms
    n_examples = 3
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(4 * n_examples, 2.5 * n_classes))
    fig.suptitle("GradCAM overlaid on raw waveforms", fontsize=13, y=1.01)

    for c in range(n_classes):
        mask_c = (analysis_labels == c).numpy()
        idx_c = np.where(mask_c)[0]
        if len(idx_c) == 0:
            continue

        picks = idx_c[np.linspace(0, len(idx_c) - 1, n_examples, dtype=int)]
        direction = centroids[c]

        for j, pi in enumerate(picks):
            win = analysis_windows[pi:pi + 1]
            cam = compute_gradcam_1d(win, direction)[0]

            ax = axes[c, j]
            signal = analysis_windows[pi, 0].numpy()
            t = np.arange(len(signal)) / FS

            ax.plot(t, signal, color="black", linewidth=0.8)
            ax.fill_between(t, signal.min(), signal.max(),
                            alpha=cam * 0.4, color="red")

            if j == 0:
                ax.set_ylabel(STATE_NAMES[c], fontsize=10)
            if c == n_classes - 1:
                ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_title(f"Example {j + 1}", fontsize=9)

    fig.tight_layout()
    savefig(fig, "gradcam_waveforms.png")

    # Figure: gradcam_average.png — average GradCAM per motif
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 2 * n_classes), sharex=True)
    fig.suptitle("Average GradCAM temporal profile per motif", fontsize=13, y=1.01)

    for c in range(n_classes):
        mask_c = (labels == c).numpy()
        idx_c = np.where(mask_c)[0]
        if len(idx_c) == 0:
            continue

        direction = centroids[c]
        cams = []
        batch_size = 256
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
    print("\n--- Analysis 1: GradCAM skipped (ConvNet/ConvNetV2 only) ---")


# ======================================================================
# Analysis 2: Filter Visualization (FilterbankNet only)
# ======================================================================
if isinstance(backbone, FilterbankNet):
    print("\n--- Analysis 2: Filter Visualization ---")

    # Extract learned FIR filters
    fir_weights = backbone.filterbank.weight.detach().cpu().numpy()  # (n_filters, 1, filter_length)
    n_filters = fir_weights.shape[0]
    filter_length = fir_weights.shape[2]

    # Figure: filter_visualization.png — 2 rows: time domain + frequency response
    fig, axes = plt.subplots(2, n_filters, figsize=(3 * n_filters, 6))
    fig.suptitle("Learned FIR filterbank", fontsize=13, y=1.02)

    for i in range(n_filters):
        h = fir_weights[i, 0]
        t = np.arange(len(h)) / FS * 1000  # ms

        # Time domain
        axes[0, i].plot(t, h, color="steelblue", linewidth=1)
        axes[0, i].set_title(f"Filter {i}", fontsize=9)
        if i == 0:
            axes[0, i].set_ylabel("Amplitude")
        axes[0, i].set_xlabel("Time (ms)")

        # Frequency response
        w, H = freqz(h, worN=512, fs=FS)
        axes[1, i].plot(w, 20 * np.log10(np.abs(H) + 1e-8), color="darkorange", linewidth=1)
        axes[1, i].set_xlim(0, FS / 2)
        axes[1, i].set_ylim(-40, None)
        if i == 0:
            axes[1, i].set_ylabel("Magnitude (dB)")
        axes[1, i].set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    savefig(fig, "filter_visualization.png")

    # Figure: attention_weights.png — per-query attention over time
    print("  Computing attention patterns...")
    n_show = min(200, len(windows_all))
    show_idx = np.random.RandomState(0).choice(len(windows_all), size=n_show, replace=False)
    show_windows = windows_all[show_idx].to(device)
    show_labels = labels[show_idx].numpy()

    # Forward pass up to attention computation
    with torch.no_grad():
        B, C, T = show_windows.shape
        x = show_windows.reshape(B * C, 1, T)
        x = backbone.filterbank(x)
        x = x.reshape(B, C * backbone.n_filters, T)
        x = x.abs()
        x = backbone.envelope_smooth(x)
        x = torch.log1p(x)
        x = backbone.fb_norm(x)
        x = backbone.channel_mix(x)
        x = backbone.temporal(x)  # (B, hidden_dim, T')
        T_prime = x.shape[2]
        x = x.transpose(1, 2)  # (B, T', hidden_dim)
        keys = backbone.attn_key(x)
        attn = torch.bmm(
            backbone.attn_queries.expand(B, -1, -1),
            keys.transpose(1, 2),
        ) / (keys.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)  # (B, n_queries, T')

    attn_np = attn.cpu().numpy()
    n_queries = attn_np.shape[1]

    fig, axes = plt.subplots(n_queries, 1, figsize=(10, 2.5 * n_queries))
    fig.suptitle("Attention weights per query (averaged by motif class)", fontsize=13, y=1.01)

    for q in range(n_queries):
        ax = axes[q] if n_queries > 1 else axes
        t_prime = np.arange(T_prime)
        for c in range(n_classes):
            mask_c = show_labels == c
            if mask_c.sum() == 0:
                continue
            avg_attn = attn_np[mask_c, q].mean(axis=0)
            ax.plot(t_prime, avg_attn, label=STATE_NAMES[c], color=STATE_COLORS[c], linewidth=1.5)
        ax.set_ylabel(f"Query {q}", fontsize=10)
        if q == 0:
            ax.legend(fontsize=8, loc="upper right")

    (axes[-1] if n_queries > 1 else axes).set_xlabel("Temporal position (downsampled)")
    fig.tight_layout()
    savefig(fig, "attention_weights.png")
else:
    print("\n--- Analysis 2: Filter Visualization skipped (FilterbankNet only) ---")


# ======================================================================
# Analysis 2b: Attention weights and stem branch analysis (ConvNetV2 only)
# ======================================================================
if isinstance(backbone, ConvNetV2):
    print("\n--- Analysis 2b: Attention & Stem Branch Analysis (ConvNetV2) ---")

    # --- Attention weight visualization ---
    n_show = min(500, len(windows_all))
    show_idx = np.random.RandomState(0).choice(len(windows_all), size=n_show, replace=False)
    show_windows = windows_all[show_idx].to(device)
    show_labels = labels[show_idx].numpy()

    with torch.no_grad():
        _, attn_w = backbone(show_windows, return_attention=True)
    attn_np = attn_w.cpu().numpy()  # (n_show, T')
    T_prime = attn_np.shape[1]

    # Average attention per motif
    fig, ax = plt.subplots(figsize=(10, 4))
    t_prime = np.arange(T_prime)
    for c in range(n_classes):
        mask_c = show_labels == c
        if mask_c.sum() == 0:
            continue
        avg_attn = attn_np[mask_c].mean(axis=0)
        ax.plot(t_prime, avg_attn, label=STATE_NAMES[c], color=STATE_COLORS[c], linewidth=1.5)
    ax.set_xlabel("Temporal position (downsampled)")
    ax.set_ylabel("Attention weight")
    ax.set_title("Temporal attention weights averaged by motif class (ConvNetV2)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "attention_weights.png")

    # Individual attention patterns: 3 examples per motif
    n_examples = 3
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(4 * n_examples, 2 * n_classes))
    fig.suptitle("Attention weights on individual windows (ConvNetV2)", fontsize=13, y=1.01)

    for c in range(n_classes):
        mask_c = show_labels == c
        idx_c = np.where(mask_c)[0]
        if len(idx_c) < n_examples:
            continue
        picks = idx_c[np.linspace(0, len(idx_c) - 1, n_examples, dtype=int)]
        for j, pi in enumerate(picks):
            ax = axes[c, j]
            signal = show_windows[pi, 0].cpu().numpy()
            t_sig = np.arange(len(signal)) / FS
            # Plot waveform
            ax.plot(t_sig, signal, color="black", linewidth=0.8)
            # Overlay attention as shading (upsample to input length)
            attn_up = np.interp(np.linspace(0, 1, len(signal)),
                                np.linspace(0, 1, T_prime), attn_np[pi])
            ax.fill_between(t_sig, signal.min(), signal.max(),
                            alpha=attn_up / attn_up.max() * 0.5, color="blue")
            if j == 0:
                ax.set_ylabel(STATE_NAMES[c], fontsize=9)
            if c == 0:
                ax.set_title(f"Example {j + 1}", fontsize=9)

    fig.tight_layout()
    savefig(fig, "attention_individual.png")

    # --- Stem branch activation analysis ---
    print("  Computing stem branch activations...")
    with torch.no_grad():
        branch_means = []
        for branch in backbone.stem_branches:
            act = branch(show_windows)  # (B, stem_channels, T)
            branch_means.append(act.abs().mean(dim=(1, 2)).cpu().numpy())  # (B,)
    branch_means = np.stack(branch_means, axis=1)  # (n_show, n_branches)

    kernel_labels = [f"k={k}" for k in [7, 15, 31]]
    n_branches = len(kernel_labels)

    # Grouped bar chart: mean activation per branch per motif
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_classes)
    width = 0.8 / n_branches
    branch_colors = ["#4c72b0", "#dd8452", "#55a868"]

    for b in range(n_branches):
        means_per_class = []
        sems_per_class = []
        for c in range(n_classes):
            mask_c = show_labels == c
            vals = branch_means[mask_c, b]
            means_per_class.append(vals.mean())
            sems_per_class.append(vals.std() / np.sqrt(len(vals)))
        offset = (b - n_branches / 2 + 0.5) * width
        ax.bar(x + offset, means_per_class, width, yerr=sems_per_class,
               label=kernel_labels[b], color=branch_colors[b], alpha=0.85,
               capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, fontsize=9)
    ax.set_ylabel("Mean |activation|")
    ax.set_title("Stem branch activation by motif class (ConvNetV2)")
    ax.legend(title="Branch kernel", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, "stem_branch_activation.png")
else:
    print("\n--- Analysis 2b: ConvNetV2 analysis skipped ---")


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
ax1.set_title(f"PCA of backbone features ({backbone_type}, {feat_dim}-dim)")

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

# Figure: pca_extremes.png — top 5 PCs, waveforms at low/high extremes
analysis_pca = pca.transform(analysis_feats.numpy())

n_show_pcs = 5
n_extremes = 3
fig, axes = plt.subplots(n_show_pcs, 2 * n_extremes + 1,
                          figsize=(3 * (2 * n_extremes + 1), 2 * n_show_pcs))
fig.suptitle(f"PCA extremes: waveforms at low <-- PC --> high ({backbone_type})", fontsize=13, y=1.01)

for pc in range(n_show_pcs):
    scores = analysis_pca[:, pc]
    low_idx = np.argsort(scores)[:n_extremes]
    high_idx = np.argsort(scores)[-n_extremes:][::-1]

    for j, idx in enumerate(low_idx):
        ax = axes[pc, j]
        plot_waveform(ax, analysis_windows[idx])
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
        plot_waveform(ax, analysis_windows[idx])
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
fig.suptitle(f"t-SNE coloured by motif and top PCs ({backbone_type})", fontsize=13)

for c in range(n_classes):
    m = labels_sub == c
    axes[0].scatter(embed[m, 0], embed[m, 1], s=3, alpha=0.6,
                    color=STATE_COLORS[c], label=STATE_NAMES[c])
axes[0].legend(title="Motif", markerscale=4, fontsize=8, loc="best")
axes[0].set_title("Motif labels")
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
# Analysis 4: NN Retrieval — show raw waveforms
# ======================================================================
print("\n--- Analysis 4: NN Retrieval ---")

sim_matrix = analysis_feats @ analysis_feats.t()
sim_matrix.fill_diagonal_(0)

n_neighbors = 8
queries = []
for c in range(n_classes):
    mask_c = (analysis_labels == c).numpy()
    idx_c = np.where(mask_c)[0]
    if len(idx_c) == 0:
        continue
    class_feats = analysis_feats[idx_c]
    sims_c = (class_feats @ centroids[c]).numpy()
    queries.append(idx_c[np.argmax(sims_c)])

fig, axes = plt.subplots(len(queries), n_neighbors + 1,
                          figsize=(2.5 * (n_neighbors + 1), 2.5 * len(queries)))
fig.suptitle(f"NN retrieval: query waveform + nearest neighbors ({backbone_type})",
             fontsize=12, y=1.01)

for row, qi in enumerate(queries):
    q_label = analysis_labels[qi].item()

    ax = axes[row, 0]
    plot_waveform(ax, analysis_windows[qi])
    ax.set_title(f"Query: {STATE_NAMES[q_label]}", fontsize=8, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(3)
        spine.set_visible(True)

    nn_idx = sim_matrix[qi].topk(n_neighbors).indices
    for col, ni in enumerate(nn_idx):
        ni = ni.item()
        n_label = analysis_labels[ni].item()
        ax = axes[row, col + 1]
        plot_waveform(ax, analysis_windows[ni])
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

band_names = list(BAND_RANGES.keys())
n_bands = len(band_names)

print("  Computing band power per window...")
band_powers = np.zeros((len(windows_all), n_bands))
for i in range(len(windows_all)):
    win_np = windows_all[i].numpy()  # (1, T)
    freqs_w, psd_w = welch(win_np[0], fs=FS, nperseg=min(64, eval_window_length))
    freq_res = freqs_w[1] - freqs_w[0]
    for b, (bname, (flo, fhi)) in enumerate(BAND_RANGES.items()):
        freq_mask = (freqs_w >= flo) & (freqs_w <= fhi)
        band_powers[i, b] = psd_w[freq_mask].sum() * freq_res

# Correlate each feature dimension with each band
feats_np = feats.numpy()
corr_matrix = np.zeros((feat_dim, n_bands))
for d in range(feat_dim):
    for b in range(n_bands):
        corr_matrix[d, b], _ = pearsonr(feats_np[:, d], band_powers[:, b])

# Figure: feature_frequency_heatmap.png
max_abs_corr = np.abs(corr_matrix).max(axis=1)
sort_idx = np.argsort(max_abs_corr)[::-1]
corr_sorted = corr_matrix[sort_idx]

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(corr_sorted, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax.set_xticks(range(n_bands))
ax.set_xticklabels(band_names, fontsize=9)
ax.set_ylabel("Feature dimension (sorted by max |r|)")
ax.set_title(f"Feature-frequency band correlation ({backbone_type})")
plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
fig.tight_layout()
savefig(fig, "feature_frequency_heatmap.png")

# Figure: feature_frequency_top_neurons.png
n_top = 5
top_neurons = sort_idx[:n_top]
fig, axes = plt.subplots(n_top, n_bands, figsize=(4 * n_bands, 3 * n_top))
fig.suptitle(f"Top band-selective neurons ({backbone_type})", fontsize=13, y=1.01)

n_scatter = min(2000, len(feats_np))
scatter_idx = np.random.RandomState(0).choice(len(feats_np), size=n_scatter, replace=False)

for row_i, neuron_idx in enumerate(top_neurons):
    for col_i, bname in enumerate(band_names):
        ax = axes[row_i, col_i]
        r_val = corr_matrix[neuron_idx, col_i]
        ax.scatter(band_powers[scatter_idx, col_i], feats_np[scatter_idx, neuron_idx],
                   s=2, alpha=0.3, color=STATE_COLORS[min(col_i, len(STATE_COLORS) - 1)])
        ax.set_title(f"Feat {neuron_idx} vs {bname} (r={r_val:.2f})", fontsize=8)
        if row_i == n_top - 1:
            ax.set_xlabel("Band power")
        if col_i == 0:
            ax.set_ylabel(f"Feature {neuron_idx}")

fig.tight_layout()
savefig(fig, "feature_frequency_top_neurons.png")


# ======================================================================
# Analysis 6: Inter-Motif Similarity — key result
# ======================================================================
print("\n--- Analysis 6: Inter-motif similarity ---")

centroid_sim = (centroids @ centroids.t()).numpy()

# Figure: interstate_similarity.png
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(centroid_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("Motif")
ax.set_ylabel("Motif")
ax.set_title(f"Inter-motif cosine similarity ({backbone_type})")

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
ax.set_title(f"Hierarchical clustering of motifs ({backbone_type})")
fig.tight_layout()
savefig(fig, "interstate_dendrogram.png")


# ======================================================================
print(f"\nAll figures saved to {OUT_DIR}")
