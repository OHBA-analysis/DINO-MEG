"""Generate summary figures for ConvNetV2 DINO performance.

Produces figures in exp_motif_1ch/figures/convnet_v2/:
  summary_performance.png  – 4-panel: training curves, per-motif accuracy,
                             normalised confusion matrix, t-SNE
  summary_interpretability.png – 3-panel: average GradCAM, attention weights,
                                 stem branch activation

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/summary.py
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
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNetV2
from modules.data import MEGLabeledDataset

# ------------------------------------------------------------------ config
backbone_type = "convnet_v2"
CKPT = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}", "backbone_final.pt")
METRICS_FILE = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}", "metrics.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", backbone_type)
os.makedirs(OUT_DIR, exist_ok=True)

n_channels = 1
feat_dim = 256
knn_k = 10
eval_window_length = 75
eval_stride = 37
n_classes = 5
FS = 250

STATE_NAMES = ["theta_sin", "alpha_spindle", "beta_sawtooth", "sharp_wave", "background"]
STATE_LABELS = ["Theta\nsin", "Alpha\nspindle", "Beta\nsaw", "Sharp\nwave", "Back-\nground"]
STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#cccccc"]

# Supervised ceiling from the sanity check run (ConvNetV2, same data)
SUPERVISED_ACC = {
    "theta_sin": 0.791,
    "alpha_spindle": 0.813,
    "beta_sawtooth": 0.879,
    "sharp_wave": 0.801,
    "background": 0.954,
}
SUPERVISED_OVERALL = 0.899

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ load
print("Loading metrics...")
with open(METRICS_FILE) as f:
    metrics = json.load(f)

print("Loading backbone...")
backbone = ConvNetV2(
    in_channels=n_channels, feat_dim=feat_dim,
    stem_channels=32, stem_kernel_sizes=(7, 15, 31),
    block_channels=(128, 256), block_kernel_sizes=(9, 5),
    attn_hidden=64,
).to(device)
backbone.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
backbone.eval()

print("Loading data & extracting features...")
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


train_feats, train_labels = extract(train_dl)
eval_feats, eval_labels = extract(eval_dl)

# k-NN predictions
sims = eval_feats @ train_feats.t()
topk = sims.topk(knn_k, dim=1).indices
neighbor_labels = train_labels[topk]
preds = torch.tensor([
    torch.bincount(neighbor_labels[i], minlength=n_classes).argmax().item()
    for i in range(len(eval_feats))
])
overall_top1 = (preds == eval_labels).float().mean().item()

# Confusion matrix
conf = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(eval_labels.numpy(), preds.numpy()):
    conf[true, pred] += 1
per_class_acc = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)

# Normalised confusion matrix (row-normalised)
conf_norm = conf.astype(float) / np.maximum(conf.sum(axis=1, keepdims=True), 1)


# ======================================================================
# Figure 1: Performance Summary (2x2)
# ======================================================================
print("\n--- Performance summary ---")
fig = plt.figure(figsize=(14, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel A: Training curves ---
ax_loss = fig.add_subplot(gs[0, 0])
epochs_list = [m["epoch"] for m in metrics]
losses = [m["loss"] for m in metrics]
knn_epochs = [m["epoch"] for m in metrics if "knn_top1" in m]
knn_vals = [m["knn_top1"] * 100 for m in metrics if "knn_top1" in m]
lp_vals = [m["lp_top1"] * 100 for m in metrics if "lp_top1" in m]

ln1, = ax_loss.plot(epochs_list, losses, color="steelblue", linewidth=1.2, label="Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("DINO loss", color="steelblue")
ax_loss.tick_params(axis="y", labelcolor="steelblue")

ax_acc = ax_loss.twinx()
ln2, = ax_acc.plot(knn_epochs, knn_vals, "o-", color="green", markersize=3,
                   linewidth=1.2, label="k-NN")
ln3, = ax_acc.plot(knn_epochs, lp_vals, "s-", color="purple", markersize=3,
                   linewidth=1.2, label="Linear probe")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_ylim(60, 100)

lns = [ln1, ln2, ln3]
ax_loss.legend(lns, [l.get_label() for l in lns], loc="center right", fontsize=8)
ax_loss.set_title("A. Training convergence", fontsize=11, fontweight="bold", loc="left")

# --- Panel B: Per-motif accuracy (DINO vs supervised) ---
ax_bar = fig.add_subplot(gs[0, 1])
x = np.arange(n_classes)
width = 0.35

supervised_vals = [SUPERVISED_ACC[s] * 100 for s in STATE_NAMES]
dino_vals = per_class_acc * 100

bars1 = ax_bar.bar(x - width / 2, supervised_vals, width, label="Supervised ceiling",
                   color=[c + "80" for c in STATE_COLORS], edgecolor=STATE_COLORS, linewidth=1.5)
bars2 = ax_bar.bar(x + width / 2, dino_vals, width, label="DINO k-NN",
                   color=STATE_COLORS, edgecolor="black", linewidth=0.5)

ax_bar.axhline(SUPERVISED_OVERALL * 100, ls="--", color="gray", alpha=0.5, linewidth=0.8)
ax_bar.axhline(overall_top1 * 100, ls="-", color="red", alpha=0.5, linewidth=0.8)

for i, (sv, dv) in enumerate(zip(supervised_vals, dino_vals)):
    ax_bar.text(i - width / 2, sv + 1, f"{sv:.0f}", ha="center", va="bottom", fontsize=7,
                color="gray")
    ax_bar.text(i + width / 2, dv + 1, f"{dv:.0f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(STATE_LABELS, fontsize=8)
ax_bar.set_ylabel("Accuracy (%)")
ax_bar.set_ylim(0, 105)
ax_bar.legend(fontsize=8, loc="lower right")
ax_bar.set_title("B. Per-motif accuracy", fontsize=11, fontweight="bold", loc="left")

# --- Panel C: Normalised confusion matrix ---
ax_cm = fig.add_subplot(gs[1, 0])
im = ax_cm.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
ax_cm.set_xticks(range(n_classes))
ax_cm.set_yticks(range(n_classes))
ax_cm.set_xticklabels(STATE_LABELS, fontsize=8)
ax_cm.set_yticklabels(STATE_LABELS, fontsize=8)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")

for i in range(n_classes):
    for j in range(n_classes):
        val = conf_norm[i, j]
        color = "white" if val > 0.5 else "black"
        ax_cm.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax_cm, shrink=0.8, label="Proportion")
ax_cm.set_title(f"C. Confusion matrix (k-NN, overall {overall_top1*100:.1f}%)",
                fontsize=11, fontweight="bold", loc="left")

# --- Panel D: t-SNE ---
ax_tsne = fig.add_subplot(gs[1, 1])
n_tsne = min(3000, len(eval_feats))
rng = np.random.RandomState(42)
idx = rng.choice(len(eval_feats), size=n_tsne, replace=False)
feats_sub = eval_feats[idx].numpy()
labels_sub = eval_labels[idx].numpy()

print("  Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000)
embed = tsne.fit_transform(feats_sub)

for c in range(n_classes):
    m = labels_sub == c
    ax_tsne.scatter(embed[m, 0], embed[m, 1], s=5, alpha=0.6,
                    color=STATE_COLORS[c], label=STATE_NAMES[c])
ax_tsne.legend(title="Motif", markerscale=3, fontsize=8, loc="best")
ax_tsne.axis("off")
ax_tsne.set_title("D. t-SNE of backbone features", fontsize=11, fontweight="bold", loc="left")

fig.suptitle("ConvNetV2 DINO — Performance Summary", fontsize=14, fontweight="bold", y=0.98)

path = os.path.join(OUT_DIR, "summary_performance.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path}")


# ======================================================================
# Figure 2: Interpretability Summary (1x3)
# ======================================================================
print("\n--- Interpretability summary ---")

# Get attention weights + stem branch activations
eval_windows_all = []
with torch.no_grad():
    for windows, _ in eval_dl:
        eval_windows_all.append(windows)
eval_windows_all = torch.cat(eval_windows_all)

n_show = min(500, len(eval_windows_all))
show_idx = rng.choice(len(eval_windows_all), size=n_show, replace=False)
show_windows = eval_windows_all[show_idx].to(device)
show_labels = eval_labels[show_idx].numpy()

with torch.no_grad():
    _, attn_w = backbone(show_windows, return_attention=True)
attn_np = attn_w.cpu().numpy()
T_prime = attn_np.shape[1]

# Stem branch activations
with torch.no_grad():
    branch_means = []
    for branch in backbone.stem_branches:
        act = branch(show_windows)
        branch_means.append(act.abs().mean(dim=(1, 2)).cpu().numpy())
branch_means = np.stack(branch_means, axis=1)

# GradCAM average per motif
centroids = torch.stack([eval_feats[eval_labels == c].mean(dim=0) for c in range(n_classes)])
centroids = F.normalize(centroids, dim=1)

print("  Computing average GradCAM per motif...")
gradcam_avgs = {}
for c in range(n_classes):
    mask_c = (eval_labels == c).numpy()
    idx_c = np.where(mask_c)[0]
    if len(idx_c) == 0:
        continue
    direction = centroids[c].to(device)
    sample_idx = idx_c[:300]
    cams = []
    for start in range(0, len(sample_idx), 128):
        batch_idx = sample_idx[start:start + 128]
        batch = eval_windows_all[batch_idx].to(device)

        with torch.no_grad():
            branches = [br(batch) for br in backbone.stem_branches]
            x = torch.cat(branches, dim=1)
            for block in backbone.blocks:
                x = block(x)
        activations = x.detach().requires_grad_(True)
        x_t = activations.transpose(1, 2)
        attn_logits = backbone.attn_net(x_t).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled = (x_t * attn_weights.unsqueeze(-1)).sum(dim=1)
        features = backbone.head(pooled)
        score = (features * direction.unsqueeze(0)).sum(dim=1)
        score.sum().backward()

        grads = activations.grad
        weights = grads.mean(dim=2, keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        input_len = batch.shape[2]
        cam = F.interpolate(cam.unsqueeze(1), size=input_len, mode="linear",
                            align_corners=False).squeeze(1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cams.append(cam.detach().cpu().numpy())

    gradcam_avgs[c] = np.concatenate(cams, axis=0).mean(axis=0)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel A: Average GradCAM per motif ---
ax = axes[0]
t = np.arange(eval_window_length) / FS * 1000  # ms
for c in range(n_classes):
    if c in gradcam_avgs:
        ax.plot(t, gradcam_avgs[c], color=STATE_COLORS[c], linewidth=1.5,
                label=STATE_NAMES[c])
ax.set_xlabel("Time (ms)")
ax.set_ylabel("GradCAM (normalised)")
ax.set_title("A. Average GradCAM per motif", fontsize=11, fontweight="bold", loc="left")
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)

# --- Panel B: Temporal attention weights ---
ax = axes[1]
t_prime_ax = np.arange(T_prime)
for c in range(n_classes):
    mask_c = show_labels == c
    if mask_c.sum() == 0:
        continue
    avg_attn = attn_np[mask_c].mean(axis=0)
    ax.plot(t_prime_ax, avg_attn, color=STATE_COLORS[c], linewidth=1.5,
            label=STATE_NAMES[c])
ax.set_xlabel("Temporal position (downsampled)")
ax.set_ylabel("Attention weight")
ax.set_title("B. Temporal attention per motif", fontsize=11, fontweight="bold", loc="left")
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)

# --- Panel C: Stem branch activation ---
ax = axes[2]
kernel_labels = ["k=7\n(28ms)", "k=15\n(60ms)", "k=31\n(124ms)"]
n_branches = len(kernel_labels)
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
           label=kernel_labels[b], color=branch_colors[b], alpha=0.85, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(STATE_LABELS, fontsize=8)
ax.set_ylabel("Mean |activation|")
ax.set_title("C. Multi-scale stem activation", fontsize=11, fontweight="bold", loc="left")
ax.legend(title="Branch", fontsize=8, loc="best")
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("ConvNetV2 DINO — Interpretability Summary", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()

path = os.path.join(OUT_DIR, "summary_interpretability.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path}")

print("\nDone.")
