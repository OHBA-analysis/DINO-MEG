"""Post-hoc analysis of MNIST DINO representations.

Produces 11 figures in exp_mnist/figures/:
  gradcam_per_class.png    – average GradCAM saliency per digit class
  gradcam_examples.png     – individual GradCAM examples
  pca_variance.png         – PCA explained variance
  pca_extremes.png         – digits at extremes of top PCs
  tsne_by_pc.png           – t-SNE coloured by top 3 PC values
  nn_retrieval.png         – nearest-neighbor retrieval examples
  subclusters.png          – intra-class sub-clusters
  interclass_similarity.png – cosine similarity between class centroids
  interclass_dendrogram.png – hierarchical clustering of classes
  probe_r2.png             – feature probing R² scores
  probe_scatter.png        – predicted vs actual for probed properties

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_mnist/analyse.py
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
from matplotlib.colors import Normalize as mplNormalize
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.model import ConvNet2D

# ------------------------------------------------------------------ config
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "backbone_final.pt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

feat_dim = 256
n_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

MEAN, STD = 0.1307, 0.3081

# ------------------------------------------------------------------ load model
backbone = ConvNet2D(feat_dim=feat_dim, base_channels=64).to(device)
backbone.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
backbone.eval()
print("Backbone loaded.")

# ------------------------------------------------------------------ load data & extract features
_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_norm)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

# Also load raw (unnormalised) images for display and property computation
raw_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

print("Extracting features...")
all_feats, all_labels, all_images = [], [], []
with torch.no_grad():
    for imgs, lbls in test_dl:
        f = backbone(imgs.to(device))
        f = F.normalize(f, dim=1).cpu()
        all_feats.append(f)
        all_labels.append(lbls)
        all_images.append(imgs)

feats = torch.cat(all_feats)       # (10000, 256)
labels = torch.cat(all_labels)     # (10000,)
images = torch.cat(all_images)     # (10000, 1, 28, 28)  normalised
print(f"  Features: {feats.shape}, Labels: {labels.shape}")

# Raw images as numpy for display
raw_images = np.array([raw_ds[i][0].squeeze().numpy() for i in range(len(raw_ds))])  # (10000, 28, 28)

# Class centroids in feature space
centroids = torch.stack([feats[labels == c].mean(dim=0) for c in range(n_classes)])
centroids = F.normalize(centroids, dim=1)  # (10, 256)


# ------------------------------------------------------------------ helpers
def denormalize(t):
    """Undo MNIST normalisation for display."""
    return t * STD + MEAN


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ======================================================================
# Analysis 1: GradCAM Saliency Maps
# ======================================================================
print("\n--- Analysis 1: GradCAM ---")

# Split backbone.net into spatial layers (conv/bn/relu) and head (pool/flatten/linear)
# ConvNet2D.net indices: 0-8 spatial, 9 AdaptiveAvgPool2d, 10 Flatten, 11 Linear
spatial_layers = backbone.net[:9]
head_layers = backbone.net[9:]


def compute_gradcam(img_batch, target_direction):
    """Compute GradCAM heatmaps for a batch of images.

    Uses split-forward to avoid inplace ReLU issues with hooks.
    target_direction: (feat_dim,) unit vector — the direction to maximise.
    Returns heatmaps of shape (B, 28, 28).
    """
    # Forward through spatial layers (no grad needed here)
    with torch.no_grad():
        spatial_out = spatial_layers(img_batch)  # (B, C, H, W)

    # Now require grad on spatial output for backward
    activations = spatial_out.detach().requires_grad_(True)

    # Forward through head
    pooled = head_layers[0](activations)  # AdaptiveAvgPool2d → (B, C, 1, 1)
    flat = head_layers[1](pooled)          # Flatten → (B, C)
    features = head_layers[2](flat)        # Linear → (B, feat_dim)

    # Score = dot product with target direction
    score = (features * target_direction.unsqueeze(0)).sum(dim=1)  # (B,)
    score.sum().backward()

    grads = activations.grad  # (B, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    cam = (weights * activations).sum(dim=1)  # (B, H, W)
    cam = torch.relu(cam)

    # Upsample to 28×28
    cam = F.interpolate(cam.unsqueeze(1), size=(28, 28), mode="bilinear",
                        align_corners=False).squeeze(1)

    # Normalise each map to [0, 1]
    B = cam.shape[0]
    cam = cam.view(B, -1)
    cam_min = cam.min(dim=1, keepdim=True).values
    cam_max = cam.max(dim=1, keepdim=True).values
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = cam.view(B, 28, 28)

    return cam.detach().cpu().numpy()


# Figure: gradcam_per_class.png — 10×2 grid (average digit + averaged GradCAM)
fig, axes = plt.subplots(n_classes, 2, figsize=(4, 18))
fig.suptitle("Average GradCAM per class", fontsize=13, y=1.01)

for c in range(n_classes):
    mask = (labels == c).numpy()
    class_imgs = images[mask].to(device)  # normalised
    direction = centroids[c].to(device)

    # Process in batches to save memory
    cams = []
    for start in range(0, len(class_imgs), 256):
        batch = class_imgs[start:start + 256]
        cams.append(compute_gradcam(batch, direction))
    cam_all = np.concatenate(cams, axis=0)
    avg_cam = cam_all.mean(axis=0)

    avg_img = raw_images[mask].mean(axis=0)

    axes[c, 0].imshow(avg_img, cmap="gray", vmin=0, vmax=1)
    axes[c, 0].set_ylabel(str(c), fontsize=12, rotation=0, labelpad=15)
    axes[c, 0].set_xticks([]); axes[c, 0].set_yticks([])

    axes[c, 1].imshow(avg_img, cmap="gray", vmin=0, vmax=1)
    axes[c, 1].imshow(avg_cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[c, 1].set_xticks([]); axes[c, 1].set_yticks([])

axes[0, 0].set_title("Average digit", fontsize=10)
axes[0, 1].set_title("GradCAM overlay", fontsize=10)
fig.tight_layout()
savefig(fig, "gradcam_per_class.png")

# Figure: gradcam_examples.png — 10×6 grid (3 examples per class: digit + overlay)
fig, axes = plt.subplots(n_classes, 6, figsize=(11, 18))
fig.suptitle("GradCAM examples (3 per class)", fontsize=13, y=1.01)

for c in range(n_classes):
    mask = np.where((labels == c).numpy())[0]
    # Pick 3 examples near centroid
    class_feats = feats[mask]
    sims_to_centroid = (class_feats @ centroids[c]).numpy()
    # Pick samples at different similarity levels: high, median, low
    sorted_idx = np.argsort(sims_to_centroid)
    picks = [sorted_idx[-1], sorted_idx[len(sorted_idx) // 2], sorted_idx[0]]

    for j, pi in enumerate(picks):
        global_idx = mask[pi]
        img = images[global_idx:global_idx + 1].to(device)
        direction = centroids[c].to(device)
        cam = compute_gradcam(img, direction)[0]

        col = j * 2
        axes[c, col].imshow(raw_images[global_idx], cmap="gray", vmin=0, vmax=1)
        axes[c, col].set_xticks([]); axes[c, col].set_yticks([])

        axes[c, col + 1].imshow(raw_images[global_idx], cmap="gray", vmin=0, vmax=1)
        axes[c, col + 1].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[c, col + 1].set_xticks([]); axes[c, col + 1].set_yticks([])

    axes[c, 0].set_ylabel(str(c), fontsize=12, rotation=0, labelpad=15)

for j in range(3):
    axes[0, j * 2].set_title("Digit", fontsize=9)
    axes[0, j * 2 + 1].set_title("GradCAM", fontsize=9)
fig.tight_layout()
savefig(fig, "gradcam_examples.png")


# ======================================================================
# Analysis 2: PCA of Features
# ======================================================================
print("\n--- Analysis 2: PCA ---")

n_pca = 50
pca = PCA(n_components=n_pca)
feats_pca = pca.fit_transform(feats.numpy())  # (10000, 50)

# Figure: pca_variance.png
fig, ax1 = plt.subplots(figsize=(10, 4))

var_ratio = pca.explained_variance_ratio_
cumvar = np.cumsum(var_ratio)

ax1.bar(range(n_pca), var_ratio * 100, color="steelblue", alpha=0.8, label="Individual")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance (%)")
ax1.set_title("PCA of backbone features (256-dim → 50 PCs)")

ax2 = ax1.twinx()
ax2.plot(range(n_pca), cumvar * 100, color="red", marker="o", markersize=3, label="Cumulative")
ax2.set_ylabel("Cumulative variance (%)")

# Threshold annotations
for thresh, ls in [(0.90, "--"), (0.95, ":")]:
    n_needed = np.searchsorted(cumvar, thresh) + 1
    ax2.axhline(thresh * 100, color="gray", linestyle=ls, alpha=0.5)
    ax2.annotate(f"{thresh*100:.0f}% at PC{n_needed}",
                 xy=(n_needed, thresh * 100), fontsize=9,
                 xytext=(n_needed + 3, thresh * 100 - 3),
                 arrowprops=dict(arrowstyle="->", color="gray"))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
fig.tight_layout()
savefig(fig, "pca_variance.png")

# Figure: pca_extremes.png — top 5 PCs, 5 lowest + 5 highest scoring digits each
n_show_pcs = 5
n_extremes = 5
fig, axes = plt.subplots(n_show_pcs, 2 * n_extremes + 1, figsize=(16, 8))
fig.suptitle("PCA extremes: 5 lowest ← PC → 5 highest", fontsize=13, y=1.01)

for pc in range(n_show_pcs):
    scores = feats_pca[:, pc]
    low_idx = np.argsort(scores)[:n_extremes]
    high_idx = np.argsort(scores)[-n_extremes:][::-1]

    for j, idx in enumerate(low_idx):
        axes[pc, j].imshow(raw_images[idx], cmap="gray", vmin=0, vmax=1)
        axes[pc, j].set_xticks([]); axes[pc, j].set_yticks([])
        if pc == 0:
            axes[pc, j].set_title(f"low {j+1}", fontsize=7)

    # Separator column
    mid = n_extremes
    axes[pc, mid].axis("off")
    axes[pc, mid].text(0.5, 0.5, f"PC{pc+1}\n({var_ratio[pc]*100:.1f}%)",
                       ha="center", va="center", fontsize=10, fontweight="bold",
                       transform=axes[pc, mid].transAxes)

    for j, idx in enumerate(high_idx):
        col = mid + 1 + j
        axes[pc, col].imshow(raw_images[idx], cmap="gray", vmin=0, vmax=1)
        axes[pc, col].set_xticks([]); axes[pc, col].set_yticks([])
        if pc == 0:
            axes[pc, col].set_title(f"high {j+1}", fontsize=7)

    axes[pc, 0].set_ylabel(f"PC{pc+1}", fontsize=10, rotation=0, labelpad=20)

fig.tight_layout()
savefig(fig, "pca_extremes.png")

# Figure: tsne_by_pc.png — t-SNE coloured by top 3 PC values
print("  Running t-SNE (5000 samples)...")
n_tsne = 5000
tsne_idx = np.random.RandomState(42).choice(len(feats), size=n_tsne, replace=False)
feats_sub = feats[tsne_idx].numpy()
labels_sub = labels[tsne_idx].numpy()
pca_sub = feats_pca[tsne_idx]

tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
            init="pca", random_state=0, max_iter=1000)
embed = tsne.fit_transform(feats_sub)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("t-SNE coloured by class and top PCs", fontsize=13)

# Panel 0: class labels
cmap10 = matplotlib.colormaps.get_cmap("tab10")
for c in range(n_classes):
    m = labels_sub == c
    axes[0].scatter(embed[m, 0], embed[m, 1], s=3, alpha=0.6, color=cmap10(c), label=str(c))
axes[0].legend(title="Digit", markerscale=4, fontsize=8, loc="best")
axes[0].set_title("Class labels")
axes[0].axis("off")

# Panels 1-3: PC1, PC2, PC3
for i in range(3):
    sc = axes[i + 1].scatter(embed[:, 0], embed[:, 1], s=3, alpha=0.6,
                              c=pca_sub[:, i], cmap="coolwarm")
    axes[i + 1].set_title(f"PC{i+1} value")
    axes[i + 1].axis("off")
    plt.colorbar(sc, ax=axes[i + 1], shrink=0.7)

fig.tight_layout()
savefig(fig, "tsne_by_pc.png")


# ======================================================================
# Analysis 3: Nearest-Neighbor Retrieval
# ======================================================================
print("\n--- Analysis 3: NN Retrieval ---")

# Full cosine similarity matrix
sim_matrix = feats @ feats.t()  # (10000, 10000)
sim_matrix.fill_diagonal_(0)

# Pick 1 query per class + 2 misclassified queries
# For misclassified: find via kNN
topk_sims, topk_idx = sim_matrix.topk(20, dim=1)
knn_labels = labels[topk_idx]
knn_preds = torch.stack([torch.bincount(knn_labels[i], minlength=10).argmax()
                         for i in range(len(feats))])
misclassified = (knn_preds != labels).numpy()

queries = []
for c in range(n_classes):
    # Pick sample closest to class centroid
    mask_c = np.where((labels == c).numpy())[0]
    sims_c = (feats[mask_c] @ centroids[c]).numpy()
    queries.append(mask_c[np.argmax(sims_c)])

# Add 2 misclassified
misc_idx = np.where(misclassified)[0]
if len(misc_idx) >= 2:
    np.random.RandomState(0).shuffle(misc_idx)
    queries.extend(misc_idx[:2].tolist())

n_neighbors = 10
fig, axes = plt.subplots(len(queries), n_neighbors + 1, figsize=(16, 1.5 * len(queries)))
fig.suptitle("Nearest-neighbor retrieval (green = same class, red = different)", fontsize=13, y=1.02)

for row, qi in enumerate(queries):
    q_label = labels[qi].item()
    pred_label = knn_preds[qi].item()

    # Query image
    axes[row, 0].imshow(raw_images[qi], cmap="gray", vmin=0, vmax=1)
    title = f"Q: {q_label}"
    if pred_label != q_label:
        title += f" (pred {pred_label})"
    axes[row, 0].set_title(title, fontsize=8, fontweight="bold")
    axes[row, 0].axis("off")
    # Blue border for query
    for spine in axes[row, 0].spines.values():
        spine.set_edgecolor("blue"); spine.set_linewidth(3); spine.set_visible(True)

    # Neighbors
    nn_idx = sim_matrix[qi].topk(n_neighbors).indices
    for col, ni in enumerate(nn_idx):
        ni = ni.item()
        n_label = labels[ni].item()
        axes[row, col + 1].imshow(raw_images[ni], cmap="gray", vmin=0, vmax=1)
        axes[row, col + 1].set_title(f"{n_label}", fontsize=7)
        axes[row, col + 1].axis("off")
        color = "green" if n_label == q_label else "red"
        for spine in axes[row, col + 1].spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2); spine.set_visible(True)

fig.tight_layout()
savefig(fig, "nn_retrieval.png")


# ======================================================================
# Analysis 4: Intra-Class Sub-Clusters
# ======================================================================
print("\n--- Analysis 4: Sub-clusters ---")

n_subclusters = 3
n_exemplars = 5
fig, axes = plt.subplots(n_classes, n_subclusters * n_exemplars,
                          figsize=(n_subclusters * n_exemplars * 1.1, n_classes * 1.2))
fig.suptitle(f"Intra-class sub-clusters (k={n_subclusters}, {n_exemplars} exemplars each)",
             fontsize=13, y=1.01)

for c in range(n_classes):
    mask_c = np.where((labels == c).numpy())[0]
    class_feats = feats[mask_c].numpy()

    km = KMeans(n_clusters=n_subclusters, random_state=0, n_init=10)
    cluster_ids = km.fit_predict(class_feats)

    for k in range(n_subclusters):
        cluster_mask = np.where(cluster_ids == k)[0]
        centroid_k = km.cluster_centers_[k]
        # Distance to sub-cluster centroid
        dists = np.linalg.norm(class_feats[cluster_mask] - centroid_k, axis=1)
        nearest = cluster_mask[np.argsort(dists)[:n_exemplars]]

        for j, ni in enumerate(nearest):
            col = k * n_exemplars + j
            global_idx = mask_c[ni]
            axes[c, col].imshow(raw_images[global_idx], cmap="gray", vmin=0, vmax=1)
            axes[c, col].set_xticks([]); axes[c, col].set_yticks([])
            if j == 0:
                # Light separator
                for spine in axes[c, col].spines.values():
                    spine.set_edgecolor("orange"); spine.set_linewidth(2); spine.set_visible(True)

    axes[c, 0].set_ylabel(str(c), fontsize=12, rotation=0, labelpad=15)

# Column group labels
for k in range(n_subclusters):
    mid_col = k * n_exemplars + n_exemplars // 2
    axes[0, mid_col].set_title(f"Sub-cluster {k+1}", fontsize=9, fontweight="bold")

fig.tight_layout()
savefig(fig, "subclusters.png")


# ======================================================================
# Analysis 5: Inter-Class Similarity
# ======================================================================
print("\n--- Analysis 5: Inter-class similarity ---")

# Cosine similarity between class centroids
centroid_sim = (centroids @ centroids.t()).numpy()

# Figure: interclass_similarity.png
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(centroid_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xlabel("Digit class"); ax.set_ylabel("Digit class")
ax.set_title("Inter-class cosine similarity (feature centroids)")

for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if centroid_sim[i, j] > 0.5 else "black"
        ax.text(j, i, f"{centroid_sim[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color=color)

plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
savefig(fig, "interclass_similarity.png")

# Figure: interclass_dendrogram.png
# Convert similarity to distance
cos_dist = 1.0 - centroid_sim
np.fill_diagonal(cos_dist, 0)
# Make symmetric (numerical precision)
cos_dist = (cos_dist + cos_dist.T) / 2
condensed = squareform(cos_dist)

Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(Z, labels=[str(c) for c in range(n_classes)], ax=ax,
           leaf_font_size=12, color_threshold=0.5)
ax.set_ylabel("Cosine distance")
ax.set_title("Hierarchical clustering of digit classes (average linkage)")
fig.tight_layout()
savefig(fig, "interclass_dendrogram.png")


# ======================================================================
# Analysis 6: Feature Probing
# ======================================================================
print("\n--- Analysis 6: Feature probing ---")


def compute_image_properties(imgs):
    """Compute visual properties from raw images (N, 28, 28) in [0, 1].

    Returns dict of arrays, each (N,):
      stroke_thickness: mean fraction of foreground pixels
      width_height_ratio: bounding box width / height
      vertical_com: vertical centre of mass (normalised to [0, 1])
      horizontal_com: horizontal centre of mass (normalised to [0, 1])
      slant_angle: estimated slant in degrees from vertical
    """
    N = imgs.shape[0]
    threshold = 0.3  # binarise threshold

    stroke_thickness = np.zeros(N)
    wh_ratio = np.zeros(N)
    v_com = np.zeros(N)
    h_com = np.zeros(N)
    slant = np.zeros(N)

    yy, xx = np.mgrid[0:28, 0:28]

    for i in range(N):
        img = imgs[i]
        binary = (img > threshold).astype(float)
        total = binary.sum()

        stroke_thickness[i] = total / (28 * 28)

        if total < 5:
            # Degenerate — skip
            continue

        # Bounding box
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        wh_ratio[i] = w / max(h, 1)

        # Centre of mass
        v_com[i] = (binary * yy).sum() / total / 27.0
        h_com[i] = (binary * xx).sum() / total / 27.0

        # Slant angle from second-order central moments
        cy = (binary * yy).sum() / total
        cx = (binary * xx).sum() / total
        mu20 = (binary * (xx - cx) ** 2).sum() / total
        mu02 = (binary * (yy - cy) ** 2).sum() / total
        mu11 = (binary * (xx - cx) * (yy - cy)).sum() / total
        slant[i] = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))

    return {
        "stroke_thickness": stroke_thickness,
        "width_height_ratio": wh_ratio,
        "vertical_com": v_com,
        "horizontal_com": h_com,
        "slant_angle": slant,
    }


props = compute_image_properties(raw_images)
prop_names = list(props.keys())
prop_labels = ["Stroke thickness", "Width/height ratio", "Vertical CoM",
               "Horizontal CoM", "Slant angle (°)"]

# Train/test split within the test set
n_probe = len(feats)
n_train_probe = 7000
rng = np.random.RandomState(42)
perm = rng.permutation(n_probe)
train_idx = perm[:n_train_probe]
test_idx = perm[n_train_probe:]

X_train = feats[train_idx].numpy()
X_test = feats[test_idx].numpy()

r2_scores = {}
predictions = {}
actuals = {}

for pname, plabel in zip(prop_names, prop_labels):
    y = props[pname]
    y_train = y[train_idx]
    y_test = y[test_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    r2_scores[pname] = r2
    predictions[pname] = y_pred
    actuals[pname] = y_test

    print(f"  {plabel}: R² = {r2:.3f}")

# Figure: probe_r2.png
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = range(len(prop_names))
bars = ax.bar(x_pos, [r2_scores[p] for p in prop_names], color="steelblue", edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(prop_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("R² score")
ax.set_title("Feature probing: Ridge regression R² (DINO features → visual properties)")
ax.set_ylim(0, 1.05)
for i, p in enumerate(prop_names):
    ax.text(i, r2_scores[p] + 0.02, f"{r2_scores[p]:.2f}", ha="center", fontsize=9)
fig.tight_layout()
savefig(fig, "probe_r2.png")

# Figure: probe_scatter.png — 2×3 grid of predicted vs actual
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Feature probing: predicted vs actual", fontsize=13)

for i, (pname, plabel) in enumerate(zip(prop_names, prop_labels)):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    ax.scatter(actuals[pname], predictions[pname], s=2, alpha=0.3, color="steelblue")

    # Identity line
    lo = min(actuals[pname].min(), predictions[pname].min())
    hi = max(actuals[pname].max(), predictions[pname].max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, linewidth=1)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{plabel}  (R²={r2_scores[pname]:.2f})", fontsize=10)

# Hide the 6th subplot (2×3 grid with 5 properties)
axes[1, 2].axis("off")
fig.tight_layout()
savefig(fig, "probe_scatter.png")


# ======================================================================
print("\nAll 11 figures saved to", OUT_DIR)
