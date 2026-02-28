"""Save a grid of MNIST samples for visual verification."""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ds_train = datasets.MNIST(DATA_DIR, train=True, transform=transforms.ToTensor())
ds_test = datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor())

print(f"Train: {len(ds_train)} samples")
print(f"Test:  {len(ds_test)} samples")
print(f"Image shape: {ds_train[0][0].shape}  (C, H, W)")
print(f"Pixel range: [0, 1]")

for split, ds in [("train", ds_train), ("test", ds_test)]:
    imgs = torch.stack([ds[i][0] for i in range(64)])
    labels = [ds[i][1] for i in range(64)]

    grid = make_grid(imgs, nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"MNIST {split} — first 64 samples\n"
        + "  ".join(str(l) for l in labels[:8])
        + "\n..."
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"mnist_{split}_grid.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")
