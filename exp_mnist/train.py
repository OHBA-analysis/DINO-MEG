"""Train DINO on MNIST for validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.multiprocessing.set_sharing_strategy("file_system")

import json
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from copy import deepcopy
from tqdm import tqdm

from modules.model import ConvNet2D, Projector, DINOModel
from modules.trainer import (
    train_one_epoch,
    param_groups_lrd,
    linear_warmup_cosine_decay,
    DINOLoss,
    knn_evaluate,
    linear_probe_evaluate,
)

# ----------
# Parameters
# ----------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
METRICS_FILE = os.path.join(CKPT_DIR, "metrics.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

epochs = 100
batch_size = 512
feat_dim = 256
hidden_dim = 1024
out_dim = 2048
use_predictor = False
lr = 5e-4
lr_start = 1e-6
lr_final_scale = 0.001
warmup_epochs = 10
weight_decay = 0.04
teacher_momentum = 0.996
teacher_momentum_final = 1.0
teacher_temp = 0.04
teacher_temp_final = 0.07
teacher_temp_warmup_epochs = 30
student_temp = 0.1
center_momentum = 0.9
grad_clip_norm = 1.0
knn_every = 5
knn_k = 20
save_every = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ----
# Data
# ----

_normalize = transforms.Normalize((0.1307,), (0.3081,))

_global_crop = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, shear=10, translate=(0.1, 0.1)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.5),
    transforms.ToTensor(),
    _normalize,
])
_local_crop = transforms.Compose([
    transforms.RandomResizedCrop(20, scale=(0.4, 0.7)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, shear=10, translate=(0.1, 0.1)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.5),
    transforms.ToTensor(),
    _normalize,
])

n_local_crops = 2


class MNISTMultiView(Dataset):
    """MNIST returning a list of augmented views (no labels).

    Views: 2 global (28x28) + n_local (20x20).
    """
    def __init__(self, train):
        self._base = datasets.MNIST(DATA_DIR, train=train, download=True)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, _ = self._base[idx]
        views = [_global_crop(img), _global_crop(img)]
        for _ in range(n_local_crops):
            views.append(_local_crop(img))
        return views


print("\nLoading data...")
dataset = MNISTMultiView(train=True)
dl = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=lambda b: b,
)

print(f"  Train samples: {len(dataset)}, steps/epoch: {len(dl)}")

# kNN eval uses plain MNIST with ToTensor + Normalize
_plain = transforms.Compose([transforms.ToTensor(), _normalize])
knn_train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_plain)
knn_val_ds   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_plain)
knn_train_dl = DataLoader(knn_train_ds, batch_size=512, shuffle=False, num_workers=12, persistent_workers=True, prefetch_factor=4)
knn_val_dl   = DataLoader(knn_val_ds,   batch_size=512, shuffle=False, num_workers=12, persistent_workers=True, prefetch_factor=4)

# -----
# Model
# -----

print("\nBuilding model...")
backbone = ConvNet2D(feat_dim=feat_dim, base_channels=64)
projector = Projector(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim)

student = DINOModel(backbone, projector, use_predictor=use_predictor).to(device)
teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in student.parameters())
print(f"  Parameters: {n_params:,}")

# ---------
# Optimizer
# ---------

param_groups = param_groups_lrd(student, weight_decay=weight_decay)
opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

n_iter_per_epoch = len(dl)

lr_schedule = linear_warmup_cosine_decay(
    base_value=lr,
    final_value=lr * lr_final_scale,
    epochs=epochs,
    n_iter_per_epoch=n_iter_per_epoch,
    warmup_epochs=warmup_epochs,
    start_warmup_value=lr_start,
)

teacher_m_schedule = linear_warmup_cosine_decay(
    base_value=teacher_momentum,
    final_value=teacher_momentum_final,
    epochs=epochs,
    n_iter_per_epoch=n_iter_per_epoch,
    warmup_epochs=0,
    start_warmup_value=teacher_momentum,
)

teacher_temp_schedule = linear_warmup_cosine_decay(
    base_value=teacher_temp_final,
    final_value=teacher_temp_final,
    epochs=epochs,
    n_iter_per_epoch=n_iter_per_epoch,
    warmup_epochs=teacher_temp_warmup_epochs,
    start_warmup_value=teacher_temp,
)

# ----
# Loss
# ----

loss_obj = DINOLoss(
    out_dim=out_dim,
    teacher_temp=teacher_temp,
    student_temp=student_temp,
    center_momentum=center_momentum,
    device=device,
)

# --------
# Training
# --------

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
global_teacher_view_idxs = [0, 1]  # first two views are global

metrics = []
step = 0

print(f"\nStarting training: {epochs} epochs, batch size {batch_size}, {n_iter_per_epoch} steps/epoch")
print(f"  Checkpoints: {CKPT_DIR}")
print(f"  Metrics:     {METRICS_FILE}\n")
pbar = tqdm(range(epochs), desc="Training", unit="epoch")
train_start = time.time()

for epoch in pbar:
    t0 = time.time()
    step, epoch_loss, grad_norm, center_norm = train_one_epoch(
        student,
        teacher,
        dl,
        opt,
        scaler,
        loss_obj,
        lr_schedule,
        teacher_m_schedule,
        global_teacher_view_idxs,
        device,
        step,
        grad_clip_norm,
        teacher_temp_schedule=teacher_temp_schedule,
    )
    epoch_secs = time.time() - t0
    elapsed = time.time() - train_start
    eta = elapsed / (epoch + 1) * (epochs - epoch - 1)

    row = {
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "grad_norm": grad_norm,
        "center_norm": center_norm,
    }

    if (epoch + 1) % knn_every == 0:
        top1, top5 = knn_evaluate(student.backbone, knn_train_dl, knn_val_dl, knn_k, device)
        row["knn_top1"] = top1
        row["knn_top5"] = top5
        lp_top1, lp_top5 = linear_probe_evaluate(student.backbone, knn_train_dl, knn_val_dl, device)
        row["lp_top1"] = lp_top1
        row["lp_top5"] = lp_top5
        msg = (
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| grad {grad_norm:.3f} | center {center_norm:.3f} "
            f"| kNN {top1*100:.1f}% | LP {lp_top1*100:.1f}% "
            f"| {epoch_secs:.1f}s/epoch | ETA {eta/60:.1f}min"
        )
    else:
        msg = (
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| grad {grad_norm:.3f} | center {center_norm:.3f} "
            f"| {epoch_secs:.1f}s/epoch | ETA {eta/60:.1f}min"
        )

    tqdm.write(msg)
    pbar.set_postfix(loss=f"{epoch_loss:.4f}", eta=f"{eta/60:.1f}min")

    metrics.append(row)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    if (epoch + 1) % save_every == 0:
        ckpt = {
            "epoch": epoch + 1,
            "step": step,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "loss": epoch_loss,
        }
        ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_epoch{epoch+1:04d}.pt")
        torch.save(ckpt, ckpt_path)
        tqdm.write(f"  Checkpoint saved: {ckpt_path}")

total_mins = (time.time() - train_start) / 60
print(f"\nTraining complete in {total_mins:.1f} min")
torch.save(student.backbone.state_dict(), os.path.join(CKPT_DIR, "backbone_final.pt"))
print(f"Backbone saved to {CKPT_DIR}/backbone_final.pt")
