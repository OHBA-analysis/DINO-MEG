"""Train DINO on MNIST for validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from copy import deepcopy

from modules.model import ConvNet2D, Projector, DINOModel
from modules.trainer import (
    train_one_epoch,
    param_groups_lrd,
    linear_warmup_cosine_decay,
    DINOLoss,
    knn_evaluate,
)

# ----------
# Parameters
# ----------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

epochs = 100
batch_size = 256
feat_dim = 256
hidden_dim = 1024
out_dim = 4096
use_predictor = False
lr = 5e-4
lr_start = 1e-6
lr_final_scale = 0.001
warmup_epochs = 10
weight_decay = 1e-6
teacher_momentum = 0.996
teacher_momentum_final = 1.0
teacher_temp = 0.04
student_temp = 0.1
center_momentum = 0.9
grad_clip_norm = 1.0
knn_every = 10
knn_k = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----
# Data
# ----

_normalize = transforms.Normalize((0.1307,), (0.3081,))

_global_crop = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _normalize,
])
_local_crop = transforms.Compose([
    transforms.RandomResizedCrop(14, scale=(0.2, 0.5)),
    transforms.ToTensor(),
    _normalize,
])


class MNISTMultiView(Dataset):
    """MNIST returning a list of 4 augmented views (no labels)."""
    def __init__(self, train):
        self._base = datasets.MNIST(DATA_DIR, train=train, download=True)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, _ = self._base[idx]
        return [
            _global_crop(img),
            _global_crop(img),
            _local_crop(img),
            _local_crop(img),
        ]


dataset = MNISTMultiView(train=True)
dl = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda b: b,
)

# kNN eval uses plain MNIST with ToTensor + Normalize
_plain = transforms.Compose([transforms.ToTensor(), _normalize])
knn_train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_plain)
knn_val_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_plain)
knn_train_dl = DataLoader(knn_train_ds, batch_size=512, shuffle=False, num_workers=4)
knn_val_dl = DataLoader(knn_val_ds, batch_size=512, shuffle=False, num_workers=4)

# -----
# Model
# -----

backbone = ConvNet2D(feat_dim=feat_dim)
projector = Projector(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim)

student = DINOModel(backbone, projector, use_predictor=use_predictor).to(device)
teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False

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

step = 0
for epoch in range(epochs):
    step, epoch_loss = train_one_epoch(
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
    )

    if (epoch + 1) % knn_every == 0:
        top1, top5 = knn_evaluate(student.backbone, knn_train_dl, knn_val_dl, knn_k, device)
        print(
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| kNN top1 {top1*100:.1f}% top5 {top5*100:.1f}%"
        )
    else:
        print(f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f}")
