"""Train DINO on HMM-MVN simulated MEG data."""

import json
import os
import time
import sys
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data import Transforms, MEGDataset, MEGLabeledDataset
from modules.model import ConvNet, Projector, DINOModel
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
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
os.makedirs(CKPT_DIR, exist_ok=True)

n_channels = 52
sampling_frequency = 250
window_length = 3000
stride = window_length // 2
crop_lengths = [1000, 500]
n_local_crops = 2
epochs = 100
batch_size = 32
feat_dim = 512
hidden_dim = 1024
out_dim = 8192
use_predictor = False
lr = 1e-3
lr_start = 1e-6
lr_final_scale = 0.001
warmup_epochs = 5
weight_decay = 0.01
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
save_every = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ----------------------
# Data and augmentations
# ----------------------

weak_transform = Transforms(
    sampling_frequency=sampling_frequency,
    add_noise_p=0.8,
    noise_std=5e-4,
    baseline_shift_p=0.2,
    baseline_shift_std=5e-4,
    scale_p=0.4,
    scale_sigma=0.05,
    amplitude_warp_p=0.2,
    warp_max=0.05,
    time_mask_p=0.3,
    time_mask_ratio=0.03,
    time_mask_n=1,
    channel_mask_p=0.2,
    channel_dropout_p=0.05,
    notch_p=0.9,
    notch_freqs=[50.0],
    notch_bandwidth=1.0,
    time_reverse_p=0.05,
)
strong_transform = Transforms(
    sampling_frequency=sampling_frequency,
    add_noise_p=0.95,
    noise_std=3e-3,
    baseline_shift_p=0.4,
    baseline_shift_std=1e-3,
    scale_p=0.6,
    scale_sigma=0.12,
    amplitude_warp_p=0.6,
    warp_max=0.15,
    time_mask_p=0.8,
    time_mask_ratio=0.08,
    time_mask_n=2,
    channel_mask_p=0.6,
    channel_dropout_p=0.15,
    notch_p=0.9,
    notch_freqs=[50.0],
    notch_bandwidth=1.0,
    time_reverse_p=0.1,
)

print("\nLoading data...")
files = [os.path.join(DATA_DIR, "X_train.npy")]
dataset = MEGDataset(
    files,
    window_length=window_length,
    stride=stride,
    crop_lengths=crop_lengths,
    n_local_crops=n_local_crops,
    weak_transform=weak_transform,
    strong_transform=strong_transform,
)
dl = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda b: b,
)

print(f"  Windows: {len(dataset)}, steps/epoch: {len(dl)}")

# Pre-load eval data for k-NN (avoids reloading from disk every eval epoch)
print("Loading eval data for k-NN...")
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))
X_train_arr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train_arr = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

knn_train_ds = MEGLabeledDataset(X_train_arr, Y_train_arr, window_length, stride)
knn_eval_ds = MEGLabeledDataset(X_eval, Y_eval, window_length, stride)
knn_train_dl = DataLoader(knn_train_ds, batch_size=128, shuffle=False, num_workers=4)
knn_eval_dl = DataLoader(knn_eval_ds, batch_size=128, shuffle=False, num_workers=4)

# -------------------------
# Student and teacher model
# -------------------------

print("\nBuilding model...")
backbone = ConvNet(in_channels=n_channels, feat_dim=feat_dim)
projector = Projector(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim)

student = DINOModel(backbone, projector, use_predictor=use_predictor).to(device)
teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in student.parameters())
print(f"  Parameters: {n_params:,}")

# -----------------------
# Optimizer and schedules
# -----------------------

param_groups = param_groups_lrd(student, weight_decay=weight_decay)
opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

n_iter_per_epoch = len(dl)
total_steps = epochs * n_iter_per_epoch

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
global_teacher_view_idxs = list(range(len(crop_lengths)))

metrics = []
step = 0

print(f"\nStarting training: {epochs} epochs, batch size {batch_size}, {n_iter_per_epoch} steps/epoch")
print(f"  Checkpoints: {CKPT_DIR}")
print(f"  Metrics:     {METRICS_FILE}\n")
train_start = time.time()

for epoch in range(epochs):
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

    row = {
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "grad_norm": grad_norm,
        "center_norm": center_norm,
    }

    if (epoch + 1) % knn_every == 0:
        top1, top5 = knn_evaluate(student.backbone, knn_train_dl, knn_eval_dl, knn_k, device)
        row["knn_top1"] = top1
        row["knn_top5"] = top5
        print(
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| grad {grad_norm:.3f} | center {center_norm:.3f} "
            f"| kNN top1 {top1*100:.1f}% top5 {top5*100:.1f}%"
        )
    else:
        print(
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| grad {grad_norm:.3f} | center {center_norm:.3f}"
        )

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
        print(f"  Checkpoint saved: {ckpt_path}")

total_mins = (time.time() - train_start) / 60
print(f"\nTraining complete in {total_mins:.1f} min")
torch.save(student.backbone.state_dict(), os.path.join(CKPT_DIR, "backbone_final.pt"))
print(f"Backbone saved to {CKPT_DIR}/backbone_final.pt")
