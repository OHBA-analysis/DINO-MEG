"""Train DINO on single-channel temporal motif data.

The simulation generates 4 recurring temporal motifs (theta sinusoid, alpha
spindle, beta sawtooth, sharp wave) on a 1/f pink noise background, switched
by a Markov chain. See simulate_data.py for details.

Supports two backbone types:
  - "convnet":    ConvNet (raw waveform → preserves shape/polarity)
  - "filterbank": FilterbankNet (envelope extraction → captures frequency, loses shape)

Checkpoints and metrics are saved to checkpoints_{backbone_type}/.

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/train.py
"""

import json
import os
import time
import sys
import numpy as np
import torch
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data import Transforms, MEGDataset, MEGLabeledDataset
from modules.model import ConvNet, ConvNetV2, FilterbankNet, Projector, DINOModel
from modules.trainer import (
    train_one_epoch,
    param_groups_lrd,
    linear_warmup_cosine_decay,
    DINOLoss,
    knn_evaluate,
    linear_probe_evaluate,
    linear_probe_regression_evaluate,
)

# ----------
# Parameters
# ----------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Backbone type: "convnet", "convnet_v2", or "filterbank"
backbone_type = "convnet_v2"

CKPT_DIR = os.path.join(os.path.dirname(__file__), f"checkpoints_{backbone_type}")
METRICS_FILE = os.path.join(CKPT_DIR, "metrics.json")
os.makedirs(CKPT_DIR, exist_ok=True)

n_channels = 1
sampling_frequency = 250
window_length = 150
stride = 75
crop_lengths = [112, 112]
local_crop_length = 100
n_local_crops = 2
epochs = 200
batch_size = 128
feat_dim = 256
hidden_dim = 512
out_dim = 4096
use_predictor = False
lr = 5e-4
lr_start = 1e-6
lr_final_scale = 0.001
warmup_epochs = 3
weight_decay = 0.05
teacher_momentum = 0.996
teacher_momentum_final = 1.0
teacher_temp = 0.04
teacher_temp_final = 0.02
teacher_temp_warmup_epochs = 30
student_temp = 0.1
center_momentum = 0.9
grad_clip_norm = 1.0
knn_every = 5
knn_k = 10
save_every = 25

eval_window_length = 75
eval_stride = 37

# ConvNet params
convnet_kernel_sizes = [25, 9, 5]
convnet_strides = [2, 2, 2]

# ConvNetV2 params
v2_stem_channels = 32
v2_stem_kernel_sizes = (7, 15, 31)
v2_block_channels = (128, 256)
v2_block_kernel_sizes = (9, 5)
v2_attn_hidden = 64

# FilterbankNet params
fb_n_filters = 8
fb_filter_length = 65
fb_hidden_dim = 128
fb_n_queries = 4

n_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"Backbone type: {backbone_type}")

# ----------------------
# Data and augmentations
# ----------------------

# Single-channel: no channel masking, no time reversal (shape is a feature),
# no notch filtering (no power line artifacts in simulation)
weak_transform = Transforms(
    sampling_frequency=sampling_frequency,
    add_noise_p=0.8,
    noise_std=0.10,
    baseline_shift_p=0.3,
    baseline_shift_std=0.03,
    scale_p=0.5,
    scale_sigma=0.10,
    amplitude_warp_p=0.3,
    warp_max=0.04,
    time_mask_p=0.5,
    time_mask_ratio=0.06,
    time_mask_n=1,
    channel_mask_p=0.0,
    notch_p=0.0,
    time_reverse_p=0.0,
    sign_flip_p=0.3,
)
strong_transform = Transforms(
    sampling_frequency=sampling_frequency,
    add_noise_p=0.95,
    noise_std=0.15,
    baseline_shift_p=0.4,
    baseline_shift_std=0.05,
    scale_p=0.6,
    scale_sigma=0.15,
    amplitude_warp_p=0.6,
    warp_max=0.05,
    time_mask_p=0.8,
    time_mask_ratio=0.08,
    time_mask_n=2,
    channel_mask_p=0.0,
    notch_p=0.0,
    time_reverse_p=0.0,
    sign_flip_p=0.3,
)

print("\nLoading data...")
X_train_raw = np.load(os.path.join(DATA_DIR, "X_train.npy"))
dataset = MEGDataset(
    arrays=[X_train_raw],
    window_length=window_length,
    stride=stride,
    crop_lengths=crop_lengths,
    n_local_crops=n_local_crops,
    local_crop_length=local_crop_length,
    weak_transform=weak_transform,
    strong_transform=strong_transform,
)
del X_train_raw
dl = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=lambda b: b,
)

print(f"  Windows: {len(dataset)}, steps/epoch: {len(dl)}")

# Pre-load eval data for k-NN
print("Loading eval data for k-NN...")
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))
X_train_arr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train_arr = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

knn_train_ds = MEGLabeledDataset(X_train_arr, Y_train_arr, eval_window_length, eval_stride)
knn_eval_ds = MEGLabeledDataset(X_eval, Y_eval, eval_window_length, eval_stride)
knn_train_dl = DataLoader(knn_train_ds, batch_size=256, shuffle=False, num_workers=8)
knn_eval_dl = DataLoader(knn_eval_ds, batch_size=256, shuffle=False, num_workers=8)

# -------------------------
# Student and teacher model
# -------------------------

print("\nBuilding model...")
if backbone_type == "convnet":
    backbone = ConvNet(
        in_channels=n_channels, feat_dim=feat_dim,
        kernel_sizes=convnet_kernel_sizes, strides=convnet_strides,
    )
elif backbone_type == "convnet_v2":
    backbone = ConvNetV2(
        in_channels=n_channels, feat_dim=feat_dim,
        stem_channels=v2_stem_channels, stem_kernel_sizes=v2_stem_kernel_sizes,
        block_channels=v2_block_channels, block_kernel_sizes=v2_block_kernel_sizes,
        attn_hidden=v2_attn_hidden,
    )
elif backbone_type == "filterbank":
    backbone = FilterbankNet(
        in_channels=n_channels, feat_dim=feat_dim,
        n_filters=fb_n_filters, filter_length=fb_filter_length,
        hidden_dim=fb_hidden_dim, n_queries=fb_n_queries,
    )
else:
    raise ValueError(f"Unknown backbone_type: {backbone_type}")

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

pbar = tqdm(range(epochs), desc="Training", unit="ep")
for epoch in pbar:
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
        lp_top1, lp_top5 = linear_probe_evaluate(student.backbone, knn_train_dl, knn_eval_dl, device)
        row["lp_top1"] = lp_top1
        row["lp_top5"] = lp_top5
        lp_r2, lp_r2_per_state = linear_probe_regression_evaluate(
            student.backbone, knn_train_dl, knn_eval_dl,
            knn_train_ds, knn_eval_ds, n_classes=n_classes, device=device,
        )
        row["lp_r2"] = lp_r2
        row["lp_r2_per_state"] = lp_r2_per_state
        tqdm.write(
            f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
            f"| grad {grad_norm:.3f} | center {center_norm:.3f} "
            f"| kNN {top1*100:.1f}% | LP {lp_top1*100:.1f}% | R² {lp_r2:.3f}"
        )
    else:
        tqdm.write(
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
        tqdm.write(f"  Checkpoint saved: {ckpt_path}")

total_mins = (time.time() - train_start) / 60
print(f"\nTraining complete in {total_mins:.1f} min")
torch.save(student.backbone.state_dict(), os.path.join(CKPT_DIR, "backbone_final.pt"))
print(f"Backbone saved to {CKPT_DIR}/backbone_final.pt")
