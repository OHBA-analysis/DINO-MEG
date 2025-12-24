"""Train DINO."""

import torch
from torch.utils.data import DataLoader
from copy import deepcopy

from modules.data import Transforms, MEGDataset
from modules.model import ConvNet, Projector, DINOModel
from modules.trainer import (
    train_one_epoch,
    param_groups_lrd,
    linear_warmup_cosine_decay,
    DINOLoss,
)

# ----------
# Parameters
# ----------

n_channels = 52
sampling_frequency = 250
window_length = 3000
stride = window_length // 2
crop_lengths = [1000, 500]
n_local_crops = 2
epochs = 10
batch_size = 4
feat_dim = 512
hidden_dim = 1024
out_dim = 8192
predictor_hidden = 512
predictor_out = None
use_predictor = False
lr = 1e-3
lr_start = 1e-6
lr_final_scale = 0.001
warmup_epochs = 2
weight_decay = 1e-6
teacher_momentum = 0.996
teacher_momentum_final = 1.0
teacher_temp = 0.04
student_temp = 0.1
center_momentum = 0.9
grad_clip_norm = 1.0
use_amp = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

files = ["data/X.npy"]
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

# -------------------------
# Student and teacher model
# -------------------------

backbone = ConvNet(in_channels=n_channels, feat_dim=feat_dim)
projector = Projector(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim)

student = DINOModel(
    backbone,
    projector,
    use_predictor=use_predictor,
    predictor_kwargs={
        "hidden_dim": predictor_hidden,
        "out_dim": out_dim if predictor_out is None else predictor_out,
    },
).to(device)
teacher = deepcopy(student).to(device)

# Don't need gradients for the teacher
for p in teacher.parameters():
    p.requires_grad = False

# -----------------------
# Optimizer and schedules
# -----------------------

param_groups = param_groups_lrd(student, weight_decay=weight_decay)
opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

n_iter_per_epoch = len(dl)
total_steps = epochs * n_iter_per_epoch

print("n_iter_per_epoch =", n_iter_per_epoch)
print("total_steps =", total_steps)
exit()

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

loss = DINOLoss(
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

global_teacher_view_idxs = list(range(len(crop_lengths)))  # teacher global view indices

step = 0
for epoch in range(epochs):
    step, epoch_loss = train_one_epoch(
        student,
        teacher,
        dl,
        opt,
        scaler,
        loss,
        lr_schedule,
        teacher_m_schedule,
        global_teacher_view_idxs,
        device,
        step,
        grad_clip_norm,
    )
    print(f"Epoch {epoch+1} finished, avg loss: {epoch_loss}")
