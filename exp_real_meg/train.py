"""Train DINO on real MEG: Cam-CAN source-reconstructed parcellated data.

Each subject has 52 misc channels. Each channel is treated as an independent
(1, T) stream. Channel and subject identity embeddings are concatenated to
backbone features before the projector.

Data: /well/win-camcan/shared/spring23/src/sub-*/sflip_parc-raw.fif

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_real_meg/train.py
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data import Transforms, MEGDataset
from modules.model import ConvNetV2, Projector
from modules.trainer import (
    DINOLoss,
    param_groups_lrd,
    linear_warmup_cosine_decay,
    momentum_update,
    prepare_view_tensors,
)

# ----------
# Parameters
# ----------

DATA_ROOT = "/well/win-camcan/shared/spring23/src"
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
METRICS_FILE = os.path.join(CKPT_DIR, "metrics.json")
os.makedirs(CKPT_DIR, exist_ok=True)

# How many subjects to use (None = all available)
max_subjects = 50

n_channels = 1  # each parcel channel treated independently
n_parcels = 52
sampling_frequency = 250
window_length = 150
stride = 125
crop_lengths = [112, 112]
local_crop_length = 100
n_local_crops = 2

epochs = 20
batch_size = 2048
feat_dim = 256
emb_dim = 32
proj_in_dim = feat_dim + 2 * emb_dim  # 320
hidden_dim = 512
out_dim = 4096
use_predictor = False

lr = 1e-3
lr_start = 1e-6
lr_final_scale = 0.01
warmup_epochs = 3
weight_decay = 0.05
teacher_momentum = 0.9998
teacher_momentum_final = 1.0
teacher_temp = 0.04
teacher_temp_final = 0.02
teacher_temp_warmup_epochs = 5
student_temp = 0.05
center_momentum = 0.9
grad_clip_norm = 1.0
save_every = 2

# ConvNetV2 params
v2_stem_channels = 32
v2_stem_kernel_sizes = (7, 15, 31)
v2_block_channels = (128, 256)
v2_block_kernel_sizes = (9, 5)
v2_attn_hidden = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------
# Conditioned DINO model (local)
# --------------------------------

class ConditionedDINOModel(nn.Module):
    """DINO model with channel and subject conditioning.

    backbone(x) -> feat (feat_dim)
    concat(feat, channel_emb, subject_emb) -> projector -> L2-norm
    """

    def __init__(self, backbone, projector, n_channels, n_subjects,
                 emb_dim=32, use_predictor=False):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.channel_emb = nn.Embedding(n_channels, emb_dim)
        self.subject_emb = nn.Embedding(n_subjects, emb_dim)
        self.use_predictor = use_predictor

    def forward(self, x, channel_id, subject_id):
        feat = self.backbone(x)  # (B, feat_dim)
        ch_e = self.channel_emb(channel_id)  # (B, emb_dim)
        su_e = self.subject_emb(subject_id)  # (B, emb_dim)
        combined = torch.cat([feat, ch_e, su_e], dim=1)  # (B, proj_in_dim)
        proj = self.projector(combined)  # (B, out_dim)
        return F.normalize(proj, dim=1)


# --------------------------------
# Data loading
# --------------------------------

print("\n--- Loading Cam-CAN data ---")

fif_files = sorted(glob.glob(os.path.join(DATA_ROOT, "sub-*/sflip_parc-raw.fif")))
if max_subjects is not None:
    fif_files = fif_files[:max_subjects]
print(f"Found {len(fif_files)} subjects")

# Lazy import MNE (heavy)
import mne
mne.set_log_level("WARNING")

arrays = []       # list of (1, T_i) arrays, one per (subject, channel)
metadata = {}     # file_id -> {subject_idx, channel_idx, subject_name, channel_name}
subject_names = []

file_id = 0
for subj_idx, fif_path in enumerate(tqdm(fif_files, desc="Loading subjects")):
    subj_name = os.path.basename(os.path.dirname(fif_path))
    subject_names.append(subj_name)

    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.pick("misc")
    data = raw.get_data()  # (52, T)
    ch_names = raw.ch_names
    fs = raw.info["sfreq"]

    if abs(fs - sampling_frequency) > 1:
        print(f"  WARNING: {subj_name} has fs={fs}, expected {sampling_frequency}. Skipping.")
        subject_names.pop()
        continue

    for ch_idx in range(data.shape[0]):
        ch_data = data[ch_idx]  # (T,)
        # z-score per channel
        mu = ch_data.mean()
        std = ch_data.std()
        if std > 0:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu
        arr = ch_data[np.newaxis, :].astype(np.float32)  # (1, T)
        arrays.append(arr)
        metadata[file_id] = {
            "subject_idx": subj_idx,
            "channel_idx": ch_idx,
            "subject_name": subj_name,
            "channel_name": ch_names[ch_idx] if ch_idx < len(ch_names) else f"ch{ch_idx}",
        }
        file_id += 1

n_subjects = len(subject_names)
n_total_files = len(arrays)
print(f"Loaded {n_subjects} subjects, {n_total_files} channel-streams")
print(f"  Example array shape: {arrays[0].shape}")

# Save metadata
meta_save = {
    "subject_names": subject_names,
    "n_subjects": n_subjects,
    "n_parcels": n_parcels,
    "n_files": n_total_files,
    "metadata": {str(k): v for k, v in metadata.items()},
}
with open(os.path.join(CKPT_DIR, "metadata.json"), "w") as f:
    json.dump(meta_save, f, indent=2)
print(f"Metadata saved to {CKPT_DIR}/metadata.json")


# --------------------------------
# Dataset with conditioning info
# --------------------------------

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


class ConditionedMEGDataset(MEGDataset):
    """MEGDataset subclass that also returns file_id for conditioning lookup."""

    def __getitem__(self, idx):
        views = super().__getitem__(idx)
        file_id, _ = self.windows[idx]
        return views, file_id


def conditioned_collate(batch):
    """Custom collate: separate views and file_ids."""
    views_list = [item[0] for item in batch]
    file_ids = [item[1] for item in batch]
    return views_list, file_ids


dataset = ConditionedMEGDataset(
    arrays=arrays,
    window_length=window_length,
    stride=stride,
    crop_lengths=crop_lengths,
    n_local_crops=n_local_crops,
    local_crop_length=local_crop_length,
    weak_transform=weak_transform,
    strong_transform=strong_transform,
)
# Free raw arrays (data is already copied into dataset.data)
del arrays

dl = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    collate_fn=conditioned_collate,
)

n_iter_per_epoch = len(dl)
print(f"  Windows: {len(dataset)}, steps/epoch: {n_iter_per_epoch}")


# --------------------------------
# Model
# --------------------------------

print("\nBuilding model...")
backbone = ConvNetV2(
    in_channels=n_channels, feat_dim=feat_dim,
    stem_channels=v2_stem_channels, stem_kernel_sizes=v2_stem_kernel_sizes,
    block_channels=v2_block_channels, block_kernel_sizes=v2_block_kernel_sizes,
    attn_hidden=v2_attn_hidden,
)
projector = Projector(in_dim=proj_in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

student = ConditionedDINOModel(
    backbone, projector, n_channels=n_parcels, n_subjects=n_subjects,
    emb_dim=emb_dim, use_predictor=use_predictor,
).to(device)

teacher = deepcopy(student).to(device)
for p in teacher.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in student.parameters())
print(f"  Parameters: {n_params:,}")


# --------------------------------
# Optimizer and schedules
# --------------------------------

param_groups = param_groups_lrd(student, weight_decay=weight_decay)
opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

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


# --------------------------------
# Loss
# --------------------------------

loss_obj = DINOLoss(
    out_dim=out_dim,
    teacher_temp=teacher_temp,
    student_temp=student_temp,
    center_momentum=center_momentum,
    device=device,
)
global_teacher_view_idxs = list(range(len(crop_lengths)))


# --------------------------------
# Helpers
# --------------------------------

def compute_effective_rank(feats):
    """Effective rank of feature matrix (higher = less collapsed)."""
    # feats: (N, D) — normalize singular values to get distribution
    _, s, _ = torch.svd(feats - feats.mean(dim=0, keepdim=True))
    p = s / s.sum()
    p = p[p > 1e-10]
    return torch.exp(-torch.sum(p * torch.log(p))).item()


# --------------------------------
# Training loop
# --------------------------------

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
metrics = []
step = 0

print(f"\nStarting training: {epochs} epochs, batch {batch_size}, {n_iter_per_epoch} steps/ep")
print(f"  Checkpoints: {CKPT_DIR}")
print(f"  Metrics:     {METRICS_FILE}\n")
train_start = time.time()

pbar = tqdm(range(epochs), desc="Training", unit="ep")
for epoch in pbar:
    student.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    epoch_feats = []

    for it, (views_list, file_ids) in enumerate(dl):
        # Update LR
        for pg in opt.param_groups:
            pg["lr"] = lr_schedule[step]

        # Prepare view tensors: list of (B, 1, L)
        view_tensors = prepare_view_tensors(views_list, device)

        # Prepare conditioning tensors
        channel_ids = torch.tensor(
            [metadata[fid]["channel_idx"] for fid in file_ids],
            dtype=torch.long, device=device,
        )
        subject_ids = torch.tensor(
            [metadata[fid]["subject_idx"] for fid in file_ids],
            dtype=torch.long, device=device,
        )

        # Forward student with AMP
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            student_outputs = []
            for vt in view_tensors:
                out = student(vt, channel_ids, subject_ids)
                student_outputs.append(out)

        # Forward teacher (no grad)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                teacher_outputs = []
                for gi in global_teacher_view_idxs:
                    out = teacher(view_tensors[gi], channel_ids, subject_ids)
                    teacher_outputs.append(out)

        # DINO loss
        t_temp = teacher_temp_schedule[step]
        loss = loss_obj.loss(
            student_outputs, teacher_outputs,
            global_teacher_view_idxs, teacher_temp=t_temp,
        )

        # Backward
        opt.zero_grad()
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), grad_clip_norm,
            ).item()
        else:
            grad_norm = 0.0
        scaler.step(opt)
        scaler.update()

        # EMA update
        m = teacher_m_schedule[step]
        momentum_update(teacher, student, m)
        loss_obj.update_center(teacher_outputs)

        running_loss += loss.item()
        running_grad_norm += grad_norm

        # Collect backbone features for rank estimation (subsample)
        if it % 10 == 0:
            with torch.no_grad():
                bf = student.backbone(view_tensors[0])
                epoch_feats.append(bf.cpu())

        step += 1

    epoch_loss = running_loss / max(1, n_iter_per_epoch)
    mean_grad_norm = running_grad_norm / max(1, n_iter_per_epoch)
    center_norm = loss_obj.registered_center.norm().item()

    # Compute feature effective rank
    if epoch_feats:
        all_feats = torch.cat(epoch_feats, dim=0)
        feat_rank = compute_effective_rank(all_feats[:2000])
    else:
        feat_rank = 0.0

    row = {
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "grad_norm": mean_grad_norm,
        "center_norm": center_norm,
        "feat_rank": feat_rank,
    }
    metrics.append(row)

    tqdm.write(
        f"Epoch {epoch+1:3d} | loss {epoch_loss:.4f} "
        f"| grad {mean_grad_norm:.3f} | center {center_norm:.3f} "
        f"| rank {feat_rank:.1f}"
    )

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

# Save final backbone + embeddings
final_state = {
    "backbone": student.backbone.state_dict(),
    "channel_emb": student.channel_emb.state_dict(),
    "subject_emb": student.subject_emb.state_dict(),
}
torch.save(final_state, os.path.join(CKPT_DIR, "backbone_final.pt"))
print(f"Backbone + embeddings saved to {CKPT_DIR}/backbone_final.pt")
