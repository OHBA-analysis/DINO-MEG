"""Trainer."""

import math
import torch
import torch.nn.functional as F


class DINOLoss:
    def __init__(
        self,
        out_dim,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
        device="cpu",
    ):
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.registered_center = torch.zeros(1, out_dim).to(device)
        self.device = device

    def loss(self, student_outputs, teacher_outputs, global_teacher_views):
        t_outs = [t.detach() for t in teacher_outputs]
        s_outs = [s for s in student_outputs]

        n_teacher = len(t_outs)
        n_student = len(s_outs)
        B = s_outs[0].shape[0]
        out_dim = s_outs[0].shape[1]

        t_logits = torch.cat(t_outs, dim=0)
        t_logits_centered = (t_logits - self.registered_center) / self.teacher_temp
        t_probs = F.softmax(t_logits_centered, dim=1)

        s_logits = torch.cat([s / self.student_temp for s in s_outs], dim=0)
        s_log_probs = F.log_softmax(s_logits, dim=1)

        total_loss = 0.0
        n_losses = 0
        for i in range(n_teacher):
            t_slice = t_probs[i * B : (i + 1) * B]
            for j in range(n_student):
                if j == global_teacher_views[i]:
                    continue
                s_slice = s_log_probs[j * B : (j + 1) * B]
                loss_ij = -(t_slice * s_slice).sum(dim=1).mean()
                total_loss += loss_ij
                n_losses += 1
        if n_losses == 0:
            raise ValueError("No student-teacher pairs to compare.")
        total_loss = total_loss / n_losses
        return total_loss

    def update_center(self, teacher_outputs):
        with torch.no_grad():
            cat = torch.cat([t.detach() for t in teacher_outputs], dim=0)
            batch_center = cat.mean(dim=0, keepdim=True)
            self.registered_center = (
                self.registered_center * self.center_momentum
                + batch_center * (1.0 - self.center_momentum)
            )


def linear_warmup_cosine_decay(
    base_value,
    final_value,
    epochs,
    n_iter_per_epoch,
    warmup_epochs=10,
    start_warmup_value=1e-6,
):
    total_iters = epochs * n_iter_per_epoch
    warmup_iters = warmup_epochs * n_iter_per_epoch
    schedule = []
    for it in range(total_iters):
        if it < warmup_iters:
            alpha = it / float(max(1, warmup_iters))
            val = start_warmup_value + alpha * (base_value - start_warmup_value)
        else:
            t = (it - warmup_iters) / float(max(1, total_iters - warmup_iters))
            val = final_value + 0.5 * (base_value - final_value) * (
                1.0 + math.cos(math.pi * t)
            )
        schedule.append(val)
    return schedule


@torch.no_grad()
def momentum_update(teacher_model, student_model, m):
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data.mul_(m).add_(s_param.data * (1.0 - m))


def param_groups_lrd(model, weight_decay):
    # Common rule: no weight decay for bias and norm layers
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim == 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def prepare_view_tensors(batch, device):
    """
    batch: list of samples where each sample is list of view tensors (C, L)
    returns: list of view tensors each shaped (B, C, L) on device
    """
    n_views = len(batch[0])
    view_tensors = []
    for v in range(n_views):
        vt = torch.stack([sample[v] for sample in batch], dim=0).to(device)
        view_tensors.append(vt)
    return view_tensors


def train_one_epoch(
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
):
    """
    Runs one epoch of training and returns updated step and epoch_loss.
    Encapsulates forward, loss, diagnostics, backward, EMA update.
    """
    student.train()
    running_loss = 0.0
    n_iter = len(dl)
    for it, batch in enumerate(dl):
        # prepare tensors per view
        view_tensors = prepare_view_tensors(batch, device)  # list of (B, C, L)

        # forward student with AMP (global and local views)
        student_outputs = []
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for vt in view_tensors:
                student_outputs.append(student(vt))

        # forward teacher (only global views)
        teacher_outputs = []
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                for gi in global_teacher_view_idxs:
                    teacher_outputs.append(teacher(view_tensors[gi]))

        # compute loss
        loss = loss_obj.loss(student_outputs, teacher_outputs, global_teacher_view_idxs)

        # backward with scaler
        opt.zero_grad()
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        scaler.step(opt)
        scaler.update()

        # EMA update for teacher
        m = teacher_m_schedule[step]
        momentum_update(teacher, student, m)

        # update DINO center
        loss_obj.update_center(teacher_outputs)

        # update lr for optimizer param groups
        for pg in opt.param_groups:
            pg["lr"] = lr_schedule[step]

        running_loss += loss.item()
        step += 1

    epoch_loss = running_loss / max(1, n_iter)
    return step, epoch_loss
