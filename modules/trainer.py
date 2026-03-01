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

    def loss(self, student_outputs, teacher_outputs, global_teacher_views, teacher_temp=None):
        teacher_temp = teacher_temp if teacher_temp is not None else self.teacher_temp
        t_outs = [t.detach() for t in teacher_outputs]

        n_teacher = len(t_outs)
        n_student = len(student_outputs)
        B = student_outputs[0].shape[0]
        out_dim = student_outputs[0].shape[1]

        t_logits = torch.cat(t_outs, dim=0)
        t_logits_centered = (t_logits - self.registered_center) / teacher_temp
        t_probs = F.softmax(t_logits_centered, dim=1)

        s_logits = torch.cat([s / self.student_temp for s in student_outputs], dim=0)
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
    batch: list of samples where each sample is list of view tensors (C, ...).
    returns: list of view tensors each shaped (B, C, ...) on device.
    """
    n_views = len(batch[0])
    view_tensors = []
    for v in range(n_views):
        vt = torch.stack([sample[v] for sample in batch], dim=0).to(device)
        view_tensors.append(vt)
    return view_tensors


@torch.no_grad()
def knn_evaluate(backbone, train_loader, val_loader, k, device):
    """k-NN evaluation using cosine similarity on backbone features.

    Returns (top1_acc, top5_acc) as floats in [0, 1].
    """
    was_training = backbone.training
    backbone.eval()

    # collect train features and labels
    train_feats, train_labels = [], []
    for images, labels in train_loader:
        images = images.to(device)
        feats = backbone(images)
        feats = F.normalize(feats, dim=1)
        train_feats.append(feats.cpu())
        train_labels.append(labels)
    train_feats = torch.cat(train_feats, dim=0)    # (N_train, D)
    train_labels = torch.cat(train_labels, dim=0)  # (N_train,)

    # evaluate on val set
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        feats = backbone(images)
        feats = F.normalize(feats, dim=1).cpu()    # (B, D)

        sims = feats @ train_feats.t()             # (B, N_train)
        topk_indices = sims.topk(k, dim=1).indices # (B, k)
        topk_labels = train_labels[topk_indices]   # (B, k)

        for i in range(feats.shape[0]):
            true_label = labels[i].item()
            neighbors = topk_labels[i]             # (k,)
            # top-1: majority vote
            counts = torch.bincount(neighbors)
            pred = counts.argmax().item()
            if pred == true_label:
                correct_top1 += 1
            # top-5: is true label among top-5 most frequent candidates?
            top5_labels = counts.topk(min(5, len(counts))).indices
            if true_label in top5_labels.tolist():
                correct_top5 += 1
        total += feats.shape[0]

    backbone.train(was_training)
    top1 = correct_top1 / total
    top5 = correct_top5 / total
    return top1, top5


def linear_probe_evaluate(backbone, train_loader, val_loader, device, probe_epochs=50, probe_lr=0.1):
    """Linear probe evaluation on frozen backbone features.

    Trains a linear classifier on extracted features and returns accuracy.
    Returns (top1_acc, top5_acc) as floats in [0, 1].
    """
    import torch.nn as nn

    was_training = backbone.training
    backbone.eval()

    # extract features
    train_feats, train_labels = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            feats = backbone(images.to(device))
            train_feats.append(feats.cpu())
            train_labels.append(labels)
    train_feats = torch.cat(train_feats, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    val_feats, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            feats = backbone(images.to(device))
            val_feats.append(feats.cpu())
            val_labels.append(labels)
    val_feats = torch.cat(val_feats, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # train linear classifier
    feat_dim = train_feats.shape[1]
    n_classes = int(train_labels.max().item()) + 1
    classifier = nn.Linear(feat_dim, n_classes)
    opt = torch.optim.SGD(classifier.parameters(), lr=probe_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    classifier.train()
    for _ in range(probe_epochs):
        for feats_b, labels_b in loader:
            logits = classifier(feats_b)
            loss = loss_fn(logits, labels_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    # evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(val_feats)
        preds = logits.argmax(dim=1)
        top1 = (preds == val_labels).float().mean().item()
        top5_preds = logits.topk(min(5, n_classes), dim=1).indices
        top5 = sum(val_labels[i] in top5_preds[i] for i in range(len(val_labels))) / len(val_labels)

    backbone.train(was_training)
    return top1, top5


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
    teacher_temp_schedule=None,
):
    """
    Runs one epoch of training and returns updated step, epoch_loss, mean_grad_norm,
    and center_norm. Encapsulates forward, loss, diagnostics, backward, EMA update.
    """
    student.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    n_iter = len(dl)
    for it, batch in enumerate(dl):
        # update lr for optimizer param groups (before step)
        for pg in opt.param_groups:
            pg["lr"] = lr_schedule[step]

        # prepare tensors per view
        view_tensors = prepare_view_tensors(batch, device)  # list of (B, C, L)

        # forward student with AMP — batch views with the same spatial size
        student_outputs = [None] * len(view_tensors)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            groups = {}
            for i, vt in enumerate(view_tensors):
                groups.setdefault(vt.shape[2:], []).append(i)
            for indices in groups.values():
                cat = torch.cat([view_tensors[i] for i in indices], dim=0)
                out = student(cat)
                for i, chunk in zip(indices, out.chunk(len(indices))):
                    student_outputs[i] = chunk

        # forward teacher (only global views) — batch same-size globals
        teacher_outputs = [None] * len(global_teacher_view_idxs)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                groups = {}
                for j, gi in enumerate(global_teacher_view_idxs):
                    groups.setdefault(view_tensors[gi].shape[2:], []).append(j)
                for indices in groups.values():
                    cat = torch.cat([view_tensors[global_teacher_view_idxs[j]] for j in indices], dim=0)
                    out = teacher(cat)
                    for j, chunk in zip(indices, out.chunk(len(indices))):
                        teacher_outputs[j] = chunk

        # compute loss
        t_temp = teacher_temp_schedule[step] if teacher_temp_schedule is not None else None
        loss = loss_obj.loss(student_outputs, teacher_outputs, global_teacher_view_idxs, teacher_temp=t_temp)

        # backward with scaler
        opt.zero_grad()
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm).item()
        else:
            grad_norm = 0.0
        scaler.step(opt)
        scaler.update()

        # EMA update for teacher
        m = teacher_m_schedule[step]
        momentum_update(teacher, student, m)

        # update DINO center
        loss_obj.update_center(teacher_outputs)

        running_loss += loss.item()
        running_grad_norm += grad_norm
        step += 1

    epoch_loss = running_loss / max(1, n_iter)
    mean_grad_norm = running_grad_norm / max(1, n_iter)
    center_norm = loss_obj.registered_center.norm().item()
    return step, epoch_loss, mean_grad_norm, center_norm
