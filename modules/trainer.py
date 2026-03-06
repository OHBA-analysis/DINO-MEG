"""Trainer."""

import math
import numpy as np
import torch
import torch.nn as nn
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


class MaskedPatchLoss(nn.Module):
    """Predict teacher patch tokens at masked positions (iBOT-style).

    A lightweight prediction head maps student patch representations to
    teacher space.  Loss is smooth-L1 on L2-normalised vectors at masked
    positions only.
    """

    def __init__(self, d_model):
        super().__init__()
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, student_patches, teacher_patches, mask):
        """Compute prediction loss at masked positions.

        Parameters
        ----------
        student_patches : (B, N, D)
        teacher_patches : (B, N, D)
        mask : (B, N) bool — True at masked positions
        """
        if not mask.any():
            return torch.tensor(0.0, device=student_patches.device)
        s = self.pred_head(student_patches[mask])  # (n_masked, D)
        t = teacher_patches[mask].detach()  # (n_masked, D)
        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        return F.smooth_l1_loss(s, t)


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


def linear_probe_regression_evaluate(backbone, train_loader, eval_loader,
                                      train_dataset, eval_dataset,
                                      n_classes, device,
                                      probe_epochs=50, probe_lr=0.01):
    """Linear probe regression: predict fractional state occupancy.

    Instead of majority-vote classification, predicts the proportion of each
    state within each window (e.g., [0.0, 0.6, 0.4, 0.0, 0.0]).

    Returns (mean_r2, per_state_r2_list).
    """
    was_training = backbone.training
    backbone.eval()

    # Extract features
    def extract_feats(loader):
        feats = []
        with torch.no_grad():
            for images, _ in loader:
                feats.append(backbone(images.to(device)).cpu())
        return torch.cat(feats)

    train_feats = extract_feats(train_loader)
    eval_feats = extract_feats(eval_loader)

    # Compute fractional occupancies from dataset label arrays
    def compute_occupancies(dataset):
        occs = []
        for start in dataset.windows:
            labels = dataset.labels[start:start + dataset.window_length]
            occ = np.bincount(labels, minlength=n_classes).astype(np.float32)
            occ /= dataset.window_length
            occs.append(occ)
        return torch.from_numpy(np.array(occs))

    train_occs = compute_occupancies(train_dataset)
    eval_occs = compute_occupancies(eval_dataset)

    # Train linear regression
    feat_dim = train_feats.shape[1]
    regressor = nn.Linear(feat_dim, n_classes)
    opt = torch.optim.Adam(regressor.parameters(), lr=probe_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(train_feats, train_occs)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)

    regressor.train()
    for _ in range(probe_epochs):
        for feats_b, occs_b in loader:
            pred = regressor(feats_b)
            loss = loss_fn(pred, occs_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    # Evaluate: R² per state
    regressor.eval()
    with torch.no_grad():
        pred = regressor(eval_feats).numpy()
    true = eval_occs.numpy()

    r2_per_state = []
    for c in range(n_classes):
        ss_res = ((true[:, c] - pred[:, c]) ** 2).sum()
        ss_tot = ((true[:, c] - true[:, c].mean()) ** 2).sum()
        r2_per_state.append(float(1.0 - ss_res / (ss_tot + 1e-8)))

    backbone.train(was_training)
    return float(np.mean(r2_per_state)), r2_per_state


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
    mask_loss_obj=None,
    mask_ratio=0.0,
    mask_loss_weight=0.0,
):
    """
    Runs one epoch of training and returns updated step, epoch_loss, mean_grad_norm,
    and center_norm. Encapsulates forward, loss, diagnostics, backward, EMA update.

    When mask_loss_obj is provided, adds iBOT-style masked patch prediction:
    student sees masked patches, predicts teacher's unmasked patch representations.
    """
    student.train()
    if mask_loss_obj is not None:
        mask_loss_obj.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    n_iter = len(dl)
    use_masking = mask_loss_obj is not None and mask_ratio > 0

    for it, batch in enumerate(dl):
        # update lr for optimizer param groups (before step)
        for pg in opt.param_groups:
            pg["lr"] = lr_schedule[step]

        # prepare tensors per view
        view_tensors = prepare_view_tensors(batch, device)  # list of (B, C, L)

        # forward student with AMP — batch views with the same spatial size
        student_outputs = [None] * len(view_tensors)
        student_patch_tokens = [None] * len(view_tensors) if use_masking else None
        view_masks = [None] * len(view_tensors) if use_masking else None

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            groups = {}
            for i, vt in enumerate(view_tensors):
                groups.setdefault(vt.shape[2:], []).append(i)
            for indices in groups.values():
                cat = torch.cat([view_tensors[i] for i in indices], dim=0)
                if use_masking:
                    ps = getattr(student.backbone, 'patch_size', None)
                    if ps is not None:
                        T_len = cat.shape[-1]
                        remainder = T_len % ps
                        n_patches = (T_len + (ps - remainder if remainder else 0)) // ps
                        m = torch.rand(cat.shape[0], n_patches, device=device) < mask_ratio
                        out, pt = student(cat, mask=m, return_patch_tokens=True)
                        B = view_tensors[indices[0]].shape[0]
                        for j, idx in enumerate(indices):
                            student_outputs[idx] = out[j * B:(j + 1) * B]
                            student_patch_tokens[idx] = pt[j * B:(j + 1) * B]
                            view_masks[idx] = m[j * B:(j + 1) * B]
                    else:
                        out = student(cat)
                        for i, chunk in zip(indices, out.chunk(len(indices))):
                            student_outputs[i] = chunk
                else:
                    out = student(cat)
                    for i, chunk in zip(indices, out.chunk(len(indices))):
                        student_outputs[i] = chunk

        # forward teacher
        if use_masking:
            # Teacher processes ALL views (unmasked) for patch token targets
            teacher_all_outputs = [None] * len(view_tensors)
            teacher_patch_tokens = [None] * len(view_tensors)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    groups = {}
                    for i, vt in enumerate(view_tensors):
                        groups.setdefault(vt.shape[2:], []).append(i)
                    for indices in groups.values():
                        cat = torch.cat([view_tensors[i] for i in indices], dim=0)
                        out, pt = teacher(cat, return_patch_tokens=True)
                        B = view_tensors[indices[0]].shape[0]
                        for j, idx in enumerate(indices):
                            teacher_all_outputs[idx] = out[j * B:(j + 1) * B]
                            teacher_patch_tokens[idx] = pt[j * B:(j + 1) * B]
            teacher_outputs = [teacher_all_outputs[gi] for gi in global_teacher_view_idxs]
        else:
            teacher_patch_tokens = None
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

        # compute DINO loss
        t_temp = teacher_temp_schedule[step] if teacher_temp_schedule is not None else None
        loss = loss_obj.loss(student_outputs, teacher_outputs, global_teacher_view_idxs, teacher_temp=t_temp)

        # compute masked patch prediction loss
        if use_masking and student_patch_tokens is not None and teacher_patch_tokens is not None:
            mask_loss = 0.0
            n_mask_terms = 0
            for i in range(len(view_tensors)):
                if (student_patch_tokens[i] is not None
                        and teacher_patch_tokens[i] is not None
                        and view_masks[i] is not None):
                    ml = mask_loss_obj(student_patch_tokens[i],
                                       teacher_patch_tokens[i],
                                       view_masks[i])
                    mask_loss = mask_loss + ml
                    n_mask_terms += 1
            if n_mask_terms > 0:
                mask_loss = mask_loss / n_mask_terms
                loss = loss + mask_loss_weight * mask_loss

        # backward with scaler
        opt.zero_grad()
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(opt)
            params_to_clip = list(student.parameters())
            if mask_loss_obj is not None:
                params_to_clip.extend(mask_loss_obj.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm).item()
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
