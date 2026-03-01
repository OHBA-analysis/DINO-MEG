# DINO-MEG

Self-supervised DINO-style representation learning for MEG data. The model uses a student-teacher architecture with EMA updates, centering, and multi-view augmentation adapted for multivariate time-series (MEG) signals.

## Environment

Conda env: `torch` at `/well/woolrich/users/wlo995/.conda/envs/torch/`

Run scripts with:
```bash
conda run -n torch --no-capture-output python <script.py>
```

Or directly:
```bash
/well/woolrich/users/wlo995/.conda/envs/torch/bin/python <script.py>
```

Jobs are run **interactively** (no SLURM submission during development).

## Project Structure

```
modules/          # Shared library
  model.py        # ConvNet (1D MEG), ConvNet2D (MNIST), Projector, DINOModel
  trainer.py      # DINOLoss, train_one_epoch, knn_evaluate, schedules
  data.py         # MEG augmentations (Transforms), MEGDataset, MEGLabeledDataset

exp_mnist/        # Sanity-check experiment on MNIST (2D conv)
  train.py
  diagnose.py     # Post-hoc: confusion matrix, t-SNE, per-class accuracy
  analyse.py      # Deep analysis: GradCAM, PCA, NN retrieval, sub-clusters, probing
  view_data.py    # Grid visualisation of raw samples
  data/           # Downloaded automatically
  checkpoints/
  figures/

exp_hmm_mvn/      # Validation experiment on HMM-MVN simulated MEG (1D conv, 52ch)
  simulate_data.py
  train.py
  data/           # X_train.npy, Y_train.npy, X_eval.npy, Y_eval.npy

envs/
  torch.yml       # Conda environment spec
```

New experiments follow the `exp_<name>/` pattern with their own `train.py`, `data/`, `checkpoints/`, and `metrics.json`.

## Architecture

- **Backbone**: `ConvNet` — 3-layer Conv1d with BatchNorm + ReLU, AdaptiveAvgPool, Linear → `feat_dim`
- **Projector**: 2-layer MLP → `out_dim` (8192 for MEG, 2048 for MNIST)
- **DINOModel**: backbone + projector + optional predictor; output is L2-normalised
- **Teacher**: EMA copy of student (no grad); momentum annealed 0.996 → 1.0

## Training

- Loss: cross-entropy between centered teacher softmax and student log-softmax
- Center: EMA-updated from teacher outputs (momentum 0.9)
- LR: linear warmup + cosine decay; AdamW with weight decay (bias/norm layers excluded)
- AMP (autocast + GradScaler), gradient clip norm = 1.0
- Evaluation: k-NN on backbone features (cosine similarity) every N epochs

## MEG Data Conventions

- Arrays are `(C, T)` — channels × time samples
- `MEGDataset` returns a list of views per sample: `[global_view_0, global_view_1, local_view_0, local_view_1]`
- Global views use weak transforms; local views use strong transforms
- `MEGLabeledDataset` windows with majority-vote labels, used for k-NN eval only

## Coding Conventions

- Keep shared logic in `modules/`; experiment scripts are self-contained
- Hyperparameters defined as module-level constants at the top of each `train.py`
- Use `torch.multiprocessing.set_sharing_strategy("file_system")` at the top of train scripts (needed on this cluster)
- Checkpoints saved as `checkpoint_epoch{epoch:04d}.pt` with full training state
- Metrics logged to `metrics.json` as a list of dicts (one per epoch)
