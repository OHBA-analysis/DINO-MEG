# DINO-MEG

Self-supervised representation learning for MEG data using a [DINO](https://arxiv.org/abs/2104.14294)-style student-teacher architecture. The model learns to produce consistent representations across augmented views of the same recording segment, without any labels. Learned backbone features are evaluated via k-nearest-neighbour (k-NN) classification on held-out labelled data.

---

## Setup

```bash
conda env create -f envs/torch.yml
conda activate torch
```

The environment installs PyTorch (CUDA 12.1), torchvision, scikit-learn, MNE, and other dependencies.

---

## Project structure

```
modules/
  model.py        # ConvNet (1D MEG), ConvNet2D (MNIST), Projector, DINOModel
  trainer.py      # DINOLoss, train_one_epoch, knn_evaluate, schedules
  data.py         # MEG augmentations (Transforms), MEGDataset, MEGLabeledDataset

exp_mnist/        # Sanity-check experiment on MNIST (2D conv)
  train.py
  diagnose.py     # Post-hoc: confusion matrix, t-SNE, per-class accuracy
  view_data.py    # Grid visualisation of raw samples

exp_hmm_mvn/      # Validation experiment on HMM-MVN simulated MEG (1D conv, 52 ch)
  simulate_data.py
  train.py

envs/
  torch.yml       # Conda environment spec
```

New experiments follow the `exp_<name>/` pattern with their own `train.py`, `data/`, `checkpoints/`, and `metrics.json`.

---

## Architecture

### Backbone

- **MEG (`ConvNet`)**: 3-layer Conv1d (11→9→5 kernel, stride 2 each) with BatchNorm + ReLU, `AdaptiveAvgPool1d(1)`, Linear → `feat_dim`. Input shape `(B, C, T)`.
- **MNIST (`ConvNet2D`)**: Same design with Conv2d and `AdaptiveAvgPool2d(1)`.

### Projection head (`Projector`)

2-layer MLP:

```
Linear(feat_dim → hidden_dim) → GELU → Linear(hidden_dim → out_dim)
```

### Student-teacher (`DINOModel`)

- **Teacher**: EMA copy of the student; no gradients; momentum annealed from 0.996 → 1.0 via cosine schedule.
- Both student and teacher share the same `backbone + projector` architecture. Output is L2-normalised.

---

## Training

### Loss

Cross-entropy between the teacher's centered, temperature-sharpened softmax and the student's log-softmax, averaged over all valid teacher–student view pairs (same-index pairs are excluded):

```
L = -Σ softmax((t - center) / τ_t) · log softmax(s / τ_s)
```

The center is updated each step as an EMA of teacher outputs (momentum 0.9).

### Schedules

| Schedule | MNIST | HMM-MVN |
|---|---|---|
| Learning rate | Linear warmup → cosine decay | Linear warmup → cosine decay |
| Weight decay | Fixed (`0.04`, decay params only) | Fixed (`0.01`, decay params only) |
| Teacher momentum | Cosine from `0.996` → `1.0` | Cosine from `0.996` → `1.0` |
| Teacher temperature | Linear warmup `0.04` → `0.07` over 30 epochs | Linear warmup `0.04` → `0.07` over 30 epochs |

### Key hyperparameters

| | MNIST | HMM-MVN |
|---|---|---|
| Backbone | ConvNet2D | ConvNet (52 ch) |
| `feat_dim` | 256 | 512 |
| `hidden_dim` | 1024 | 1024 |
| `out_dim` | 2048 | 8192 |
| Global crop size | 28 px | 1000 samples (4 s) |
| Local crop size | 14 px | 500 samples (2 s) |
| Local crops | 2 | 2 |
| Batch size | 512 | 32 |
| Grad clip norm | 1.0 | 1.0 |
| AMP | enabled (on CUDA) | enabled (on CUDA) |

### Evaluation

k-NN classification (cosine similarity, k=20) on backbone features, computed every N epochs on a held-out labelled set. No fine-tuning.

---

## Experiments

### MNIST (sanity check)

```bash
python exp_mnist/train.py
```

Trains a DINO model on unlabelled MNIST images. Checks that the self-supervised objective learns digit-discriminative features. Expected k-NN top-1: ~92–96%.

**Augmentations** (per view):
- `RandomResizedCrop` (global: 28 px / local: 14 px)
- `RandomRotation(15°)`
- `RandomAffine(shear=10, translate=0.1)` — mimics handwriting-style deformation
- `GaussianBlur(kernel=3, σ=0.1–1.0)` with probability 0.5
- Normalise

**Diagnostics:**
```bash
python exp_mnist/diagnose.py   # confusion matrix, t-SNE, per-class accuracy
python exp_mnist/view_data.py  # grid of raw samples
```

### HMM-MVN (MEG validation)

Simulates 5-state HMM data with multivariate normal emissions across 52 channels at 250 Hz. The model must learn to separate hidden states from unlabelled data.

```bash
# generate data (~5 min at 250 Hz, 5 hidden states, 52 channels)
python exp_hmm_mvn/simulate_data.py

# train
python exp_hmm_mvn/train.py
```

**Augmentations** (weak / strong, applied stochastically):
- Gaussian noise, baseline shift, amplitude scaling
- Time warping (±12–15%), time masking (up to 2 segments)
- Channel dropout (up to 15% of channels)
- FFT notch filter at 50 Hz (power-line artefact removal)
- Random time reversal

---

## MEG data conventions

- Arrays are `(C, T)` — channels × time samples.
- `MEGDataset` slices recordings into windows and returns `[global_view_0, global_view_1, local_view_0, ..., local_view_N]` per sample.
- `MEGLabeledDataset` windows with majority-vote labels, used only for k-NN evaluation.

---

## Outputs

Each experiment writes to its `checkpoints/` directory:

```
checkpoints/
  checkpoint_epoch{N:04d}.pt   # full training state (student, teacher, opt, scaler)
  backbone_final.pt             # backbone weights only, for downstream use
  metrics.json                  # list of per-epoch dicts (loss, grad_norm, knn_top1, ...)
```
