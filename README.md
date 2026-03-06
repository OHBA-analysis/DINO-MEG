# DINO-MEG

Self-supervised representation learning for MEG data using a [DINO](https://arxiv.org/abs/2104.14294)-style student-teacher architecture. The model learns to produce consistent representations across augmented views of the same recording segment, without any labels. Learned backbone features are evaluated via k-nearest-neighbour (k-NN) classification on held-out labelled data (where available) or via unsupervised analyses (clustering, nearest-neighbour retrieval, motif discovery).

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
  model.py        # ConvNet, ConvNetV2 (1D MEG), ConvNet2D (MNIST), Projector, DINOModel
  trainer.py      # DINOLoss, train_one_epoch, knn_evaluate, schedules, EMA utilities
  data.py         # MEG augmentations (Transforms), MEGDataset, MEGLabeledDataset

exp_mnist/        # Sanity-check experiment on MNIST (2D conv)
  train.py
  diagnose.py     # Post-hoc: confusion matrix, t-SNE, per-class accuracy
  analyse.py      # Deep analysis: GradCAM, PCA, NN retrieval, sub-clusters, probing
  view_data.py    # Grid visualisation of raw samples

exp_hmm_mvn/      # Validation experiment on oscillatory burst simulated MEG (1D conv, 52ch)
  simulate_data.py  # Markov-switched theta/alpha/beta/gamma bursts + pink noise, Glasser52 spatial patterns
  train.py
  diagnose.py     # Post-hoc: training curves, confusion matrix, per-state accuracy, t-SNE
  analyse.py      # Deep analysis: GradCAM, channel attribution, PCA, NN retrieval, feature-frequency correlation
  view_data.py    # Raw signal visualisation: timeseries, PSD, spectrogram, state durations
  supervised_sanity.py  # Supervised baseline for comparison

exp_motif_1ch/    # Single-channel oscillatory motif detection (1D conv, ConvNetV2)
  simulate_data.py  # Single-channel Markov-switched bursts at varying frequencies
  train.py
  diagnose.py     # Training curves, confusion matrix, t-SNE
  analyse.py      # GradCAM, NN retrieval, PCA, cluster analysis
  view_data.py    # Raw signal visualisation
  summary.py      # Summary statistics and model performance overview
  supervised_sanity.py  # Supervised baseline

exp_real_meg/     # Real MEG experiment on Cam-CAN parcellated data (ConvNetV2 + conditioning)
  train.py        # DINO with channel/subject identity embeddings
  diagnose.py     # Training curves, PCA/t-SNE scatter plots, eigenspectrum, embedding viz
  analyse.py      # Attention maps, NN retrieval, motif discovery, age correlation

envs/
  torch.yml       # Conda environment spec
```

New experiments follow the `exp_<name>/` pattern with their own `train.py`, `data/`, `checkpoints/`, and `figures/`.

---

## Architecture

### Backbones

- **ConvNet** (1D): 3-layer Conv1d (11->9->5 kernel, stride 2 each) with BatchNorm + ReLU, `AdaptiveAvgPool1d(1)`, Linear -> `feat_dim`. Input shape `(B, C, T)`.
- **ConvNetV2** (1D): Multi-scale stem (parallel convolutions at different kernel sizes), residual blocks with GroupNorm + GELU, temporal attention pooling, Linear -> `feat_dim`. Supports `return_attention=True` for interpretability.
- **ConvNet2D** (2D): Same design as ConvNet with Conv2d and `AdaptiveAvgPool2d(1)` for image data.

### Projection head (`Projector`)

2-layer MLP:

```
Linear(feat_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> out_dim)
```

### Student-teacher (`DINOModel`)

- **Teacher**: EMA copy of the student; no gradients; momentum annealed from 0.996 -> 1.0 via cosine schedule.
- Both student and teacher share the same `backbone + projector` architecture. Output is L2-normalised.

### Conditioned model (`ConditionedDINOModel`, in `exp_real_meg/train.py`)

For real MEG with multiple channels and subjects:

```
waveform (1, T) -> ConvNetV2 -> feat (256)
channel_id      -> Embedding(52, 32)  -> ch_emb (32)
subject_id      -> Embedding(N, 32)   -> subj_emb (32)
                                         |
              concat(feat, ch_emb, subj_emb) -> (320)
                                         |
                    Projector(320, hidden, out_dim) -> L2-norm -> DINO loss
```

---

## Training

### Loss

Cross-entropy between the teacher's centered, temperature-sharpened softmax and the student's log-softmax, averaged over all valid teacher-student view pairs (same-index pairs are excluded):

```
L = -sum softmax((t - center) / tau_t) . log softmax(s / tau_s)
```

The center is updated each step as an EMA of teacher outputs (momentum 0.9).

### Key hyperparameters

| | MNIST | HMM-MVN | Motif 1ch | Real MEG |
|---|---|---|---|---|
| Backbone | ConvNet2D | ConvNet (52ch) | ConvNetV2 (1ch) | ConvNetV2 (1ch) |
| `feat_dim` | 256 | 512 | 256 | 256 |
| `hidden_dim` | 1024 | 1024 | 512 | 512 |
| `out_dim` | 2048 | 8192 | 4096 | 4096 |
| Global crop size | 28 px | 1000 samples | 112 samples | 112 samples |
| Local crop size | 14 px | 500 samples | 100 samples | 100 samples |
| Batch size | 512 | 32 | 256 | 2048 |
| Learning rate | 3e-4 | 5e-4 | 5e-4 | 1e-3 |
| Teacher momentum | 0.996 | 0.996 | 0.996 | 0.9998 |
| Student temp | 0.1 | 0.1 | 0.1 | 0.05 |
| AMP | enabled | enabled | enabled | enabled |

**Note on teacher momentum scaling**: With larger datasets (more steps per epoch), teacher momentum must be increased to prevent the teacher from being fully replaced each epoch. For real MEG with ~1400 steps/epoch, `0.9998` provides equivalent per-epoch teacher retention to `0.996` with ~80 steps/epoch.

### Evaluation

- **Labelled experiments** (MNIST, HMM-MVN, Motif 1ch): k-NN classification (cosine similarity, k=20) on backbone features, computed every N epochs.
- **Real MEG** (no labels): loss, gradient norm, center norm, feature effective rank.

---

## Experiments

### MNIST (sanity check)

```bash
python exp_mnist/train.py
python exp_mnist/diagnose.py
python exp_mnist/analyse.py
```

Trains DINO on unlabelled MNIST images. Checks that the self-supervised objective learns digit-discriminative features. Expected k-NN top-1: ~93-96%.

### HMM-MVN (oscillatory burst simulation, 52 channels)

```bash
python exp_hmm_mvn/simulate_data.py
python exp_hmm_mvn/train.py
python exp_hmm_mvn/diagnose.py
python exp_hmm_mvn/analyse.py
```

Simulates 5-state Markov-switched oscillatory bursts (theta/alpha/beta/gamma) with pink noise across 52 channels using Glasser parcellation spatial patterns. The model must learn to separate hidden states from unlabelled data.

### Motif 1ch (single-channel oscillatory motif detection)

```bash
python exp_motif_1ch/simulate_data.py
python exp_motif_1ch/train.py
python exp_motif_1ch/diagnose.py
python exp_motif_1ch/analyse.py
```

Single-channel validation using ConvNetV2. Simulates Markov-switched oscillatory bursts at varying frequencies in a single channel. Tests the backbone architecture used for real MEG.

### Real MEG (Cam-CAN parcellated data)

```bash
python exp_real_meg/train.py
python exp_real_meg/diagnose.py
python exp_real_meg/analyse.py
```

Applies DINO to real Cam-CAN source-reconstructed MEG data (52 parcels per subject). Each parcel channel is treated as an independent (1, T) stream. Channel and subject identity embeddings are concatenated to backbone features before the projector.

**Data**: `/well/win-camcan/shared/spring23/src/sub-*/sflip_parc-raw.fif`

**Analyses include**:
- Temporal attention maps by brain region
- Nearest-neighbour retrieval (within and across subjects)
- Motif discovery: k-means clustering with average waveform visualization
- Per-subject motif frequency heatmap
- Age correlation of motif frequencies and PCA features (using `participants.tsv`)
- Channel and subject similarity matrices with hierarchical clustering
- Cross-subject consistency analysis

---

## MEG data conventions

- Arrays are `(C, T)` — channels x time samples.
- `MEGDataset` slices recordings into windows and returns `[global_view_0, global_view_1, local_view_0, ..., local_view_N]` per sample.
- `MEGLabeledDataset` windows with majority-vote labels, used only for k-NN evaluation.

---

## Outputs

Each experiment writes to its `checkpoints/` and `figures/` directories:

```
checkpoints/
  checkpoint_epoch{N:04d}.pt   # full training state (student, teacher, opt, scaler)
  backbone_final.pt             # backbone weights only, for downstream use
  metrics.json                  # list of per-epoch dicts (loss, grad_norm, knn_top1, ...)
  metadata.json                 # (real MEG) subject/channel mapping

figures/
  training_curves.png           # loss, grad norm, center norm, feature rank
  tsne_by_*.png                 # t-SNE scatter plots coloured by different labels
  pca_by_*.png                  # PCA scatter plots
  motif_waveforms.png           # (real MEG) average waveform per motif cluster
  motif_age_correlation.png     # (real MEG) motif frequency vs subject age
  ...                           # experiment-specific figures
```
