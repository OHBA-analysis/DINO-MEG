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
  model.py        # ConvNet, ConvNetV2 (1D MEG), ConvNet2D (MNIST), Projector, DINOModel
  trainer.py      # DINOLoss, train_one_epoch, knn_evaluate, schedules, momentum_update, prepare_view_tensors
  data.py         # MEG augmentations (Transforms), MEGDataset, MEGLabeledDataset

exp_mnist/        # Sanity-check experiment on MNIST (2D conv)
  train.py
  diagnose.py     # Post-hoc: confusion matrix, t-SNE, per-class accuracy
  analyse.py      # Deep analysis: GradCAM, PCA, NN retrieval, sub-clusters, probing
  view_data.py    # Grid visualisation of raw samples
  data/           # Downloaded automatically
  checkpoints/
  figures/

exp_hmm_mvn/      # Validation experiment on oscillatory burst simulated MEG (1D conv, 52ch)
  simulate_data.py  # Markov-switched theta/alpha/beta/gamma bursts + pink noise, Glasser52 spatial patterns
  train.py
  diagnose.py     # Post-hoc: training curves, confusion matrix, per-state accuracy, t-SNE
  analyse.py      # Deep analysis: GradCAM, channel attribution vs ground truth, PCA, NN retrieval, feature-frequency correlation, inter-state similarity
  view_data.py    # Raw signal visualisation: timeseries, PSD, spectrogram, state durations, power maps
  supervised_sanity.py  # Supervised baseline for comparison
  data/           # X_train.npy, Y_train.npy, X_eval.npy, Y_eval.npy
  checkpoints/
  figures/

exp_motif_1ch/    # Single-channel oscillatory motif detection (1D conv, ConvNetV2)
  simulate_data.py  # Single-channel Markov-switched bursts at varying frequencies
  train.py
  diagnose.py     # Training curves, confusion matrix, t-SNE
  analyse.py      # GradCAM, NN retrieval, PCA, cluster analysis
  view_data.py    # Raw signal visualisation
  summary.py      # Summary statistics and model performance overview
  supervised_sanity.py  # Supervised baseline
  data/
  checkpoints/
  figures/

exp_real_meg/     # Real MEG experiment on Cam-CAN parcellated data (ConvNetV2 + conditioning)
  train.py        # DINO with channel/subject identity embeddings, GPU augmentation, pre-extracted windows
  diagnose.py     # Training curves, PCA/t-SNE scatter plots, eigenspectrum, embedding viz
  analyse.py      # Attention maps, NN retrieval, motif discovery, per-subject frequency, age correlation
  interpret.py    # Stem filter viz, per-motif PSD, optimal input synthesis, gradient saliency, branch ablation
  checkpoints/
  figures/

envs/
  torch.yml       # Conda environment spec
```

New experiments follow the `exp_<name>/` pattern with their own `train.py`, `data/`, `checkpoints/`, and `figures/`.

## Architecture

- **Backbone (ConvNet)**: 3-layer Conv1d with BatchNorm + ReLU, AdaptiveAvgPool, Linear -> `feat_dim`
- **Backbone (ConvNetV2)**: Multi-scale stem (parallel convolutions), residual blocks with GroupNorm + GELU, temporal attention pooling, Linear -> `feat_dim`. Supports `return_attention=True` for interpretability.
- **Projector**: 2-layer MLP -> `out_dim` (4096 for motif/real MEG, 8192 for HMM-MVN, 2048 for MNIST)
- **DINOModel**: backbone + projector + optional predictor; output is L2-normalised
- **ConditionedDINOModel** (exp_real_meg): backbone + channel/subject embeddings concatenated before projector
- **Teacher**: EMA copy of student (no grad); momentum annealed 0.996 -> 1.0 (or 0.9999 -> 1.0 for full Cam-CAN)

## Training

- Loss: cross-entropy between centered teacher softmax and student log-softmax
- Center: EMA-updated from teacher outputs (momentum 0.9)
- LR: linear warmup + cosine decay; AdamW with weight decay (bias/norm layers excluded)
- AMP (autocast + GradScaler), gradient clip norm = 1.0 (0.3 for full Cam-CAN)
- Evaluation: k-NN on backbone features (cosine similarity) every N epochs (where labels available)
- Real MEG (no labels): monitor loss, grad_norm, center_norm, feature effective rank

## MEG Data Conventions

- Arrays are `(C, T)` — channels x time samples
- `MEGDataset` returns a list of views per sample: `[global_view_0, global_view_1, local_view_0, local_view_1]`
- Global views use weak transforms; local views use strong transforms
- `MEGLabeledDataset` windows with majority-vote labels, used for k-NN eval only

## Coding Conventions

- Keep shared logic in `modules/`; experiment scripts are self-contained
- Experiment-specific model wrappers (e.g. ConditionedDINOModel) live in the experiment's `train.py`
- Hyperparameters defined as module-level constants at the top of each `train.py`
- Use `torch.multiprocessing.set_sharing_strategy("file_system")` at the top of train scripts (needed on this cluster)
- Checkpoints saved as `checkpoint_epoch{epoch:04d}.pt` with full training state
- Metrics logged to `checkpoints/metrics.json` as a list of dicts (one per epoch)
- Use `plt.colormaps["name"]` (not deprecated `plt.cm.get_cmap()`) for matplotlib colormaps
