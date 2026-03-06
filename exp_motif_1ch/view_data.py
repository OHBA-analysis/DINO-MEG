"""Visualise the simulated single-channel motif data.

Produces figures in exp_motif_1ch/figures/data/:
  motif_templates.png          – isolated waveform templates for each motif type
  annotated_timeseries.png     – raw signal with colour-coded state annotations
  psd_by_state.png             – power spectral density per motif state
  state_durations.png          – histogram of burst durations per state
  transition_matrix.png        – Markov chain transition matrix heatmap
  stationary_distribution.png  – bar chart of state proportions

Run with:
  /well/woolrich/users/wlo995/.conda/envs/torch/bin/python exp_motif_1ch/view_data.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulate_data import (
    theta_sinusoid, alpha_spindle, beta_sawtooth, sharp_wave,
    build_transition_matrix, STATES, STATE_NAMES, N_STATES,
    MOTIF_STAY_PROB, BG_STAY_PROB, BG_STATE, BG_PREFERENCE, FS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "data")
os.makedirs(OUT_DIR, exist_ok=True)

STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#cccccc"]


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Load data
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))  # (1, T)
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))  # (T,)
X_eval = np.load(os.path.join(DATA_DIR, "X_eval.npy"))
Y_eval = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))
signal = X_train[0]  # single channel
labels = Y_train


# ======================================================================
# Figure 1: Motif templates — isolated waveforms for each motif type
# ======================================================================
print("\n--- Motif templates ---")
rng = np.random.default_rng(42)
burst_len = 50  # 200 ms at 250 Hz

generators = {
    "theta_sin (6 Hz)": theta_sinusoid,
    "alpha_spindle (10 Hz)": alpha_spindle,
    "beta_sawtooth (20 Hz)": beta_sawtooth,
    "sharp_wave (broadband)": sharp_wave,
}

fig, axes = plt.subplots(len(generators), 3, figsize=(12, 2.5 * len(generators)))
fig.suptitle("Motif waveform templates (3 random instances each)", fontsize=13, y=1.02)

for row, (name, gen_fn) in enumerate(generators.items()):
    for col in range(3):
        waveform = gen_fn(burst_len, FS, np.random.default_rng(row * 10 + col))
        t = np.arange(burst_len) / FS * 1000  # ms
        ax = axes[row, col]
        ax.plot(t, waveform, color=STATE_COLORS[row], linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        if col == 0:
            ax.set_ylabel(name, fontsize=9)
        if row == len(generators) - 1:
            ax.set_xlabel("Time (ms)")
        if row == 0:
            ax.set_title(f"Instance {col + 1}", fontsize=9)
        ax.set_xlim(0, t[-1])

fig.tight_layout()
savefig(fig, "motif_templates.png")


# ======================================================================
# Figure 2: Annotated time series — raw signal with state coloring
# ======================================================================
print("\n--- Annotated time series ---")

# Show 3 segments of 2 seconds each from different parts of the signal
segment_dur = int(2.0 * FS)  # 500 samples = 2 seconds
starts = [1000, 5000, 15000]  # diverse segments

fig, axes = plt.subplots(len(starts), 1, figsize=(14, 2.5 * len(starts)))
fig.suptitle("Annotated time series (coloured by motif state)", fontsize=13, y=1.02)

for i, s0 in enumerate(starts):
    ax = axes[i]
    seg = signal[s0:s0 + segment_dur]
    seg_labels = labels[s0:s0 + segment_dur]
    t = np.arange(len(seg)) / FS

    # Plot signal in black
    ax.plot(t, seg, color="black", linewidth=0.6, alpha=0.4)

    # Overlay coloured segments per state
    for k in range(N_STATES):
        mask = seg_labels == k
        seg_colored = np.where(mask, seg, np.nan)
        ax.plot(t, seg_colored, color=STATE_COLORS[k], linewidth=0.8,
                label=STATE_NAMES[k] if i == 0 else None)

    # Add coloured background bands
    for k in range(N_STATES):
        mask = seg_labels == k
        diffs = np.diff(mask.astype(np.int8))
        burst_starts = np.where(diffs == 1)[0] + 1
        burst_ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            burst_starts = np.concatenate([[0], burst_starts])
        if mask[-1]:
            burst_ends = np.concatenate([burst_ends, [len(seg)]])
        for bs, be in zip(burst_starts, burst_ends):
            ax.axvspan(t[bs], t[min(be - 1, len(t) - 1)],
                       alpha=0.15, color=STATE_COLORS[k])

    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(f"Segment at t = {s0/FS:.1f}s", fontsize=10)

axes[0].legend(fontsize=8, loc="upper right", ncol=5)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
savefig(fig, "annotated_timeseries.png")


# ======================================================================
# Figure 3: PSD by state
# ======================================================================
print("\n--- PSD by state ---")

fig, ax = plt.subplots(figsize=(10, 5))

for k in range(N_STATES):
    mask = labels == k
    # Extract contiguous segments for this state, concatenate
    diffs = np.diff(mask.astype(np.int8))
    starts_k = np.where(diffs == 1)[0] + 1
    ends_k = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts_k = np.concatenate([[0], starts_k])
    if mask[-1]:
        ends_k = np.concatenate([ends_k, [len(signal)]])

    # Collect segments long enough for Welch
    segments = []
    for s, e in zip(starts_k, ends_k):
        if e - s >= 32:
            segments.append(signal[s:e])

    if not segments:
        continue

    # Compute PSD for each segment, average
    psds = []
    for seg in segments[:500]:  # limit for speed
        nperseg = min(64, len(seg))
        f, pxx = welch(seg, fs=FS, nperseg=nperseg)
        psds.append(pxx)

    # Interpolate to common frequency grid
    f_common = np.linspace(0, FS / 2, 128)
    psd_interp = []
    for pxx in psds:
        f_orig = np.linspace(0, FS / 2, len(pxx))
        psd_interp.append(np.interp(f_common, f_orig, pxx))
    psd_mean = np.mean(psd_interp, axis=0)

    ax.semilogy(f_common, psd_mean, color=STATE_COLORS[k],
                label=STATE_NAMES[k], linewidth=1.5)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density")
ax.set_title("PSD by motif state (averaged over bursts)")
ax.set_xlim(0, 60)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "psd_by_state.png")


# ======================================================================
# Figure 3b: PSD of full composite signal
# ======================================================================
print("\n--- PSD of full signal ---")

f_full, pxx_full = welch(signal, fs=FS, nperseg=512)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(f_full, pxx_full, color="black", linewidth=1.2, label="Full signal")

# Mark expected motif frequencies
motif_freqs = {"theta (6 Hz)": 6, "alpha (10 Hz)": 10, "beta (20 Hz)": 20}
for name, freq in motif_freqs.items():
    ax.axvline(freq, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(freq + 0.5, ax.get_ylim()[1] * 0.5, name, fontsize=8,
            rotation=90, va="top", color="gray")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density")
ax.set_title("PSD of composite signal (all states mixed)")
ax.set_xlim(0, 60)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "psd_full_signal.png")


# ======================================================================
# Figure 4: State duration histograms
# ======================================================================
print("\n--- State durations ---")

fig, axes = plt.subplots(1, N_STATES, figsize=(3.5 * N_STATES, 4), sharey=True)
fig.suptitle("Burst duration distributions", fontsize=13)

for k in range(N_STATES):
    mask = labels == k
    diffs = np.diff(mask.astype(np.int8))
    starts_k = np.where(diffs == 1)[0] + 1
    ends_k = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts_k = np.concatenate([[0], starts_k])
    if mask[-1]:
        ends_k = np.concatenate([ends_k, [len(labels)]])
    durations_ms = (ends_k - starts_k) / FS * 1000

    ax = axes[k]
    ax.hist(durations_ms, bins=50, color=STATE_COLORS[k], edgecolor="white",
            alpha=0.8, range=(0, 500))
    ax.axvline(np.mean(durations_ms), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(durations_ms):.0f}ms")
    ax.axvline(np.median(durations_ms), color="black", linestyle=":", linewidth=1.5,
               label=f"median={np.median(durations_ms):.0f}ms")
    ax.set_title(STATE_NAMES[k], fontsize=10)
    ax.set_xlabel("Duration (ms)")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Count")
fig.tight_layout()
savefig(fig, "state_durations.png")


# ======================================================================
# Figure 5: Transition matrix
# ======================================================================
print("\n--- Transition matrix ---")

trans_mat = build_transition_matrix(
    N_STATES, MOTIF_STAY_PROB, BG_STAY_PROB, BG_STATE, BG_PREFERENCE
)

fig, ax = plt.subplots(figsize=(7, 6))
# Show off-diagonal more clearly by using log scale on colorbar
off_diag = trans_mat.copy()
np.fill_diagonal(off_diag, 0)
im = ax.imshow(trans_mat, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(N_STATES))
ax.set_yticks(range(N_STATES))
ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(STATE_NAMES, fontsize=9)
ax.set_xlabel("To")
ax.set_ylabel("From")
ax.set_title("Markov chain transition matrix")

for i in range(N_STATES):
    for j in range(N_STATES):
        val = trans_mat[i, j]
        color = "white" if val > 0.5 else "black"
        txt = f"{val:.3f}" if val > 0.01 else f"{val:.4f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
savefig(fig, "transition_matrix.png")


# ======================================================================
# Figure 6: Stationary distribution (empirical from data)
# ======================================================================
print("\n--- Stationary distribution ---")

fig, ax = plt.subplots(figsize=(8, 4))
proportions = np.array([np.mean(labels == k) for k in range(N_STATES)])
bars = ax.bar(range(N_STATES), proportions * 100, color=STATE_COLORS, edgecolor="white")
ax.set_xticks(range(N_STATES))
ax.set_xticklabels(STATE_NAMES, fontsize=10)
ax.set_ylabel("Proportion (%)")
ax.set_title("Empirical state proportions (train set)")
ax.set_ylim(0, 65)
for i, v in enumerate(proportions):
    ax.text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
savefig(fig, "stationary_distribution.png")


# ======================================================================
# Figure 7: Zoomed motif examples from real data
# ======================================================================
print("\n--- Zoomed motif examples ---")

fig, axes = plt.subplots(N_STATES, 3, figsize=(14, 2.5 * N_STATES))
fig.suptitle("Example bursts from simulated data (3 per motif)", fontsize=13, y=1.01)

for k in range(N_STATES):
    mask = labels == k
    diffs = np.diff(mask.astype(np.int8))
    starts_k = np.where(diffs == 1)[0] + 1
    ends_k = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts_k = np.concatenate([[0], starts_k])
    if mask[-1]:
        ends_k = np.concatenate([ends_k, [len(signal)]])

    # Find bursts of reasonable length (15-60 samples)
    durations = ends_k - starts_k
    good = (durations >= 15) & (durations <= 60)
    good_starts = starts_k[good]
    good_ends = ends_k[good]

    # Pick 3 evenly spaced
    if len(good_starts) >= 3:
        picks = np.linspace(0, len(good_starts) - 1, 3, dtype=int)
    else:
        picks = np.arange(min(3, len(good_starts)))

    for j, pi in enumerate(picks):
        ax = axes[k, j]
        s, e = good_starts[pi], good_ends[pi]
        # Show burst with context (20 samples padding on each side)
        pad = 20
        s_ctx = max(0, s - pad)
        e_ctx = min(len(signal), e + pad)
        seg = signal[s_ctx:e_ctx]
        t = np.arange(len(seg)) / FS * 1000

        # Plot context in gray
        ax.plot(t, seg, color="gray", linewidth=0.8, alpha=0.5)
        # Overlay burst in colour
        burst_start_in_seg = s - s_ctx
        burst_end_in_seg = e - s_ctx
        t_burst = t[burst_start_in_seg:burst_end_in_seg]
        ax.plot(t_burst, seg[burst_start_in_seg:burst_end_in_seg],
                color=STATE_COLORS[k], linewidth=1.5)
        # Shade burst region
        ax.axvspan(t_burst[0], t_burst[-1], alpha=0.15, color=STATE_COLORS[k])

        dur_ms = (e - s) / FS * 1000
        ax.set_title(f"{dur_ms:.0f} ms", fontsize=8)
        ax.set_xlim(t[0], t[-1])
        if j == 0:
            ax.set_ylabel(STATE_NAMES[k], fontsize=10)
        if k == N_STATES - 1:
            ax.set_xlabel("Time (ms)")

fig.tight_layout()
savefig(fig, "zoomed_motif_examples.png")


print(f"\nAll figures saved to {OUT_DIR}")
