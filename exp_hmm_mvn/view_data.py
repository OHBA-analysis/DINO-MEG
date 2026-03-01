"""Diagnostic visualisations for the switching oscillatory burst simulation.

Generates 6 figures in exp_hmm_mvn/figures/:
  1. raw_timeseries.png     — 3 channels over ~3 s with state-coloured background
  2. power_spectral_density.png — log-log PSD with 1/f reference and spectral peaks
  3. spectrogram.png        — time-frequency plot (~10 s) with state labels
  4. state_durations.png    — histogram of burst durations per state
  5. per_state_psd.png      — average PSD conditioned on each state
  6. static_power_maps.png  — time-averaged band power per parcel on glass-brain layout
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import welch, spectrogram as sp_spectrogram, butter, sosfiltfilt
import nibabel as nib
from nilearn import plotting as ni_plotting

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
ATLAS_PATH = os.path.join(
    os.path.dirname(__file__),
    "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
)
os.makedirs(FIG_DIR, exist_ok=True)

FS = 250
STATE_NAMES = ["theta", "alpha", "beta", "low_gamma", "background"]
STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#cccccc"]
STATE_CENTERS = [6, 10, 20, 35, None]

print("Loading data...")
X = np.load(os.path.join(DATA_DIR, "X_eval.npy"))  # (C, T)
Y = np.load(os.path.join(DATA_DIR, "Y_eval.npy"))  # (T,)
print(f"  X: {X.shape}, Y: {Y.shape}")


# --- 1. Raw timeseries with state-coloured background ---

def plot_raw_timeseries():
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    t_start = 0
    t_end = int(3 * FS)  # 3 seconds
    t = np.arange(t_start, t_end) / FS

    channels = [0, N_CHANNELS // 4, N_CHANNELS // 2]
    for ax, ch in zip(axes, channels):
        ax.plot(t, X[ch, t_start:t_end], linewidth=0.5, color="black")
        ax.set_ylabel(f"Ch {ch}")

        # Colour background by state
        states_seg = Y[t_start:t_end]
        prev_state = states_seg[0]
        seg_start = 0
        for i in range(1, len(states_seg)):
            if states_seg[i] != prev_state or i == len(states_seg) - 1:
                ax.axvspan(
                    t[seg_start], t[min(i, len(t) - 1)],
                    alpha=0.2, color=STATE_COLORS[prev_state], linewidth=0,
                )
                seg_start = i
                prev_state = states_seg[i]

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Raw timeseries with state labels")

    legend_patches = [Patch(color=c, alpha=0.3, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=5)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "raw_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# --- 2. Power spectral density ---

def plot_psd():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Average PSD across channels
    freqs, psd = welch(X, fs=FS, nperseg=1024, axis=1)
    psd_mean = psd.mean(axis=0)

    ax.loglog(freqs[1:], psd_mean[1:], color="black", linewidth=1, label="Mean PSD")

    # 1/f reference line
    ref = psd_mean[5] * (freqs[5] / freqs[1:]) ** 1.0
    ax.loglog(freqs[1:], ref, "--", color="gray", alpha=0.6, label="1/f reference")

    # Mark expected spectral peaks
    for name, center, color in zip(STATE_NAMES, STATE_CENTERS, STATE_COLORS):
        if center is not None:
            ax.axvline(center, color=color, linestyle=":", alpha=0.7, label=f"{name} ({center} Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power spectral density (channel average)")
    ax.legend(fontsize=7)
    ax.set_xlim(1, FS / 2)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "power_spectral_density.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# --- 3. Spectrogram ---

def plot_spectrogram():
    fig, (ax_spec, ax_state) = plt.subplots(
        2, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [4, 1]}, sharex=True,
    )

    # Use ~10 s window, channel 0
    t_end = min(int(10 * FS), X.shape[1])
    signal = X[0, :t_end]

    freqs, times, Sxx = sp_spectrogram(signal, fs=FS, nperseg=128, noverlap=120)
    ax_spec.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="viridis")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(0, 60)
    ax_spec.set_title("Spectrogram (channel 0, ~10 s)")

    # State labels along x-axis
    t_axis = np.arange(t_end) / FS
    states_seg = Y[:t_end]
    for k in range(len(STATE_NAMES)):
        mask = states_seg == k
        ax_state.fill_between(
            t_axis, 0, 1, where=mask,
            color=STATE_COLORS[k], alpha=0.6, linewidth=0,
        )
    ax_state.set_yticks([])
    ax_state.set_xlabel("Time (s)")
    ax_state.set_ylabel("State")

    legend_patches = [Patch(color=c, alpha=0.6, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)]
    ax_state.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=5)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "spectrogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# --- 4. State duration histograms ---

def plot_state_durations():
    fig, axes = plt.subplots(1, len(STATE_NAMES), figsize=(16, 3), sharey=True)

    for k, (ax, name, color) in enumerate(zip(axes, STATE_NAMES, STATE_COLORS)):
        mask = (Y == k).astype(np.int8)
        diffs = np.diff(mask)
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(Y)]])
        durations_ms = (ends - starts) / FS * 1000

        if len(durations_ms) > 0:
            ax.hist(durations_ms, bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(np.mean(durations_ms), color="red", linestyle="--", linewidth=1,
                       label=f"mean={np.mean(durations_ms):.0f} ms")
            ax.axvline(np.median(durations_ms), color="blue", linestyle="--", linewidth=1,
                       label=f"median={np.median(durations_ms):.0f} ms")
            ax.legend(fontsize=6)

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Duration (ms)")

    axes[0].set_ylabel("Count")
    fig.suptitle("Burst duration distributions", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "state_durations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# --- 5. Per-state PSD ---

def plot_per_state_psd():
    fig, ax = plt.subplots(figsize=(8, 5))

    nperseg = 256
    for k, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
        mask = Y == k
        # Gather all contiguous segments for this state
        diffs = np.diff(mask.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(Y)]])

        # Collect segments long enough for PSD estimation
        psds = []
        for s, e in zip(starts, ends):
            if e - s >= nperseg:
                seg = X[:, s:e]  # (C, seg_len)
                freqs, psd = welch(seg, fs=FS, nperseg=nperseg, axis=1)
                psds.append(psd.mean(axis=0))  # average across channels

        if psds:
            psd_mean = np.mean(psds, axis=0)
            ax.semilogy(freqs, psd_mean, color=color, linewidth=1.5, label=name)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Per-state power spectral density (channel average)")
    ax.legend()
    ax.set_xlim(0, 60)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "per_state_psd.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# --- 6. Static power maps (nilearn glass-brain) ---

BAND_RANGES = {
    "theta":     (4, 8),
    "alpha":     (7.5, 12.5),
    "beta":      (15, 25),
    "low_gamma": (28, 42),
}


def _parcel_values_to_nifti(values, atlas_img):
    """Map a length-52 vector of parcel values into a 3D NIfTI stat image."""
    atlas_data = atlas_img.get_fdata()  # (x, y, z, 52)
    vol = np.zeros(atlas_data.shape[:3])
    for i, v in enumerate(values):
        mask = atlas_data[..., i] > 0
        vol[mask] = v
    return nib.Nifti1Image(vol, atlas_img.affine)


def plot_static_power_maps():
    """Plot time-averaged band power per parcel as nilearn glass-brain maps."""
    atlas_img = nib.load(ATLAS_PATH)

    # Compute PSD per channel
    freqs, psd = welch(X, fs=FS, nperseg=1024, axis=1)  # (C, F)
    freq_res = freqs[1] - freqs[0]

    n_bands = len(BAND_RANGES)
    fig, axes = plt.subplots(n_bands, 1, figsize=(10, 3 * n_bands))

    for j, (band_name, (flo, fhi)) in enumerate(BAND_RANGES.items()):
        freq_mask = (freqs >= flo) & (freqs <= fhi)
        power = psd[:, freq_mask].sum(axis=1) * freq_res  # (52,)

        stat_img = _parcel_values_to_nifti(power, atlas_img)

        ni_plotting.plot_glass_brain(
            stat_img,
            display_mode="lyrz",
            colorbar=True,
            cmap="Reds",
            title=f"{band_name} ({flo}-{fhi} Hz)",
            axes=axes[j],
            plot_abs=False,
        )

    fig.suptitle("Static band power per parcel (Glasser52)", fontsize=12, y=1.01)
    path = os.path.join(FIG_DIR, "static_power_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# --- Run all ---

N_CHANNELS = X.shape[0]

print("\nGenerating figures...")
plot_raw_timeseries()
plot_psd()
plot_spectrogram()
plot_state_durations()
plot_per_state_psd()
plot_static_power_maps()
print(f"\nAll figures saved to {FIG_DIR}")
