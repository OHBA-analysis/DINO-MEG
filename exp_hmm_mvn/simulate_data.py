"""Simulate switching oscillatory burst data for DINO validation.

Signal model:
    x(t) = sum_k  alpha_k(t) * A_k * s_k(t) * amp_k  +  pink(t)  +  white(t)

States: theta (4-8 Hz), alpha (7.5-12.5 Hz), beta (15-25 Hz),
        low-gamma (28-42 Hz), background (no oscillation).

Markov chain with stay_prob=0.992 gives ~500 ms mean burst duration at 250 Hz.
"""

import json
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------
# Parameters
# ----------

N_CHANNELS = 52
FS = 250  # Hz
N_STATES = 5
STAY_PROB = 0.96  # mean burst duration = 1/(1-p) = 25 samples = 100 ms
PINK_EXPONENT = 1.0
WHITE_NOISE_STD = 0.05  # small sensor noise
BUTTER_ORDER = 4

# State definitions: (name, center_freq, bandwidth, amplitude)
# Amplitude = oscillation RMS relative to pink noise RMS
# Mimics real MEG: alpha strongest, gamma weakest
STATES = [
    ("theta", 6, 2, 1.5),       # 4–8 Hz, moderate
    ("alpha", 10, 2.5, 2.0),    # 7.5–12.5 Hz, strongest
    ("beta", 20, 5, 1.0),       # 15–25 Hz, weaker
    ("low_gamma", 35, 7, 0.7),  # 28–42 Hz, weakest
    ("background", None, None, 0.0),
]

# Data sizes
TRAIN_MINUTES = 120
EVAL_MINUTES = 10
N_TRAIN = FS * 60 * TRAIN_MINUTES  # 1 800 000
N_EVAL = FS * 60 * EVAL_MINUTES    # 150 000


def simulate_markov_chain(n_samples, n_states, stay_prob, rng):
    """Simulate a homogeneous Markov chain with uniform off-diagonal transitions."""
    trans_mat = np.full((n_states, n_states), (1 - stay_prob) / (n_states - 1))
    np.fill_diagonal(trans_mat, stay_prob)

    states = np.empty(n_samples, dtype=np.int32)
    states[0] = rng.integers(n_states)
    for t in range(1, n_samples):
        states[t] = rng.choice(n_states, p=trans_mat[states[t - 1]])
    return states


def generate_pink_noise(n_channels, n_samples, exponent, rng):
    """Generate 1/f noise via frequency-domain shaping. Returns (C, T), unit RMS per channel."""
    # Use next power of 2 for FFT efficiency
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)

    # 1/f^(exponent/2) amplitude shaping (power ~ 1/f^exponent)
    amplitude = np.ones_like(freqs)
    amplitude[1:] = freqs[1:] ** (-exponent / 2)

    noise = np.empty((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        white = rng.standard_normal(n_fft)
        spectrum = np.fft.rfft(white)
        spectrum *= amplitude
        shaped = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
        # Normalise to unit RMS
        rms = np.sqrt(np.mean(shaped ** 2))
        if rms > 0:
            shaped /= rms
        noise[ch] = shaped
    return noise


def generate_oscillatory_source(n_samples, center_freq, bandwidth, rng):
    """Generate a narrowband oscillatory source via bandpass-filtered white noise.

    Returns a 1-D array of length n_samples with unit RMS.
    Uses padding to avoid filter edge artifacts.
    """
    low = center_freq - bandwidth / 2
    high = center_freq + bandwidth / 2
    nyq = FS / 2
    sos = butter(BUTTER_ORDER, [low / nyq, high / nyq], btype="band", output="sos")

    # Pad to avoid edge effects (3x the filter's transient response)
    pad_len = int(3 * FS / low)  # ~3 cycles of the lowest frequency
    total_len = n_samples + 2 * pad_len

    white = rng.standard_normal(total_len)
    filtered = sosfiltfilt(sos, white)
    source = filtered[pad_len: pad_len + n_samples]

    # Normalise to unit RMS
    rms = np.sqrt(np.mean(source ** 2))
    if rms > 0:
        source /= rms
    return source


# Glasser52 parcellation: 26 bilateral regions, right hemisphere first (0-25),
# left hemisphere mirrored (26-51).  Index within each hemisphere:
#   0  Primary and Early Visual Cortex
#   1  Dorsal Stream Visual Cortex
#   2  Ventral Stream Visual Cortex
#   3  MT+ Complex and Neighboring Visual Areas
#   4  Superior Somatosensory and Motor Cortex
#   5  Inferior Somatosensory and Motor Cortex
#   6  Supplementary Motor Area
#   7  Cingulate Motor Areas & Area 5
#   8  Premotor Cortex
#   9  Insular & Frontoparietal Operculum
#  10  Early Auditory Cortex
#  11  Auditory Association Cortex
#  12  Medial Temporal Cortex
#  13  Lateral Temporal Cortex
#  14  Temporal-Parieto-Occipital Junction
#  15  Medial Bank of the Intra-parietal Sulcus
#  16  Superior Medial Parietal Cortex
#  17  Inferior Parietal Cortex Task-Positive Network
#  18  Inferior Parietal Cortex Task-Negative Network
#  19  Intraparietal Sulcus & PGP
#  20  Posterior Cingulate Cortex
#  21  Anterior Cingulate and Medial Prefrontal Cortex
#  22  Orbital and Polar Frontal Cortex
#  23  Inferior Frontal Cortex
#  24  Inferior Dorso-Lateral Prefrontal Cortex
#  25  Superior Dorso-Lateral Prefrontal Cortex

# Anatomical spatial weights for each oscillatory state.
# Keys are within-hemisphere parcel indices (applied bilaterally).
# Values: 1.0 = primary region, 0.5 = moderate involvement, 0.2 = weak.
SPATIAL_WEIGHTS = {
    # Theta (4-8 Hz): frontal midline — ACC/mPFC, orbital frontal, DLPFC
    "theta": {21: 1.0, 22: 1.0, 25: 0.5, 24: 0.5, 23: 0.5, 20: 0.2},
    # Alpha (7.5-12.5 Hz): occipital — primary visual, dorsal/ventral stream
    "alpha": {0: 1.0, 1: 1.0, 2: 0.5, 20: 0.5, 3: 0.2, 19: 0.2},
    # Beta (15-25 Hz): sensorimotor — somatosensory/motor, SMA, premotor
    "beta": {4: 1.0, 5: 1.0, 6: 0.5, 8: 0.5, 7: 0.2},
    # Low gamma (28-42 Hz): auditory/temporal — auditory cortex, lateral temporal, TPO
    "low_gamma": {10: 1.0, 11: 1.0, 13: 0.5, 14: 0.5, 9: 0.2, 12: 0.2},
}
SPATIAL_NOISE = 0.05  # small random weight on all channels for realism


def generate_spatial_patterns(n_channels, n_states, rng):
    """Generate anatomically informed spatial mixing vectors for each state.

    Uses the Glasser52 parcellation layout (26 bilateral regions) with
    weights from SPATIAL_WEIGHTS.  A small amount of random noise is added
    to all channels so patterns are not perfectly sparse.
    """
    assert n_channels == 52, "Spatial patterns assume 52 Glasser parcels"
    patterns = np.zeros((n_states, n_channels), dtype=np.float64)

    for k, (name, center, bw, amp) in enumerate(STATES):
        if center is None:
            # Background state: no spatial pattern needed (no oscillation)
            continue
        weights = SPATIAL_WEIGHTS[name]
        for hemi_idx, w in weights.items():
            patterns[k, hemi_idx] = w       # right hemisphere (0-25)
            patterns[k, hemi_idx + 26] = w  # left hemisphere (26-51)
        # Add small random noise to all channels
        patterns[k] += SPATIAL_NOISE * rng.standard_normal(n_channels)
        # Normalise to unit norm
        patterns[k] /= np.linalg.norm(patterns[k])

    return patterns


def simulate_data(n_samples, seed):
    """Simulate switching oscillatory burst MEG data.

    Returns:
        X: (n_channels, n_samples) float32 — z-scored MEG data
        Y: (n_samples,) int32 — ground truth state labels
    """
    rng = np.random.default_rng(seed)

    # 1. Markov chain state sequence
    states = simulate_markov_chain(n_samples, N_STATES, STAY_PROB, rng)

    # 2. Background: 1/f pink noise (unit RMS per channel)
    pink = generate_pink_noise(N_CHANNELS, n_samples, PINK_EXPONENT, rng)

    # 3. Spatial patterns for each state
    patterns = generate_spatial_patterns(N_CHANNELS, N_STATES, rng)

    # 4. Generate oscillatory sources with random phase per burst
    #    Each burst samples from a random offset in a long source, giving
    #    independent initial phase per activation (more realistic than gating
    #    a single continuous oscillation).
    signal = np.zeros((N_CHANNELS, n_samples), dtype=np.float64)
    for k, (name, center, bw, amp) in enumerate(STATES):
        if center is None:
            continue  # background state — no oscillation

        source = generate_oscillatory_source(n_samples + 1000, center, bw, rng)
        source_len = len(source)

        # Find burst boundaries (contiguous runs of state k)
        activation = (states == k)
        diffs = np.diff(activation.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if activation[0]:
            starts = np.concatenate([[0], starts])
        if activation[-1]:
            ends = np.concatenate([ends, [n_samples]])

        # Each burst samples from a random position → independent phase
        gated = np.zeros(n_samples, dtype=np.float64)
        for s, e in zip(starts, ends):
            burst_len = e - s
            offset = rng.integers(0, source_len - burst_len + 1)
            gated[s:e] = source[offset:offset + burst_len]

        signal += amp * np.outer(patterns[k], gated)

    # 5. Compose: pink noise + oscillatory signal + white sensor noise
    white = WHITE_NOISE_STD * rng.standard_normal((N_CHANNELS, n_samples))
    X = pink + signal + white

    # 6. Z-score per channel
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    return X.astype(np.float32), states


def print_state_statistics(states, label):
    """Print state proportions and burst duration statistics."""
    n = len(states)
    print(f"\n{label}: {n} samples ({n / FS:.0f} s)")
    print(f"  {'State':<15} {'Count':>8} {'Proportion':>10} {'Mean dur (ms)':>14} {'Median dur (ms)':>16}")
    print("  " + "-" * 65)

    for k, (name, _, _, _) in enumerate(STATES):
        mask = states == k
        count = mask.sum()
        prop = count / n

        # Compute burst durations
        diffs = np.diff(mask.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [n]])
        durations = ends - starts
        if len(durations) > 0:
            mean_dur = np.mean(durations) / FS * 1000
            median_dur = np.median(durations) / FS * 1000
        else:
            mean_dur = median_dur = 0.0

        print(f"  {name:<15} {count:>8d} {prop:>10.3f} {mean_dur:>14.1f} {median_dur:>16.1f}")


if __name__ == "__main__":
    for split, seed, n_samples in [
        ("train", 0, N_TRAIN),
        ("eval", 1, N_EVAL),
    ]:
        print(f"Simulating {split} data (seed={seed})...")
        X, Y = simulate_data(n_samples, seed)
        np.save(os.path.join(DATA_DIR, f"X_{split}.npy"), X)
        np.save(os.path.join(DATA_DIR, f"Y_{split}.npy"), Y)
        print(f"  X_{split}.npy: {X.shape}, dtype={X.dtype}")
        print(f"  Y_{split}.npy: {Y.shape}, dtype={Y.dtype}")
        print_state_statistics(Y, split)

    # Save parameters for reference
    params = {
        "n_channels": N_CHANNELS,
        "fs": FS,
        "n_states": N_STATES,
        "stay_prob": STAY_PROB,
        "state_amplitudes": {name: amp for name, _, _, amp in STATES},
        "pink_exponent": PINK_EXPONENT,
        "white_noise_std": WHITE_NOISE_STD,
        "butter_order": BUTTER_ORDER,
        "states": [(name, center, bw, amp) for name, center, bw, amp in STATES],
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
    }
    params_path = os.path.join(DATA_DIR, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParameters saved to {params_path}")
    print("Done.")
