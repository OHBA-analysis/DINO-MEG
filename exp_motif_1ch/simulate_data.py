"""Simulate single-channel temporal motif data for DINO validation.

Signal model:
    x(t) = motif_waveform(t) + pink_noise(t)

5 states driven by a Markov chain:
  0: theta_sinusoid   — 6 Hz symmetric sinusoid, Gaussian envelope
  1: alpha_spindle     — 10 Hz sinusoid, Tukey waxing-waning envelope
  2: beta_sawtooth     — 20 Hz asymmetric sawtooth (sharp rise, slow fall)
  3: sharp_wave        — broadband non-oscillatory: brief positive peak + slow negative return
  4: background        — 1/f pink noise only

Motif states have stay_prob=0.96 (mean 25 samples = 100ms burst),
background has stay_prob=0.98 (mean 50 samples = 200ms).
Off-diagonal: motif states preferentially transition to background (bg_preference=0.6).
"""

import json
import os
import numpy as np
from scipy.signal.windows import tukey

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------
# Parameters
# ----------

FS = 250  # Hz
N_STATES = 5
PINK_EXPONENT = 1.0
MOTIF_SNR = 3.0  # default motif amplitude relative to pink noise RMS
THETA_SNR = 4.0  # higher SNR for low-frequency oscillations (easily masked by 1/f noise)
SHARP_WAVE_SNR = 5.0  # higher SNR for brief transients

# Markov chain
MOTIF_STAY_PROB = 0.96   # mean 25 samples = 100 ms
BG_STAY_PROB = 0.98      # mean 50 samples = 200 ms
BG_PREFERENCE = 0.6      # fraction of off-diagonal mass going to background
BG_STATE = 4

# State definitions
STATES = [
    ("theta_sin", 6),
    ("alpha_spindle", 10),
    ("beta_sawtooth", 20),
    ("sharp_wave", None),
    ("background", None),
]
STATE_NAMES = [s[0] for s in STATES]

# Data sizes
N_TRAIN = 500_000
N_EVAL = 50_000


def build_transition_matrix(n_states, motif_stay, bg_stay, bg_state, bg_preference):
    """Build asymmetric transition matrix where motif states prefer background.

    For motif states: off-diagonal mass split as bg_preference to background,
    (1 - bg_preference) uniform among other motif states.
    For background: off-diagonal mass uniform among motif states.
    """
    T = np.zeros((n_states, n_states))

    motif_states = [i for i in range(n_states) if i != bg_state]
    n_motif = len(motif_states)

    for i in range(n_states):
        if i == bg_state:
            # Background → uniform over motif states
            off_diag = 1.0 - bg_stay
            for j in motif_states:
                T[i, j] = off_diag / n_motif
            T[i, i] = bg_stay
        else:
            # Motif → prefer background, rest uniform over other motifs
            off_diag = 1.0 - motif_stay
            T[i, bg_state] = off_diag * bg_preference
            other_motifs = [j for j in motif_states if j != i]
            if other_motifs:
                per_other = off_diag * (1.0 - bg_preference) / len(other_motifs)
                for j in other_motifs:
                    T[i, j] = per_other
            T[i, i] = motif_stay

    return T


def simulate_markov_chain(n_samples, trans_mat, rng):
    """Simulate a Markov chain from a full transition matrix."""
    n_states = trans_mat.shape[0]
    states = np.empty(n_samples, dtype=np.int32)
    states[0] = rng.integers(n_states)
    for t in range(1, n_samples):
        states[t] = rng.choice(n_states, p=trans_mat[states[t - 1]])
    return states


def generate_pink_noise(n_samples, exponent, rng):
    """Generate 1/f noise for a single channel, unit RMS."""
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)

    amplitude = np.ones_like(freqs)
    amplitude[1:] = freqs[1:] ** (-exponent / 2)

    white = rng.standard_normal(n_fft)
    spectrum = np.fft.rfft(white)
    spectrum *= amplitude
    shaped = np.fft.irfft(spectrum, n=n_fft)[:n_samples]

    rms = np.sqrt(np.mean(shaped ** 2))
    if rms > 0:
        shaped /= rms
    return shaped


def theta_sinusoid(duration, fs, rng):
    """6 Hz sinusoid with Gaussian envelope, random phase. Returns unit-RMS waveform."""
    freq = 6.0
    t = np.arange(duration) / fs
    phase = rng.uniform(0, 2 * np.pi)
    osc = np.sin(2 * np.pi * freq * t + phase)

    # Gaussian envelope centered on burst, sigma = duration/5
    center = (duration - 1) / 2.0
    sigma = duration / 5.0
    envelope = np.exp(-0.5 * ((np.arange(duration) - center) / sigma) ** 2)

    waveform = osc * envelope
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        waveform /= rms
    return waveform


def alpha_spindle(duration, fs, rng):
    """10 Hz sinusoid with Tukey (waxing-waning) envelope, random phase. Returns unit-RMS waveform."""
    freq = 10.0
    t = np.arange(duration) / fs
    phase = rng.uniform(0, 2 * np.pi)
    osc = np.sin(2 * np.pi * freq * t + phase)

    # Tukey window with alpha=0.5 (25% cosine taper on each side)
    envelope = tukey(duration, alpha=0.5)

    waveform = osc * envelope
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        waveform /= rms
    return waveform


def beta_sawtooth(duration, fs, rng):
    """20 Hz asymmetric sawtooth (sharp rise, slow fall), random phase. Returns unit-RMS waveform."""
    freq = 20.0
    t = np.arange(duration) / fs
    phase = rng.uniform(0, 2 * np.pi)

    # Sawtooth via Fourier: sum of harmonics with alternating sign
    # Sharp rise, slow fall: use phase-shifted sawtooth
    cycle_phase = (freq * t + phase / (2 * np.pi)) % 1.0
    # Asymmetric: fast rise (20% of cycle), slow fall (80%)
    rise_frac = 0.2
    waveform = np.where(
        cycle_phase < rise_frac,
        cycle_phase / rise_frac,
        1.0 - (cycle_phase - rise_frac) / (1.0 - rise_frac),
    )
    # Center at zero
    waveform = waveform - waveform.mean()

    # Apply Gaussian envelope to make it burst-like
    center = (duration - 1) / 2.0
    sigma = duration / 5.0
    envelope = np.exp(-0.5 * ((np.arange(duration) - center) / sigma) ** 2)
    waveform = waveform * envelope

    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        waveform /= rms
    return waveform


def sharp_wave(duration, fs, rng):
    """Non-oscillatory sharp wave: sharp positive peak + weak negative return + ripple.

    Always positive-first (polarity invariance learned via augmentation instead).
    More distinctive than v1: narrower peak, weaker return, post-peak ripple.
    Returns unit-RMS waveform.
    """
    t = np.arange(duration, dtype=np.float64)

    # Random position for peak within central 60% of burst
    margin = int(duration * 0.2)
    peak_pos = rng.integers(margin, max(margin + 1, duration - margin))

    # Sharp positive peak: very narrow Gaussian (sigma ~ 1.5 samples = 6 ms)
    sigma_pos = max(1.5, duration / 16.0)
    pos_peak = np.exp(-0.5 * ((t - peak_pos) / sigma_pos) ** 2)

    # Weak negative return (0.3x amplitude, wider)
    sigma_neg = sigma_pos * 3.0
    neg_center = peak_pos + sigma_pos * 2.5
    neg_peak = -0.3 * np.exp(-0.5 * ((t - neg_center) / sigma_neg) ** 2)

    # Post-peak high-frequency ripple (~30-40 Hz, 2-3 cycles)
    ripple_freq = rng.uniform(30.0, 40.0)
    ripple_env_sigma = sigma_pos * 2.0
    ripple_env = 0.25 * np.exp(-0.5 * ((t - peak_pos - sigma_pos * 1.5) / ripple_env_sigma) ** 2)
    ripple = ripple_env * np.sin(2 * np.pi * ripple_freq * (t - peak_pos) / fs)

    waveform = pos_peak + neg_peak + ripple

    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        waveform /= rms
    return waveform


# Map state index to waveform generator
WAVEFORM_GENERATORS = {
    0: theta_sinusoid,
    1: alpha_spindle,
    2: beta_sawtooth,
    3: sharp_wave,
}


def simulate_data(n_samples, seed):
    """Simulate single-channel switching motif data.

    Returns:
        X: (1, n_samples) float32 — z-scored signal
        Y: (n_samples,) int32 — ground truth state labels
    """
    rng = np.random.default_rng(seed)

    # 1. Build transition matrix and simulate Markov chain
    trans_mat = build_transition_matrix(
        N_STATES, MOTIF_STAY_PROB, BG_STAY_PROB, BG_STATE, BG_PREFERENCE
    )
    states = simulate_markov_chain(n_samples, trans_mat, rng)

    # 2. Background: 1/f pink noise (unit RMS)
    pink = generate_pink_noise(n_samples, PINK_EXPONENT, rng)

    # 3. Generate motif waveforms per burst
    signal = np.zeros(n_samples, dtype=np.float64)

    # Per-state SNR: theta and sharp waves get higher amplitude
    state_snr = {k: MOTIF_SNR for k in WAVEFORM_GENERATORS}
    state_snr[0] = THETA_SNR  # theta_sin: low freq easily masked by 1/f noise
    state_snr[3] = SHARP_WAVE_SNR  # sharp_wave: brief transients

    for k, gen_fn in WAVEFORM_GENERATORS.items():
        activation = states == k
        diffs = np.diff(activation.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if activation[0]:
            starts = np.concatenate([[0], starts])
        if activation[-1]:
            ends = np.concatenate([ends, [n_samples]])

        snr = state_snr[k]
        for s, e in zip(starts, ends):
            burst_len = e - s
            if burst_len < 2:
                continue
            waveform = gen_fn(burst_len, FS, rng)
            signal[s:e] = snr * waveform

    # 4. Compose: pink noise + motif signal
    X = pink + signal

    # 5. Z-score
    mean = X.mean()
    std = X.std()
    if std > 0:
        X = (X - mean) / std

    # Reshape to (1, T) for single channel
    X = X.reshape(1, -1).astype(np.float32)

    return X, states


def print_state_statistics(states, label):
    """Print state proportions and burst duration statistics."""
    n = len(states)
    print(f"\n{label}: {n} samples ({n / FS:.0f} s)")
    print(f"  {'State':<18} {'Count':>8} {'Proportion':>10} {'Mean dur (ms)':>14} {'Median dur (ms)':>16}")
    print("  " + "-" * 70)

    for k, (name, _) in enumerate(STATES):
        mask = states == k
        count = mask.sum()
        prop = count / n

        diffs = np.diff(mask.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
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

        print(f"  {name:<18} {count:>8d} {prop:>10.3f} {mean_dur:>14.1f} {median_dur:>16.1f}")


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

    # Print transition matrix for reference
    trans_mat = build_transition_matrix(
        N_STATES, MOTIF_STAY_PROB, BG_STAY_PROB, BG_STATE, BG_PREFERENCE
    )
    print("\nTransition matrix:")
    print("  " + "  ".join(f"{STATE_NAMES[i]:>14}" for i in range(N_STATES)))
    for i in range(N_STATES):
        row = "  ".join(f"{trans_mat[i, j]:>14.4f}" for j in range(N_STATES))
        print(f"  {STATE_NAMES[i]:<14} {row}")

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    print("\nStationary distribution:")
    for i, name in enumerate(STATE_NAMES):
        print(f"  {name:<18}: {pi[i]:.3f}")

    # Save parameters for reference
    params = {
        "fs": FS,
        "n_states": N_STATES,
        "motif_stay_prob": MOTIF_STAY_PROB,
        "bg_stay_prob": BG_STAY_PROB,
        "bg_preference": BG_PREFERENCE,
        "motif_snr": MOTIF_SNR,
        "theta_snr": THETA_SNR,
        "sharp_wave_snr": SHARP_WAVE_SNR,
        "pink_exponent": PINK_EXPONENT,
        "states": [(name, freq) for name, freq in STATES],
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
    }
    params_path = os.path.join(DATA_DIR, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParameters saved to {params_path}")
    print("Done.")
