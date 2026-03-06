"""Data."""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_amplitude_envelopes(X, fs, freqs, n_jobs=4,
                                log_transform=False, standardize=False):
    """Morlet wavelet amplitude envelopes.

    Parameters
    ----------
    X : ndarray, shape (C, T)
        Multichannel time-series.
    fs : int
        Sampling frequency in Hz.
    freqs : array-like of float
        Center frequencies for Morlet wavelets.
    n_jobs : int
        Number of parallel jobs for MNE.
    log_transform : bool
        If True, apply log(amplitude + 1e-6) to compress dynamic range.
    standardize : bool
        If True, z-score each (channel, freq) pair over time.

    Returns
    -------
    envelopes : ndarray, shape (C, n_freqs, T), float32
    """
    from mne.time_frequency import tfr_array_morlet

    freqs = np.asarray(freqs, dtype=float)
    n_cycles = np.full_like(freqs, 3.0)

    # tfr_array_morlet expects (n_epochs, n_channels, n_times)
    power = tfr_array_morlet(
        X[np.newaxis],
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        use_fft=True,
        zero_mean=True,
        n_jobs=n_jobs,
    )  # (1, C, n_freqs, T)

    amplitude = np.sqrt(power[0])  # (C, n_freqs, T)

    if log_transform:
        amplitude = np.log(amplitude + 1e-6)

    if standardize:
        mean = amplitude.mean(axis=-1, keepdims=True)  # (C, n_freqs, 1)
        std = amplitude.std(axis=-1, keepdims=True)
        std[std == 0] = 1.0
        amplitude = (amplitude - mean) / std

    return amplitude.astype(np.float32)


def _rand_choice(p):
    return random.random() < p


def _fft_notch(x, fs, freq, bandwidth=1.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    X = np.fft.rfft(x)
    mask = ~((freqs >= (freq - bandwidth / 2.0)) & (freqs <= (freq + bandwidth / 2.0)))
    X = X * mask
    return np.fft.irfft(X, n=N)


def _time_warp(arr, max_warp=0.12):
    """Time warp via interpolation. Only supports 2D (C, T) arrays."""
    if arr.ndim != 2:
        return arr
    C, T = arr.shape
    scale = 1.0 + np.random.uniform(-max_warp, max_warp)
    old_t = np.arange(T)
    new_t = np.linspace(0, T - 1, int(round(T * scale))).astype(np.float32)
    warped = np.zeros((C, T), dtype=arr.dtype)
    for c in range(C):
        vals = np.interp(new_t, old_t, arr[c])
        warped[c] = np.interp(
            np.linspace(0, len(new_t) - 1, T), np.arange(len(new_t)), vals
        )
    return warped


def _time_mask(arr, mask_ratio=0.05, n_masks=1):
    """Mask random time segments. Works with (..., T) arrays."""
    T = arr.shape[-1]
    out = arr.copy()
    L = int(T * mask_ratio)
    for _ in range(n_masks):
        if L <= 0:
            break
        start = random.randint(0, max(0, T - L))
        out[..., start : start + L] = 0.0
    return out


def _channel_mask(arr, mask_p=0.15):
    """Mask random channels (first axis). Works with (C, ...) arrays."""
    C = arr.shape[0]
    mask = (np.random.rand(C) >= mask_p).astype(arr.dtype)
    shape = (C,) + (1,) * (arr.ndim - 1)
    return arr * mask.reshape(shape)


def _amplitude_scaling(arr, sigma=0.1):
    """Scale channels randomly. Works with (C, ...) arrays."""
    C = arr.shape[0]
    scales = np.random.normal(1.0, sigma, size=(C,))
    shape = (C,) + (1,) * (arr.ndim - 1)
    return arr * scales.reshape(shape)


def _add_gaussian_noise(arr, std):
    if std <= 0:
        return arr
    noise = np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype)
    return arr + noise


def _freq_mask(arr, n_masks=1, max_width=2):
    """Mask random frequency bands. Only for 3D (C, F, T) arrays."""
    if arr.ndim != 3:
        return arr
    out = arr.copy()
    n_freqs = arr.shape[1]
    for _ in range(n_masks):
        width = random.randint(1, max_width)
        start = random.randint(0, max(0, n_freqs - width))
        out[:, start:start + width, :] = 0.0
    return out


def _baseline_shift(arr, shift_std=1e-3):
    """Shift channels by random constant. Works with (C, ...) arrays."""
    C = arr.shape[0]
    shifts = np.random.normal(0.0, shift_std, size=(C,))
    shape = (C,) + (1,) * (arr.ndim - 1)
    return arr + shifts.reshape(shape)


def _random_crop(arr, crop_length):
    """Random crop along last axis. Works with (..., T) arrays."""
    T = arr.shape[-1]
    if T <= crop_length:
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, crop_length - T)]
        out = np.pad(arr, pad_width, mode="constant")
        return out[..., :crop_length].astype(arr.dtype)
    start = random.randint(0, T - crop_length)
    return arr[..., start : start + crop_length].astype(arr.dtype)


class Transforms:
    def __init__(
        self,
        sampling_frequency=None,
        add_noise_p=0.9,
        noise_std=1e-3,
        baseline_shift_p=0.2,
        baseline_shift_std=5e-4,
        scale_p=0.5,
        scale_sigma=0.08,
        amplitude_warp_p=0.5,
        warp_max=0.12,
        time_mask_p=0.6,
        time_mask_ratio=0.05,
        time_mask_n=1,
        channel_mask_p=0.5,
        channel_dropout_p=0.15,
        notch_p=0.9,
        notch_freqs=None,
        notch_bandwidth=1.0,
        freq_mask_p=0.0,
        freq_mask_n=1,
        freq_mask_max_width=2,
        time_reverse_p=0.05,
        sign_flip_p=0.0,
        random_crop_len=None,
    ):
        self.sampling_frequency = sampling_frequency
        self.add_noise_p = add_noise_p
        self.noise_std = noise_std
        self.baseline_shift_p = baseline_shift_p
        self.baseline_shift_std = baseline_shift_std
        self.scale_p = scale_p
        self.scale_sigma = scale_sigma
        self.amplitude_warp_p = amplitude_warp_p
        self.warp_max = warp_max
        self.time_mask_p = time_mask_p
        self.time_mask_ratio = time_mask_ratio
        self.time_mask_n = time_mask_n
        self.channel_mask_p = channel_mask_p
        self.channel_dropout_p = channel_dropout_p
        self.notch_p = notch_p
        self.notch_freqs = notch_freqs or []
        self.notch_bandwidth = notch_bandwidth
        self.freq_mask_p = freq_mask_p
        self.freq_mask_n = freq_mask_n
        self.freq_mask_max_width = freq_mask_max_width
        self.time_reverse_p = time_reverse_p
        self.sign_flip_p = sign_flip_p
        self.random_crop_len = random_crop_len

    def __call__(self, arr):
        out = arr.copy().astype(np.float32)
        if self.random_crop_len is not None:
            out = _random_crop(out, self.random_crop_len)
        if _rand_choice(self.add_noise_p):
            out = _add_gaussian_noise(out, self.noise_std)
        if _rand_choice(self.baseline_shift_p):
            out = _baseline_shift(out, shift_std=self.baseline_shift_std)
        if _rand_choice(self.scale_p):
            out = _amplitude_scaling(out, sigma=self.scale_sigma)
        if _rand_choice(self.amplitude_warp_p):
            out = _time_warp(out, max_warp=self.warp_max)
        if _rand_choice(self.time_mask_p):
            out = _time_mask(
                out, mask_ratio=self.time_mask_ratio, n_masks=self.time_mask_n
            )
        if _rand_choice(self.channel_mask_p):
            out = _channel_mask(out, mask_p=self.channel_dropout_p)
        if _rand_choice(self.freq_mask_p):
            out = _freq_mask(out, n_masks=self.freq_mask_n,
                             max_width=self.freq_mask_max_width)
        if (
            out.ndim == 2
            and self.sampling_frequency is not None
            and self.notch_freqs
            and _rand_choice(self.notch_p)
        ):
            for f0 in self.notch_freqs:
                for c in range(out.shape[0]):
                    out[c] = _fft_notch(
                        out[c],
                        fs=self.sampling_frequency,
                        freq=f0,
                        bandwidth=self.notch_bandwidth,
                    )
        if _rand_choice(self.time_reverse_p):
            out = np.flip(out, axis=-1).copy()
        if _rand_choice(self.sign_flip_p):
            out = -out
        return out.astype(arr.dtype)


class MEGLabeledDataset(Dataset):
    """MEG dataset with labels, for k-NN evaluation.

    Parameters
    ----------
    data : ndarray, shape (..., T) — e.g. (C, T) or (C, F, T)
    labels : ndarray, shape (T,)
    window_length : int
    stride : int
    """
    def __init__(self, data, labels, window_length, stride):
        self.data = data
        self.labels = labels
        self.window_length = int(window_length)
        self.stride = int(stride)

        T = data.shape[-1]
        self.windows = []
        if T < self.window_length:
            self.windows.append(0)
        else:
            for s in range(0, T - self.window_length + 1, self.stride):
                self.windows.append(s)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        window = self.data[..., start:start + self.window_length].astype(np.float32)
        window_labels = self.labels[start:start + self.window_length]
        majority_label = int(np.bincount(window_labels).argmax())
        return torch.from_numpy(window), majority_label


class MEGDataset(Dataset):
    """MEG Dataset.

    Each __getitem__ returns:
        [view0, view1, ..., viewN]
    where each view is a (..., L) tensor — (C, L) for raw or (C, F, L) for TF.

    Parameters
    ----------
    files : list[str]
        Paths to .npy files of shape (..., T)
    window_length : int
        Length of each window in samples
    stride : int
        Stride between consecutive windows
    crop_lengths : list[int]
        Global crop lengths (DINO-style)
    n_local_crops : int
        Number of local crops
    weak_transform, strong_transform
        Same as before
    """
    def __init__(
        self,
        files=None,
        window_length=None,
        stride=None,
        crop_lengths=None,
        n_local_crops=2,
        local_crop_length=None,
        weak_transform=None,
        strong_transform=None,
        arrays=None,
        temporal_neighbor=False,
    ):
        if arrays is not None:
            self.data = list(arrays)
        elif files is not None:
            self.data = [np.load(f) for f in files]
        else:
            raise ValueError("Either files or arrays must be provided")

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.crop_lengths = list(crop_lengths)
        self.n_local_crops = int(n_local_crops)
        self.local_crop_length = int(local_crop_length) if local_crop_length is not None else min(self.crop_lengths)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.temporal_neighbor = temporal_neighbor

        # precompute window indices: (file_id, start)
        self.windows = []
        for file_id, arr in enumerate(self.data):
            T = arr.shape[-1]
            if T < self.window_length:
                self.windows.append((file_id, 0))
            else:
                starts = list(range(0, T - self.window_length + 1, self.stride))
                for s in starts:
                    self.windows.append((file_id, s))

    def __len__(self):
        return len(self.windows)

    def _random_crop(self, arr, L):
        T = arr.shape[-1]
        if T <= L:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, L - T)]
            arr = np.pad(arr, pad_width, mode="constant")
            return arr[..., :L].astype(np.float32)
        start = random.randint(0, T - L)
        return arr[..., start:start+L].astype(np.float32)

    def __getitem__(self, idx):
        file_id, start = self.windows[idx]
        arr = self.data[file_id][..., start:start + self.window_length]

        views = []

        # global crops (weak)
        for L in self.crop_lengths:
            crop = self._random_crop(arr, L)
            if self.weak_transform:
                crop = self.weak_transform(crop)
            views.append(torch.from_numpy(crop).float())

        # local crops (strong)
        for _ in range(self.n_local_crops):
            crop = self._random_crop(arr, self.local_crop_length)
            if self.strong_transform:
                crop = self.strong_transform(crop)
            views.append(torch.from_numpy(crop).float())

        # temporal neighbor crop (weak) — from adjacent window in same file
        if self.temporal_neighbor:
            neighbor_idx = None
            if idx + 1 < len(self.windows) and self.windows[idx + 1][0] == file_id:
                neighbor_idx = idx + 1
            elif idx - 1 >= 0 and self.windows[idx - 1][0] == file_id:
                neighbor_idx = idx - 1

            if neighbor_idx is not None:
                nf, ns = self.windows[neighbor_idx]
                neighbor_arr = self.data[nf][..., ns:ns + self.window_length]
            else:
                # No adjacent window in same file — use self with different crop
                neighbor_arr = arr

            crop = self._random_crop(neighbor_arr, self.crop_lengths[0])
            if self.weak_transform:
                crop = self.weak_transform(crop)
            views.append(torch.from_numpy(crop).float())

        return views
