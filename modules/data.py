"""Data."""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


def _rand_choice(p):
    return random.random() < p


def _fft_notch(x, fs, freq, bandwidth=1.0):
    N = x.shape[-1]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    X = np.fft.rfft(x)
    mask = ~((freqs >= (freq - bandwidth / 2.0)) & (freq <= (freq + bandwidth / 2.0)))
    X = X * mask
    return np.fft.irfft(X, n=N)


def _time_warp(arr, max_warp=0.12):
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
    C, T = arr.shape
    out = arr.copy()
    L = int(T * mask_ratio)
    for _ in range(n_masks):
        if L <= 0:
            break
        start = random.randint(0, max(0, T - L))
        out[:, start : start + L] = 0.0
    return out


def _channel_mask(arr, mask_p=0.15):
    C, T = arr.shape
    mask = (np.random.rand(C) >= mask_p).astype(arr.dtype)
    return arr * mask[:, None]


def _amplitude_scaling(arr, sigma=0.1):
    C, T = arr.shape
    scales = np.random.normal(1.0, sigma, size=(C,))
    return arr * scales[:, None]


def _add_gaussian_noise(arr, std):
    if std <= 0:
        return arr
    noise = np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype)
    return arr + noise


def _baseline_shift(arr, shift_std=1e-3):
    C, T = arr.shape
    shifts = np.random.normal(0.0, shift_std, size=(C,))
    return arr + shifts[:, None]


def _random_crop(arr, crop_length):
    C, T = arr.shape
    if T <= crop_length:
        pad = crop_length - T
        out = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
        return out.astype(arr.dtype)
    start = random.randint(0, T - crop_length)
    return arr[:, start : start + crop_length].astype(arr.dtype)


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
        time_reverse_p=0.05,
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
        self.time_reverse_p = time_reverse_p
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
        if (
            self.sampling_frequency is not None
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
            out = np.flip(out, axis=1).copy()
        return out.astype(arr.dtype)


class MEGDataset(Dataset):
    def __init__(
        self,
        file_list,
        crop_lengths,
        n_local_crops=2,
        weak_transform=None,
        strong_transform=None,
    ):
        self.files = file_list
        self.crop_lengths = crop_lengths
        self.n_local_crops = n_local_crops
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.files)

    def random_crop(self, arr, L):
        _, T = arr.shape
        if T <= L:
            pad = L - T
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
            return arr[:, :L].astype(np.float32)
        start = random.randint(0, T - L)
        return arr[:, start : start + L].astype(np.float32)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # (C, T)
        views = []
        # global crops (weak)
        for L in self.crop_lengths:
            crop = self.random_crop(arr, L)
            if self.weak_transform:
                crop = self.weak_transform(crop)
            views.append(crop)
        # local crops (strong)
        smallest = self.crop_lengths[-1]
        for _ in range(self.n_local_crops):
            crop = self.random_crop(arr, smallest)
            if self.strong_transform:
                crop = self.strong_transform(crop)
            views.append(crop)
        return [torch.from_numpy(v).float() for v in views]
