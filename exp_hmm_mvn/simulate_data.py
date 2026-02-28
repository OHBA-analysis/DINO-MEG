"""Simulate HMM-MVN data for DINO validation (train + eval splits)."""

import os
import sys
import numpy as np
from osl_dynamics.simulation import HMM_MVN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_SAMPLES = 250 * 300   # 75 000 samples (~5 min at 250 Hz)
N_STATES = 5
N_CHANNELS = 52
HMM_KWARGS = dict(
    n_samples=N_SAMPLES,
    n_states=N_STATES,
    n_channels=N_CHANNELS,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)

for split, seed in [("train", 0), ("eval", 1)]:
    np.random.seed(seed)
    sim = HMM_MVN(**HMM_KWARGS)
    X = sim.time_series.astype(np.float32).T          # (C, T)
    Y = np.argmax(sim.mode_time_course, axis=1).astype(np.int32)  # (T,)
    np.save(os.path.join(DATA_DIR, f"X_{split}.npy"), X)
    np.save(os.path.join(DATA_DIR, f"Y_{split}.npy"), Y)
    print(f"{split}: X={X.shape}, Y={Y.shape}, states={np.unique(Y)}")

print("Done. Files saved to", DATA_DIR)
