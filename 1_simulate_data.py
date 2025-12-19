"""Simulate HMM data."""

import os
import numpy as np
from osl_dynamics.simulation import HMM_MVN

os.makedirs("data", exist_ok=True)

sim = HMM_MVN(
    #n_samples=50*10*60*250,  # 50 subjects x 10 min at 250 Hz
    n_samples=10000,
    n_states=5,
    n_channels=52,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
data = sim.time_series.astype(np.float32).T  # (channels, time)

np.save("data/X.npy", data)
