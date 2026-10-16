"""
Step 5 helpers shared by the torch side (step5_gan.py, step5_generate.py; env swm) and the periodogram side (step5_eval.py; env astro).
numpy + pandas only: in env astro, torch and numpy's MKL load two different OpenMP runtimes and crash in the same process.
"""

from pathlib import Path

import numpy as np
import pandas as pd

repo = Path(__file__).resolve().parents[1]
data_dir = repo / "data" / "gan"
half = 672  # bins per half, must match step5_data.py
clip = 10.0  # normalised units; flares beyond this are clipped in model inputs and training targets
min_scale = 1e-4  # relative flux; floor on the per-sample scale so flat curves do not blow up


def normalise(flux: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample centre and scale from the observed bins of a conditioning segment.
    centre = median, scale = 1.4826 x median absolute deviation (a robust std), floored at min_scale.
    flux, mask: (N, L). Returns centre (N,) and scale (N,).
    """
    centre = np.zeros(len(flux))
    scale = np.zeros(len(flux))
    for i in range(len(flux)):
        observed = flux[i][mask[i]]
        centre[i] = np.median(observed)
        scale[i] = max(1.4826 * np.median(np.abs(observed - centre[i])), min_scale)
    return centre, scale


def model_input(flux: np.ndarray, mask: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Normalise, clip and zero the unobserved bins, then stack [flux, mask] as channels.
    Returns (N, 2, L) float32.
    """
    x = np.clip((flux - centre[:, None]) / scale[:, None], -clip, clip) * mask
    return np.stack([x, mask], axis=1).astype(np.float32)


def load_data(flux_name: str, tag: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Read the gridded sectors written by step5_data.py and their metadata (same row order).
    """
    arrays = np.load(data_dir / f"{flux_name}{tag}.npz")
    meta = pd.read_csv(data_dir / f"{flux_name}{tag}_meta.csv")
    return arrays["flux"], arrays["mask"], meta
