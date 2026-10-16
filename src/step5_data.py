"""
Step 5 of docs/ROADMAP.md, data: every downloaded SPOC sector on a fixed 30-min grid, split into two halves.
Half 1 (first 14 d) is the model input, half 2 (next 14 d) is the forecast target (Theissen: first half of a sector predicts the second half).
Stars are assigned to 5 folds (stratified by period bin) so every star is a held-out test star exactly once.
Writes data/gan/{flux}.npz (flux, mask; shape (N, 1344)) and data/gan/{flux}_meta.csv (one row per star-sector, same order).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from step2_ls import bin_days, progress_path, read_sector, repo, truth_path

n_grid = 1344  # 30-min bins = 28 d, one full sector
half = n_grid // 2  # 672 bins = 14 d, about one TESS orbit
min_half_points = 100  # observed bins required in each half
n_folds = 5
out_dir = repo / "data" / "gan"


def to_grid(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Place one binned sector on a uniform grid that starts at its first bin.
    read_sector already bins to 30 min, so each bin centre rounds to one grid index; anything past 28 d is cropped.
    Returns (flux, observed-mask, grid start time in BTJD).
    """
    idx = np.round((t - t[0]) / bin_days).astype(int)
    keep = idx < n_grid
    flux = np.zeros(n_grid, dtype=np.float32)
    mask = np.zeros(n_grid, dtype=bool)
    flux[idx[keep]] = y[keep]
    mask[idx[keep]] = True
    return flux, mask, float(t[0])


def assign_folds(stars: pd.DataFrame, seed: int) -> pd.Series:
    """
    Shuffle stars within each period bin and deal them round-robin into folds, so each fold gets a share of every bin.
    """
    rng = np.random.default_rng(seed)
    folds = pd.Series(0, index=stars["tic_id"].values)
    for _, group in stars.groupby("period_bin"):
        order = rng.permutation(group["tic_id"].values)
        for i, tic_id in enumerate(order):
            folds[tic_id] = i % n_folds
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--limit-stars", type=int, default=None, help="only the first N stars (smoke test)")
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke")
    args = parser.parse_args()

    truth = pd.read_csv(truth_path)
    progress = pd.read_csv(progress_path, keep_default_na=False)
    progress = progress[progress["status"] == "done"]
    stars = truth[truth["tic_id"].isin(progress["tic_id"])]
    if args.limit_stars is not None:
        stars = stars.head(args.limit_stars)
    folds = assign_folds(stars, seed=0)
    progress = progress[progress["tic_id"].isin(stars["tic_id"])]
    star_info = stars.set_index("tic_id")

    fluxes, masks, rows = [], [], []
    for _, row in tqdm(progress.iterrows(), desc=f"grid {args.flux}", total=len(progress)):
        t, y = read_sector(repo / row["path"], args.flux)
        status = "done"
        if len(t) == 0:
            status = "empty"
        else:
            flux, mask, t0 = to_grid(t, y)
            if mask[:half].sum() < min_half_points or mask[half:].sum() < min_half_points:
                status = "too_few_points"
        star = star_info.loc[row["tic_id"]]
        meta = {"tic_id": row["tic_id"], "sector": row["sector"], "sample": star["sample"], "period": star["period"], "period_bin": star["period_bin"], "fold": folds[row["tic_id"]], "status": status}
        if status == "done":
            fluxes.append(flux)
            masks.append(mask)
            meta["t0"] = t0
            meta["n_obs_half1"] = mask[:half].sum()
            meta["n_obs_half2"] = mask[half:].sum()
        rows.append(meta)

    meta = pd.DataFrame(rows)
    print(meta["status"].value_counts().to_string())
    done = meta[meta["status"] == "done"].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.flux}{args.tag}.npz", flux=np.stack(fluxes), mask=np.stack(masks))
    done.to_csv(out_dir / f"{args.flux}{args.tag}_meta.csv", index=False)
    print(done.groupby(["fold", "period_bin"]).size().unstack().to_string())
    print(f"wrote {len(done)} star-sectors to {out_dir}")


if __name__ == "__main__":
    main()
