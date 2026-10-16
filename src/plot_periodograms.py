"""
Example periodograms for the Theissen report: a mix of hits and misses per period bin.
Recomputes each chosen periodogram with the step2_ls code, so nothing has to be cached.
Writes results/plots/ls_{flux}_{tic}_{sector}.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step2_ls import orbit_period, periodogram_row, progress_path, read_sector, repo, results_dir


def pick_examples(table: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """
    From the MEarth single-sector rows, take n_per_cell hits and n_per_cell misses per period bin when available.
    """
    rows = table[(table["status"] == "done") & (table["mode"] == "sector") & table["sample"].str.startswith("mearth")]
    picks = []
    for period_bin in ["<13.7", "13.7-27", ">27"]:
        in_bin = rows[rows["period_bin"] == period_bin]
        for hit in [True, False]:
            cell = in_bin[(in_bin["outcome"] == "hit") == hit]
            if len(cell):
                picks.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=seed))
    return pd.concat(picks)


def plot_one(row: pd.Series, flux: str, files: pd.DataFrame, out_dir: Path) -> Path:
    """
    Recompute and draw one single-sector periodogram, with the MEarth period, its half and double, and 13.7 d marked.
    """
    path = files[(files["tic_id"] == row["tic_id"]) & (files["sector"] == int(row["sector"]))]["path"].iloc[0]
    t, y = read_sector(repo / path, flux)
    _, periods, power = periodogram_row(t, y, np.zeros(len(t), dtype=int), "sector", row["period"])

    plt.figure(figsize=(7, 3.2))
    plt.plot(periods, power, linewidth=0.8)
    plt.axvline(row["period"], color="tab:green", label=f"MEarth period {round(row['period'], 2)} d")
    plt.axvline(row["period"] / 2, color="tab:green", linestyle=":")
    plt.axvline(row["period"] * 2, color="tab:green", linestyle=":")
    plt.axvline(orbit_period, color="tab:red", linestyle="--", label="13.7 d (TESS orbit)")
    plt.xscale("log")
    plt.xlim(0.1, 100)
    plt.xlabel("period (d)")
    plt.ylabel("LS power")
    plt.title(f"TIC {row['tic_id']}, sector {row['sector']}, {flux.upper()}: peak {round(row['peak_period'], 2)} d ({row['outcome']})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / f"ls_{flux}_{row['tic_id']}_s{row['sector']}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="suffix of the step2_ls output to read; plots go to results/plots{tag}/")
    args = parser.parse_args()

    table = pd.read_csv(results_dir / f"ls_{args.flux}{args.tag}.csv", dtype={"sector": str})
    files = pd.read_csv(progress_path, keep_default_na=False)
    files = files[files["status"] == "done"]
    out_dir = results_dir / f"plots{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pick_examples(table, args.per_cell, args.seed).iterrows():
        print(plot_one(row, args.flux, files, out_dir))


if __name__ == "__main__":
    main()
