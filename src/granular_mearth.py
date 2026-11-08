"""
Theissen 2026-10-16: where exactly does the LS hit rate drop past 13.7 d? First try, using only MEarth truth and the existing Step 2/3 runs.
Hit rate in 2-day period bins from 8 to 30 d, for PDCSAP / SAP x sector / combined, with Wilson 68% intervals and n per bin.
Sector mode counts star-sectors, combined mode counts stars (as in step2_ls.summarize).
No new periodograms: reads results/ls_pdcsap.csv and results/ls_sap.csv.
Writes results/granular_mearth.csv and results/plots/granular_mearth.png.
--truth mcquillan reads results/ls_{flux}_mcquillan.csv (McQuillan+2013 Kepler M dwarfs) and writes granular_mcquillan.*.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_ls import orbit_period, results_dir, tolerance

bin_edges = np.arange(8, 32, 2)  # d
z = 1.0  # Wilson interval at 68%
colors = {"pdcsap": "#2a78d6", "sap": "#eb6834"}  # categorical slots 1-2 of the dataviz reference palette, as in plot_report.py
ink = "#52514e"


def wilson(hits: int, n: int) -> tuple[float, float]:
    """
    Wilson score interval for a binomial rate; stays inside [0, 1] when n is tiny, unlike the normal approximation.
    """
    if n == 0:
        return np.nan, np.nan
    rate = hits / n
    centre = (rate + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half_width = z * np.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half_width, centre + half_width


def binned_rates(table: pd.DataFrame, flux: str, sample: str = "mearth") -> pd.DataFrame:
    done = table[(table["status"] == "done") & table["sample"].str.startswith(sample)].copy()
    done["bin_low"] = bin_edges[np.digitize(done["period"], bin_edges) - 1]
    done = done[(done["period"] >= bin_edges[0]) & (done["period"] < bin_edges[-1])]
    rows = []
    for (mode, bin_low), group in done.groupby(["mode", "bin_low"]):
        hits = int((group["outcome"] == "hit").sum())
        low, high = wilson(hits, len(group))
        # chance: this bin's truth periods paired with every peak of the 8-30 d pool (same mode), i.e. a peak that ignores the star
        pool = done.loc[done["mode"] == mode, "peak_period"].values  # (M,)
        truth = group["period"].values[:, None]  # (N, 1)
        chance = np.mean(np.abs(pool[None, :] - truth) < tolerance * truth)
        rows.append({
            "flux": flux,
            "mode": mode,
            "bin_low": bin_low,
            "bin_high": bin_low + 2,
            "n": len(group),
            "n_stars": group["tic_id"].nunique(),
            "hits": hits,
            "hit_rate": hits / len(group),
            "wilson68_low": low,
            "wilson68_high": high,
            "chance_rate": chance,
        })
    return pd.DataFrame(rows)


def plot(rates: pd.DataFrame, truth_name: str = "MEarth") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ["sector", "combined"]):
        for i, flux in enumerate(["pdcsap", "sap"]):
            part = rates[(rates["mode"] == mode) & (rates["flux"] == flux)]
            x = part["bin_low"] + 1 + (i - 0.5) * 0.3  # small offset so the two error bars do not overlap
            y = part["hit_rate"]
            ax.errorbar(x, y, yerr=[y - part["wilson68_low"], part["wilson68_high"] - y], fmt="o", color=colors[flux], capsize=2, label=flux.upper())
            ax.hlines(part["chance_rate"], part["bin_low"], part["bin_low"] + 2, color=colors[flux], linewidth=1.0, linestyle=":", label=f"{flux.upper()} chance")
            if flux == "pdcsap":
                for xi, n in zip(part["bin_low"] + 1, part["n"]):
                    ax.text(xi, 1.06, str(n), ha="center", fontsize=8, color=ink)
        ax.axvline(orbit_period, color=ink, linewidth=0.8, linestyle="--")
        ax.set_xticks(bin_edges)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel(f"{truth_name} period (d), 2-day bins; dashed = 13.7 d")
        unit = "star-sectors" if mode == "sector" else "stars"
        ax.set_title(f"{mode} mode (numbers on top: {unit} per bin)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"LS peak within 10% of {truth_name} period")
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", choices=["mearth", "mcquillan"], default="mearth")
    args = parser.parse_args()
    tag = "" if args.truth == "mearth" else "_mcquillan"  # step2_ls.py --tag used for the McQuillan run
    truth_name = {"mearth": "MEarth", "mcquillan": "McQuillan"}[args.truth]

    parts = []
    for flux in ["pdcsap", "sap"]:
        parts.append(binned_rates(pd.read_csv(results_dir / f"ls_{flux}{tag}.csv", dtype={"sector": str}), flux, args.truth))
    rates = pd.concat(parts).sort_values(["mode", "flux", "bin_low"]).reset_index(drop=True)
    rates.to_csv(results_dir / f"granular_{args.truth}.csv", index=False)
    fig = plot(rates, truth_name)
    fig.savefig(results_dir / "plots" / f"granular_{args.truth}.png", dpi=150)
    plt.close(fig)
    print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
