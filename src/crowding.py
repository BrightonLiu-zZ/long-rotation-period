"""
Theissen 2026-11-06: TESS has much more contamination than Kepler, so clean the McQuillan sample.
Reads CROWDSAP (SPOC header: fraction of flux in the optimal aperture that comes from the target) for every
downloaded McQuillan star-sector, then recomputes the LS hit rate by CROWDSAP tier and period range.
Combined mode uses the star's median CROWDSAP over its sectors.
Writes results/crowding_sectors.csv, results/crowding_hit_rate.csv and results/plots/crowding_hit_rate.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm

from granular_mearth import colors, ink, wilson
from step2_ls import progress_path, repo, results_dir, tolerance

truth_path = results_dir / "mcquillan_truth_table.csv"
crowd_edges = [0.0, 0.5, 0.8, 1.01]
crowd_labels = ["< 0.5", "0.5-0.8", ">= 0.8"]
period_edges = [8, 14, 20, 30]  # d
period_labels = ["8-14", "14-20", "20-30"]


def read_crowdsap() -> pd.DataFrame:
    truth = pd.read_csv(truth_path)
    progress = pd.read_csv(progress_path, keep_default_na=False)
    progress = progress[(progress["status"] == "done") & progress["tic_id"].isin(truth["tic_id"])]
    rows = []
    for row in tqdm(progress.itertuples(), desc="read CROWDSAP", total=len(progress)):
        try:
            with fits.open(repo / row.path) as hdul:
                crowd = hdul["LIGHTCURVE"].header.get("CROWDSAP", np.nan)
        except (OSError, KeyError) as exc:
            print(f"{row.path}: {exc!r}")
            crowd = np.nan
        rows.append({"tic_id": int(row.tic_id), "sector": str(row.sector), "crowdsap": crowd})
    return pd.DataFrame(rows)


def hit_rates(crowd: pd.DataFrame) -> pd.DataFrame:
    star_crowd = crowd.groupby("tic_id")["crowdsap"].median().rename("crowdsap")
    rows = []
    for flux in ["pdcsap", "sap"]:
        table = pd.read_csv(results_dir / f"ls_{flux}_mcquillan.csv", dtype={"sector": str})
        table = table[table["status"] == "done"]
        sectors = table[table["mode"] == "sector"].merge(crowd, on=["tic_id", "sector"])
        combined = table[table["mode"] == "combined"].merge(star_crowd, left_on="tic_id", right_index=True)
        for mode, part in [("sector", sectors), ("combined", combined)]:
            part = part.copy()
            part["crowd_tier"] = pd.cut(part["crowdsap"], crowd_edges, labels=crowd_labels, right=False)
            part["period_range"] = pd.cut(part["period"], period_edges, labels=period_labels, right=False)
            pool = part["peak_period"].values  # (M,) chance: truth paired with every peak of this flux and mode
            for (tier, prange), group in part.groupby(["crowd_tier", "period_range"], observed=True):
                hits = int((group["outcome"] == "hit").sum())
                low, high = wilson(hits, len(group))
                truth = group["period"].values[:, None]  # (N, 1)
                rows.append({
                    "flux": flux,
                    "mode": mode,
                    "crowd_tier": tier,
                    "period_range": prange,
                    "n": len(group),
                    "hits": hits,
                    "hit_rate": hits / len(group),
                    "wilson68_low": low,
                    "wilson68_high": high,
                    "chance_rate": np.mean(np.abs(pool[None, :] - truth) < tolerance * truth),
                })
    return pd.DataFrame(rows)


def plot(rates: pd.DataFrame) -> plt.Figure:
    part = rates[rates["mode"] == "combined"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, prange in zip(axes, period_labels):
        for i, flux in enumerate(["pdcsap", "sap"]):
            sub = part[(part["period_range"] == prange) & (part["flux"] == flux)].set_index("crowd_tier").reindex(crowd_labels)
            x = np.arange(len(crowd_labels)) + (i - 0.5) * 0.25
            y = sub["hit_rate"]
            ax.errorbar(x, y, yerr=[y - sub["wilson68_low"], sub["wilson68_high"] - y], fmt="o", color=colors[flux], capsize=2, label=flux.upper())
            ax.hlines(sub["chance_rate"], x - 0.1, x + 0.1, color=colors[flux], linewidth=1.2, linestyle=":")
            if flux == "pdcsap":
                for xi, n in zip(np.arange(len(crowd_labels)), sub["n"]):
                    ax.text(xi, 0.95, f"n={int(n) if n == n else 0}", ha="center", fontsize=8, color=ink)
        ax.set_xticks(range(len(crowd_labels)), crowd_labels)
        ax.set_xlabel("CROWDSAP (target share of aperture flux)")
        ax.set_title(f"P = {prange} d", fontsize=10)
        ax.set_ylim(-0.03, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of stars recovered\n(dotted: by chance)")
    axes[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    return fig


def main() -> None:
    crowd = read_crowdsap()
    crowd.to_csv(results_dir / "crowding_sectors.csv", index=False)
    rates = hit_rates(crowd)
    rates.to_csv(results_dir / "crowding_hit_rate.csv", index=False)
    fig = plot(rates)
    fig.savefig(results_dir / "plots" / "crowding_hit_rate.png", dpi=150)
    plt.close(fig)
    print(rates[rates["mode"] == "combined"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
