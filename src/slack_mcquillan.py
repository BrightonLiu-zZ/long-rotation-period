"""
Figures and table for the 2026-11-06 Slack follow-up to Theissen: McQuillan+2013 stars as long-period truth.
Reads results/granular_mearth.csv and results/granular_mcquillan.csv (written by granular_mearth.py).
Writes results/plots/slack_1106_stars_per_bin.png, results/plots/slack_1106_hit_rate.png and results/slack_1106_table.csv.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from granular_mearth import bin_edges, colors, ink
from step2_ls import orbit_period, results_dir

sample_colors = {"McQuillan (Kepler)": "#1baf7a", "MEarth": "#b5b4ae"}  # aqua slot 3; MEarth in gray so it stays the background reference


def load(truth: str) -> pd.DataFrame:
    rates = pd.read_csv(results_dir / f"granular_{truth}.csv")
    return rates[rates["mode"] == "combined"]  # one periodogram per star, all its sectors


def plot_stars_per_bin(mcq: pd.DataFrame, mearth: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.8
    for i, (name, rates) in enumerate([("McQuillan (Kepler)", mcq), ("MEarth", mearth)]):
        part = rates[rates["flux"] == "pdcsap"]
        x = part["bin_low"] + 1 + (i - 0.5) * width
        ax.bar(x, part["n_stars"], width=width - 0.08, color=sample_colors[name], label=name)
    ax.set_xticks(bin_edges)
    ax.set_xlabel("Rotation period (d)")
    ax.set_ylabel("Stars with SPOC 2-min data")
    ax.set_title("Stars per 2-day period bin", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_hit_rate(mcq: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, flux in enumerate(["pdcsap", "sap"]):
        part = mcq[mcq["flux"] == flux]
        x = part["bin_low"] + 1 + (i - 0.5) * 0.3
        y = part["hit_rate"]
        ax.errorbar(x, y, yerr=[y - part["wilson68_low"], part["wilson68_high"] - y], fmt="o", color=colors[flux], capsize=2, label=f"{flux.upper()}")
        ax.hlines(part["chance_rate"], part["bin_low"], part["bin_low"] + 2, color=colors[flux], linewidth=1.2, linestyle=":", label=f"{flux.upper()}, by chance")
    ax.axvline(orbit_period, color=ink, linewidth=0.8, linestyle="--")
    ax.text(orbit_period + 0.2, 0.9, "13.7 d", color=ink, fontsize=8)
    ax.set_xticks(bin_edges)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("McQuillan rotation period (d)")
    ax.set_ylabel("Fraction of stars recovered\n(LS peak within 10%)")
    ax.set_title("Lomb-Scargle recovery of McQuillan periods (bars: 68% interval)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def table(mcq: pd.DataFrame, mearth: pd.DataFrame) -> pd.DataFrame:
    wide = mcq.pivot(index="bin_low", columns="flux", values=["hit_rate", "chance_rate", "n_stars"])
    out = pd.DataFrame({
        "period (d)": [f"{b}-{b + 2}" for b in wide.index],
        "McQuillan stars": wide[("n_stars", "pdcsap")].astype(int).values,
        "MEarth stars": mearth[mearth["flux"] == "pdcsap"].set_index("bin_low").reindex(wide.index)["n_stars"].fillna(0).astype(int).values,
        "PDCSAP recovered": (100 * wide[("hit_rate", "pdcsap")]).round().astype(int).astype(str).values,
        "PDCSAP chance": (100 * wide[("chance_rate", "pdcsap")]).round().astype(int).astype(str).values,
        "SAP recovered": (100 * wide[("hit_rate", "sap")]).round().astype(int).astype(str).values,
        "SAP chance": (100 * wide[("chance_rate", "sap")]).round().astype(int).astype(str).values,
    })
    for col in ["PDCSAP recovered", "PDCSAP chance", "SAP recovered", "SAP chance"]:
        out[col] = out[col] + "%"
    return out


def main() -> None:
    mcq, mearth = load("mcquillan"), load("mearth")
    for name, fig in [("stars_per_bin", plot_stars_per_bin(mcq, mearth)), ("hit_rate", plot_hit_rate(mcq))]:
        fig.savefig(results_dir / "plots" / f"slack_1106_{name}.png", dpi=150)
        plt.close(fig)
    out = table(mcq, mearth)
    out.to_csv(results_dir / "slack_1106_table.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
