"""
Summary figures for the Theissen update (Steps 2, 3 and 5).
Figure 1: LS hit rate per MEarth period bin, PDCSAP vs SAP, all sectors of a star combined (one point per star), with the shuffle chance rate.
Figure 2: Step 5 on held-out star-sectors. Left: forecast error of half 2 from half 1 (lower is better). Right: LS hit rate after extending each sector by 28 d, with chance.
The two figures count different units (stars vs star-sectors), so they are never drawn on one axis.
Reads results/ls_*_summary.csv, results/step3_chance_baseline.csv, results/step5_*_summary.csv; writes results/plots/report_*.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_ls import period_labels, results_dir

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 of the dataviz reference palette, fixed order
ink = "#52514e"
bin_names = {"<13.7": "< 13.7 d", "13.7-27": "13.7 to 27 d", ">27": "> 27 d"}


def grouped_bars(ax: plt.Axes, values: pd.DataFrame, chance: pd.DataFrame | None, ns: list[int], percent: bool) -> None:
    """
    One group per period bin, one bar per column of values (rows = period_labels order).
    A short dark tick on a bar marks the chance rate for that bar; each bar carries its value as a label.
    """
    width = 0.8 / len(values.columns)
    x = np.arange(len(values))
    for i, column in enumerate(values.columns):
        positions = x - 0.4 + width * (i + 0.5)
        heights = values[column].values
        ax.bar(positions, heights, width * 0.92, color=colors[i], label=column)
        for position, height in zip(positions, heights):
            if percent:
                label = f"{round(100 * height)}%"
            else:
                label = f"{round(height, 2)}"
            ax.text(position, height, label, ha="center", va="bottom", fontsize=8, color=ink)
        if chance is not None:
            ax.hlines(chance[column].values, positions - width * 0.46, positions + width * 0.46, color="black", linewidth=2)
    ticks = []
    for label, n in zip(period_labels, ns):
        ticks.append(f"{bin_names[label]}\nn = {n}")
    ax.set_xticks(x, ticks)
    ax.spines[["top", "right"]].set_visible(False)


def figure_pdcsap_vs_sap() -> None:
    chance = pd.read_csv(results_dir / "step3_chance_baseline.csv")
    chance = chance[chance["mode"] == "combined"]
    observed = chance.pivot(index="period_bin", columns="flux", values="observed_hit_rate").loc[period_labels]
    floor = chance.pivot(index="period_bin", columns="flux", values="chance_hit_rate_mean").loc[period_labels]
    ns = chance[chance["flux"] == "pdcsap"].set_index("period_bin").loc[period_labels, "n"].tolist()
    observed = observed[["pdcsap", "sap"]].rename(columns={"pdcsap": "PDCSAP", "sap": "SAP"})
    floor = floor[["pdcsap", "sap"]].rename(columns={"pdcsap": "PDCSAP", "sap": "SAP"})

    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, observed, floor, ns, percent=True)
    ax.set_ylabel("MEarth stars with LS peak within 10% of true period")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.plot([], [], color="black", linewidth=2, label="chance (periods shuffled)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Lomb-Scargle on SPOC light curves, all sectors of a star combined", fontsize=10)
    fig.tight_layout()
    fig.savefig(results_dir / "plots" / "report_ls_pdcsap_vs_sap.png", dpi=150)
    plt.close(fig)


def figure_step5() -> None:
    forecast = pd.read_csv(results_dir / "step5_forecast_summary.csv")
    forecast = forecast[forecast["truth"] == "MEarth"]
    methods = {"mean": "flat line", "sinusoid": "fitted sinusoid", "l1": "conv net (L1 only)", "gan_1": "GAN"}
    errors = forecast.pivot(index="period_bin", columns="method", values="median_mae").loc[period_labels, list(methods)].rename(columns=methods)
    ns = forecast[forecast["method"] == "mean"].set_index("period_bin").loc[period_labels, "n"].tolist()

    ls = pd.read_csv(results_dir / "step5_ls_summary.csv")
    ls = ls[ls["truth"] == "MEarth"]
    extensions = {"none": "no extension", "l1": "conv net (L1 only)", "gan": "GAN"}
    hits = ls.pivot(index="period_bin", columns="method", values="frac_within_10pct").loc[period_labels, list(extensions)].rename(columns=extensions)
    floor = ls.pivot(index="period_bin", columns="method", values="chance_hit_rate_mean").loc[period_labels, list(extensions)].rename(columns=extensions)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    grouped_bars(left, errors, None, ns, percent=False)
    left.set_ylabel("median forecast error\n(units of first-half scatter, lower is better)")
    left.set_title("Predict 2nd half of a sector from the 1st half (held-out stars)", fontsize=10)
    left.set_ylim(0, 1.4)
    left.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

    hit_colors = ["#a8a6a0", colors[2], colors[3]]  # neutral grey for the baseline; models keep their left-panel colours
    width = 0.8 / len(hits.columns)
    x = np.arange(len(hits))
    for i, column in enumerate(hits.columns):
        positions = x - 0.4 + width * (i + 0.5)
        right.bar(positions, hits[column].values, width * 0.92, color=hit_colors[i], label=column)
        right.hlines(floor[column].values, positions - width * 0.46, positions + width * 0.46, color="black", linewidth=2)
        for position, height in zip(positions, hits[column].values):
            right.text(position, height, f"{round(100 * height)}%", ha="center", va="bottom", fontsize=8, color=ink)
    ticks = []
    for label, n in zip(period_labels, ns):
        ticks.append(f"{bin_names[label]}\nn = {n}")
    right.set_xticks(x, ticks)
    right.spines[["top", "right"]].set_visible(False)
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    right.set_ylabel("sectors with LS peak within 10% of true period")
    right.set_title("LS after extending each sector by 28 d of model output", fontsize=10)
    right.plot([], [], color="black", linewidth=2, label="chance (periods shuffled)")
    right.legend(frameon=False, fontsize=9)
    left.set_xlabel("MEarth period bin (n = star-sectors)")
    right.set_xlabel("MEarth period bin (n = star-sectors)")
    fig.tight_layout()
    fig.savefig(results_dir / "plots" / "report_step5.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure_pdcsap_vs_sap()
    figure_step5()
    print("wrote results/plots/report_ls_pdcsap_vs_sap.png and results/plots/report_step5.png")
