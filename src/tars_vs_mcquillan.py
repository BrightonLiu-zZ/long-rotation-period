"""
Theissen 2026-11-06: "maybe this is where things start to get bad for TARS".
McQuillan+2013 (Kepler periods, independent of TESS) is the truth; TARS adopted_period and our SPOC LS
(combined mode, PDCSAP and SAP) are both scored against it, on the same stars.
TARS lists only stars where it reports a period, so its rate is agreement among reported periods, not completeness.
Writes results/tars_vs_mcquillan.csv and results/plots/tars_vs_mcquillan.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granular_mearth import colors, ink, wilson
from step1_truth_table import tars_path
from step2_ls import orbit_period, results_dir, tolerance

bin_edges = np.arange(0, 32, 2)  # d
tars_color = "#1baf7a"  # aqua, categorical slot 3
tars_max_search = 20.0  # d, TARS searches P < 20 d (arXiv 2603.05586, App. A)


def load() -> pd.DataFrame:
    mcq = pd.read_csv(results_dir / "mcquillan_spoc.csv")
    tars = pd.read_feather(tars_path, columns=["TICID", "adopted_period"])
    both = mcq.merge(tars, left_on="tic_id", right_on="TICID")
    both = both[both["period"] < bin_edges[-1]].copy()
    both["tars_hit"] = np.abs(both["adopted_period"] - both["period"]) < tolerance * both["period"]
    for flux in ["pdcsap", "sap"]:
        ls = pd.read_csv(results_dir / f"ls_{flux}_mcquillan.csv")
        ls = ls[(ls["mode"] == "combined") & (ls["status"] == "done")]
        both = both.merge(ls[["tic_id", "outcome"]].rename(columns={"outcome": f"{flux}_outcome"}), on="tic_id", how="left")
    both["bin_low"] = bin_edges[np.digitize(both["period"], bin_edges) - 1]
    return both


def rates(both: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_low, group in both.groupby("bin_low"):
        row = {"bin_low": bin_low, "bin_high": bin_low + 2, "n_tars": len(group)}
        hits = int(group["tars_hit"].sum())
        row["tars_hit_rate"] = hits / len(group)
        row["tars_low"], row["tars_high"] = wilson(hits, len(group))
        for flux in ["pdcsap", "sap"]:
            ls = group[f"{flux}_outcome"].dropna()  # LS was run only for stars at 8-30 d
            row[f"n_{flux}"] = len(ls)
            if len(ls):
                hits = int((ls == "hit").sum())
                row[f"{flux}_hit_rate"] = hits / len(ls)
                row[f"{flux}_low"], row[f"{flux}_high"] = wilson(hits, len(ls))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    series = [("tars", "TARS (FFI)", tars_color), ("pdcsap", "Our LS, PDCSAP", colors["pdcsap"]), ("sap", "Our LS, SAP", colors["sap"])]
    for i, (key, label, color) in enumerate(series):
        part = table.dropna(subset=[f"{key}_hit_rate"])
        x = part["bin_low"] + 1 + (i - 1) * 0.35
        y = part[f"{key}_hit_rate"]
        ax.errorbar(x, y, yerr=[y - part[f"{key}_low"], part[f"{key}_high"] - y], fmt="o", color=color, capsize=2, label=label)
    for xi, n in zip(table["bin_low"] + 1, table["n_tars"]):
        ax.text(xi, 1.06, str(n), ha="center", fontsize=8, color=ink)
    ax.axvline(orbit_period, color=ink, linewidth=0.8, linestyle="--")
    ax.axvline(tars_max_search, color=tars_color, linewidth=0.8, linestyle="--")
    ax.text(tars_max_search + 0.2, 0.9, "TARS search limit", color=ink, fontsize=8)
    ax.set_xticks(bin_edges)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("McQuillan (Kepler) rotation period (d); numbers on top: stars with a TARS period")
    ax.set_ylabel("Fraction within 10%\nof the Kepler period")
    ax.set_title("Same stars: TARS vs our SPOC Lomb-Scargle", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    both = load()
    table = rates(both)
    table.to_csv(results_dir / "tars_vs_mcquillan.csv", index=False)
    fig = plot(table)
    fig.savefig(results_dir / "plots" / "tars_vs_mcquillan.png", dpi=150)
    plt.close(fig)
    cols = ["bin_low", "n_tars", "tars_hit_rate", "n_pdcsap", "pdcsap_hit_rate", "sap_hit_rate"]
    print(table[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
