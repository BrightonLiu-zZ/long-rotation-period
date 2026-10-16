"""
Step 3 of docs/ROADMAP.md: is the true period recovered in SAP but not PDCSAP, in both, or in neither?
Reads results/ls_pdcsap.csv and results/ls_sap.csv (written by step2_ls.py) and pairs rows by star, sector and mode.
Writes results/step3_sap_vs_pdcsap.csv (paired rows) and results/step3_counts.csv (category counts per bin and mode).
"""

import argparse

import numpy as np
import pandas as pd

from step2_ls import period_labels, results_dir

keys = ["tic_id", "sample", "period", "period_bin", "mode", "sector"]
value_cols = ["peak_period", "peak_power", "outcome", "near_orbit", "peak_at_max_period"]


def category(pdc_hit: bool, sap_hit: bool) -> str:
    """
    The four cells of the ROADMAP Step 3 table (PDCSAP-only added for completeness).
    """
    if sap_hit and not pdc_hit:
        return "SAP only"
    if pdc_hit and not sap_hit:
        return "PDCSAP only"
    if sap_hit and pdc_hit:
        return "both"
    return "neither"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="suffix of the step2_ls outputs to read and of the outputs written")
    args = parser.parse_args()
    pdc = pd.read_csv(results_dir / f"ls_pdcsap{args.tag}.csv", dtype={"sector": str})
    sap = pd.read_csv(results_dir / f"ls_sap{args.tag}.csv", dtype={"sector": str})
    pdc = pdc[pdc["status"] == "done"][keys + value_cols]
    sap = sap[sap["status"] == "done"][keys + value_cols]
    paired = pdc.merge(sap, on=keys, suffixes=("_pdcsap", "_sap"))

    categories = []
    for _, row in paired.iterrows():
        categories.append(category(row["outcome_pdcsap"] == "hit", row["outcome_sap"] == "hit"))
    paired["category"] = categories
    paired["truth"] = np.where(paired["sample"] == "tars", "TARS", "MEarth")
    paired.to_csv(results_dir / f"step3_sap_vs_pdcsap{args.tag}.csv", index=False)

    rows = []
    for (mode, truth, period_bin), group in paired.groupby(["mode", "truth", "period_bin"]):
        counts = group["category"].value_counts()
        rows.append({
            "mode": mode,
            "truth": truth,
            "period_bin": period_bin,
            "n": len(group),
            "both": counts.get("both", 0),
            "SAP only": counts.get("SAP only", 0),
            "PDCSAP only": counts.get("PDCSAP only", 0),
            "neither": counts.get("neither", 0),
            "peak near 13.7 d, PDCSAP": int(group["near_orbit_pdcsap"].sum()),
            "peak near 13.7 d, SAP": int(group["near_orbit_sap"].sum()),
        })
    counts = pd.DataFrame(rows)
    counts["period_bin"] = pd.Categorical(counts["period_bin"], categories=period_labels, ordered=True)
    counts = counts.sort_values(["mode", "truth", "period_bin"]).reset_index(drop=True)
    counts.to_csv(results_dir / f"step3_counts{args.tag}.csv", index=False)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
