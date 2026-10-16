"""
Chance baseline for Steps 2 and 3: how often would a periodogram peak land within 10% of a MEarth period by luck?
Truth periods are shuffled among stars of the same period bin (1000 times) and the hit rate recomputed.
Needed because SAP peaks pile up at long periods, and long truth periods have wide 10% windows.
Writes results/step3_chance_baseline.csv.
"""

import numpy as np
import pandas as pd

from step2_ls import results_dir, tolerance

n_shuffles = 1000


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for flux in ["pdcsap", "sap"]:
        table = pd.read_csv(results_dir / f"ls_{flux}.csv", dtype={"sector": str})
        table = table[(table["status"] == "done") & table["sample"].str.startswith("mearth")]
        for (period_bin, mode), group in table.groupby(["period_bin", "mode"]):
            peak = group["peak_period"].values
            observed = np.mean(np.abs(peak - group["period"].values) < tolerance * group["period"].values)
            truth_by_star = group.groupby("tic_id")["period"].first()
            shuffled_rates = []
            for _ in range(n_shuffles):
                shuffled = pd.Series(rng.permutation(truth_by_star.values), index=truth_by_star.index)
                truth = group["tic_id"].map(shuffled).values  # all sectors of a star share one shuffled period
                shuffled_rates.append(np.mean(np.abs(peak - truth) < tolerance * truth))
            shuffled_rates = np.array(shuffled_rates)
            rows.append({
                "flux": flux,
                "period_bin": period_bin,
                "mode": mode,
                "n": len(group),
                "observed_hit_rate": observed,
                "chance_hit_rate_mean": shuffled_rates.mean(),
                "chance_hit_rate_95pct": np.quantile(shuffled_rates, 0.95),
                "excess_over_chance": observed - shuffled_rates.mean(),
                "p_value": (np.sum(shuffled_rates >= observed) + 1) / (n_shuffles + 1),
            })
    result = pd.DataFrame(rows).sort_values(["mode", "period_bin", "flux"])
    result.to_csv(results_dir / "step3_chance_baseline.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
