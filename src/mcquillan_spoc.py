"""
Alan Xia's suggestion (2026-10-30): McQuillan, Aigrain & Mazeh 2013 (MNRAS 432, 1203), Kepler M dwarfs, 1570 periods.
How many of them have SPOC 2-min light curves, per 2-day period bin?
Table 2 comes from the arXiv ancillary files (arXiv 1303.6787, anc/table2_periodic.txt), saved in data/mcquillan/;
VizieR does not answer from this machine.
KIC --> TIC through the TIC v8.2 KIC column on MAST, then the same SPOC 2-min filter as Step 1.
Writes results/mcquillan_spoc.csv, results/mcquillan_spoc_summary.csv and results/mcquillan_truth_table.csv
(covered stars at 8-30 d, Step 1 columns, input to step2_download.py / step2_ls.py --truth).
"""

import logging
import sys

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from tqdm.auto import tqdm

from step1_truth_table import cache_dir, network_retry, period_bins, period_labels, repo, results_dir, spoc_2min_sectors

table_path = repo / "data" / "mcquillan" / "table2_periodic.txt"
tic_cache_path = cache_dir / "mcquillan_kic_tic.csv"
truth_table_path = results_dir / "mcquillan_truth_table.csv"
truth_range = (8.0, 30.0)  # d, same span as granular_mearth.py
tic_batch_size = 200
bin_edges = np.array([0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 70])  # d; 2-day bins match granular_mearth.py

logger = logging.getLogger("mcquillan")


@network_retry
def query_tic_batch(kic_ids: list[int]) -> pd.DataFrame:
    result = Catalogs.query_criteria(catalog="Tic", KIC=[str(k) for k in kic_ids])
    return result.to_pandas()[["ID", "KIC", "Tmag", "disposition"]]


def kic_to_tic(kic_ids: list[int]) -> pd.DataFrame:
    """
    One TIC per KIC: drop TIC rows flagged ARTIFACT/DUPLICATE/SPLIT, then keep the brightest if several remain.
    Cached to disk; a batch that fails after retries is logged and retried on the next run.
    """
    if tic_cache_path.exists():
        cache = pd.read_csv(tic_cache_path)
    else:
        cache = pd.DataFrame(columns=["ID", "KIC", "Tmag", "disposition"])
    seen = set(cache["KIC"].dropna().astype(int))
    todo = [k for k in kic_ids if k not in seen]
    batches = [todo[i:i + tic_batch_size] for i in range(0, len(todo), tic_batch_size)]
    for batch in tqdm(batches, desc="MAST TIC query by KIC", total=len(batches)):
        try:
            found = query_tic_batch(batch)
        except Exception as exc:
            logger.error(f"TIC batch starting at KIC {batch[0]} failed after retries: {exc!r}")
            continue
        cache = pd.concat([cache, found], ignore_index=True)
        cache.to_csv(tic_cache_path, index=False)

    cache["KIC"] = cache["KIC"].astype(int)
    clean = cache[cache["disposition"].isna()].sort_values("Tmag")
    clean = clean.drop_duplicates("KIC")
    return clean.rename(columns={"ID": "tic_id", "KIC": "kic", "Tmag": "tmag"})[["kic", "tic_id", "tmag"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    stars = pd.read_csv(table_path, comment="#").rename(columns={"KID": "kic", "Period": "period", "Flag": "flag"})

    tic = kic_to_tic(stars["kic"].tolist())
    stars = stars.merge(tic, on="kic", how="left")
    logger.info(f"{stars['tic_id'].isna().sum()} of {len(stars)} KIC IDs have no clean TIC match")
    stars = stars[stars["tic_id"].notna()].copy()
    stars["tic_id"] = stars["tic_id"].astype(int)

    sectors = spoc_2min_sectors(stars["tic_id"].tolist(), cache_dir / "mcquillan_spoc_2min.csv")
    per_star = sectors.groupby("tic_id")["sector"].apply(lambda s: ";".join(str(x) for x in sorted(s)))
    stars["spoc_sectors"] = stars["tic_id"].map(per_star).fillna("")
    stars["n_spoc_sectors"] = stars["spoc_sectors"].str.count(";") + (stars["spoc_sectors"] != "")
    stars["bin_low"] = bin_edges[np.digitize(stars["period"], bin_edges) - 1]
    stars.to_csv(results_dir / "mcquillan_spoc.csv", index=False)

    # Step-1-format truth table for step2_download.py / step2_ls.py --truth: covered stars at 8-30 d only
    truth = stars[(stars["period"] >= truth_range[0]) & (stars["period"] < truth_range[1]) & (stars["n_spoc_sectors"] > 0)].copy()
    truth["sample"] = "mcquillan"
    truth["period_bin"] = pd.cut(truth["period"], bins=period_bins, labels=period_labels, right=False)
    truth[["sample", "kic", "tic_id", "tmag", "period", "period_bin", "flag", "spoc_sectors", "n_spoc_sectors"]].to_csv(truth_table_path, index=False)
    logger.info(f"{len(truth)} stars, {truth['n_spoc_sectors'].sum()} star-sectors written to {truth_table_path.name}")

    rows = []
    for bin_low, group in stars.groupby("bin_low"):
        covered = group[group["n_spoc_sectors"] > 0]
        rows.append({
            "bin_low": bin_low,
            "bin_high": bin_edges[np.searchsorted(bin_edges, bin_low) + 1],
            "n_stars": len(group),
            "n_with_spoc_2min": len(covered),
            "n_star_sectors": int(covered["n_spoc_sectors"].sum()),
            "n_with_spoc_2min_flag_NF": int((covered["flag"] == "NF").sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "mcquillan_spoc_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
