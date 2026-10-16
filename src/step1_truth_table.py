"""
Step 1 of docs/ROADMAP.md: build the truth table.
MEarth rotators (Newton+2016 north, Newton+2018 south, grades A and B) --> TIC IDs --> SPOC 2-min sectors.
A random TARS sample with P < 10 d is added as a short-period control.
Writes results/step1_truth_table.csv and results/step1_summary.csv.
"""

import argparse
import logging
import socket
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astroquery.mast import Observations
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

socket.setdefaulttimeout(300)

repo = Path(__file__).resolve().parents[1]
tars_path = Path(r"C:\git_repo\Stellar-World-Model\data\tars_table_2.feather")
cache_dir = repo / "data" / "step1"
results_dir = repo / "results"

mearth_catalogs = {
    "mearth_n": "J/ApJ/821/93",  # Newton+2016, table1
    "mearth_s": "J/AJ/156/217",  # Newton+2018, table1
}
tic_vizier = "vizier:IV/39/tic82"
match_radius_arcsec = 10.0
mast_batch_size = 100
tars_max_period = 10.0
tars_n_candidates = 1200  # only ~9% of TARS stars have SPOC 2-min data
tars_n_keep = 75
period_bins = [0.0, 13.7, 27.0, np.inf]  # d; 13.7 d = one TESS orbit
period_labels = ["<13.7", "13.7-27", ">27"]

logger = logging.getLogger("step1")

network_retry = retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, socket.timeout)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@network_retry
def fetch_vizier_table(catalog_id: str) -> pd.DataFrame:
    """
    Download table1 of one MEarth catalog from VizieR, selected by name, not position.
    """
    tables = Vizier(columns=["**"], row_limit=-1).get_catalogs(catalog_id)
    for table in tables:
        if table.meta.get("name", "").endswith("/table1"):
            return table.to_pandas()
    raise KeyError(f"No table1 in {catalog_id}; found {[t.meta.get('name') for t in tables]}")


def load_mearth() -> pd.DataFrame:
    """
    Load grade A and B rotators from both MEarth papers into one frame.
    Grade U (uncertain) and N (non-detection) are dropped: they carry no usable truth period.
    The contamination flag is kept so contaminated stars can be excluded later if they look odd.
    """
    frames = []
    for sample, catalog_id in mearth_catalogs.items():
        df = fetch_vizier_table(catalog_id)
        df = df[df["Type"].isin(["A", "B"])].copy()
        if "Cont" in df.columns:
            contaminated = df["Cont"].notna()
        else:
            contaminated = df["Flag"].notna()
        frames.append(pd.DataFrame({
            "sample": sample,
            "twomass": df["2MASS"].str.lstrip("J").str.strip(),
            "ra": df["RAJ2000"],
            "dec": df["DEJ2000"],
            "period": df["Per"],
            "grade": df["Type"],
            "amp_mag": df["Amp"],
            "contaminated": contaminated,
        }))
        logger.info(f"{sample} ({catalog_id}): {len(df)} grade A/B rotators")
    mearth = pd.concat(frames, ignore_index=True)

    duplicated = mearth["twomass"].duplicated(keep="first")
    if duplicated.any():
        logger.warning(f"{duplicated.sum()} stars appear in both MEarth papers; keeping the north entry")
    return mearth[~duplicated].reset_index(drop=True)


@network_retry
def xmatch_tic(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-match star positions against TIC v8.2 with the CDS XMatch service.
    Both catalogs give J2000 positions, so proper motion only shifts matches by under a few arcsec.
    """
    from astropy.table import Table

    table = Table.from_pandas(stars[["twomass", "ra", "dec"]])
    result = XMatch.query(
        cat1=table,
        cat2=tic_vizier,
        max_distance=match_radius_arcsec * u.arcsec,
        colRA1="ra",
        colDec1="dec",
    )
    return result.to_pandas()


def pick_tic(stars: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Choose one TIC entry per MEarth star.
    Preferred: the TIC row carrying the same 2MASS ID, not flagged ARTIFACT/DUPLICATE/SPLIT, closest first.
    Fallback when the 2MASS-ID row is flagged (TIC v8 replaced it): the closest unflagged row,
    accepted only if within 3 arcsec and 1 mag of the flagged row, marked match_flag = "position_only".
    Without this check the fallback picked Tmag 17-19 background stars next to Tmag 12 M dwarfs.
    Also counts other unflagged TIC sources within the radius and 3 mag, a rough blending warning.
    """
    matches = matches.copy()
    matches["same_2mass"] = matches["2MASS"].fillna("").str.strip() == matches["twomass"]
    flagged = matches["Disp"].fillna("").str.strip() != ""
    flagged_tmag = matches[flagged & matches["same_2mass"]].groupby("twomass")["Tmag"].min()
    matches = matches[~flagged]
    matches = matches.sort_values(["twomass", "same_2mass", "angDist"], ascending=[True, False, True])

    rows = []
    for twomass, group in matches.groupby("twomass"):
        best = group.iloc[0]
        if best["same_2mass"]:
            flag = "2mass_id"
        else:
            ref_tmag = flagged_tmag.get(twomass, np.nan)
            if best["angDist"] > 3 or not abs(best["Tmag"] - ref_tmag) < 1:
                logger.info(f"{twomass}: rejected position-only TIC {int(best['TIC'])} (Tmag {best['Tmag']} vs {ref_tmag}, {best['angDist']} arcsec)")
                continue
            flag = "position_only"
        others = group.iloc[1:]
        n_neighbors = int((others["Tmag"] < best["Tmag"] + 3).sum())
        rows.append({
            "twomass": twomass,
            "tic_id": int(best["TIC"]),
            "tmag": best["Tmag"],
            "match_dist_arcsec": best["angDist"],
            "match_flag": flag,
            "n_tic_neighbors_10as": n_neighbors,
        })
    picked = pd.DataFrame(rows)
    out = stars.merge(picked, on="twomass", how="left")
    missing = out["tic_id"].isna().sum()
    if missing:
        logger.warning(f"{missing} MEarth stars have no TIC match within {match_radius_arcsec} arcsec")
    return out


@network_retry
def query_mast_batch(tic_ids: list[int]) -> pd.DataFrame:
    """
    One MAST Observations query for a batch of TIC IDs, TESS time series only.
    """
    targets = []
    for tic in tic_ids:
        targets.append(str(tic))
    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries", target_name=targets)
    return obs.to_pandas()


def spoc_2min_sectors(tic_ids: list[int], cache_path: Path) -> pd.DataFrame:
    """
    Return one row per (tic_id, sector) that has a single-sector SPOC 2-min light curve.
    Kept: provenance SPOC, 120 s cadence, obs_id ending in "-s".
    Dropped: 20 s "fast" files, TESS-SPOC FFI products, multi-sector DV products ("s0001-s0036").
    Batch results are cached to disk so a rerun only queries TIC IDs it has not seen;
    a batch that fails after all retries is logged and left out of the cache, so the next run retries it.
    """
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["tic_id", "sector", "obs_id", "status"])
    seen = set(cache["tic_id"].astype(int))

    todo = []
    for tic in tic_ids:
        if tic not in seen:
            todo.append(tic)
    batches = []
    for start in range(0, len(todo), mast_batch_size):
        batches.append(todo[start:start + mast_batch_size])

    for batch in tqdm(batches, desc="MAST SPOC 2-min query", total=len(batches)):
        try:
            obs = query_mast_batch(batch)
        except Exception as exc:
            logger.error(f"MAST batch starting at TIC {batch[0]} failed after retries: {exc!r}")
            continue
        keep = (
            (obs["provenance_name"] == "SPOC")
            & (obs["t_exptime"] == 120)
            & obs["obs_id"].str.endswith("-s")
        )
        obs = obs[keep]
        new_rows = []
        found = set()
        for _, row in obs.iterrows():
            tic = int(row["target_name"])
            found.add(tic)
            new_rows.append({"tic_id": tic, "sector": int(row["sequence_number"]), "obs_id": row["obs_id"], "status": "done"})
        for tic in batch:
            if tic not in found:
                new_rows.append({"tic_id": tic, "sector": -1, "obs_id": "", "status": "no_data"})
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)

    cache = cache[cache["tic_id"].isin(tic_ids)]
    return cache[cache["status"] == "done"].drop_duplicates(["tic_id", "sector"])


def attach_sectors(stars: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_spoc_sectors and a ";"-joined sector list to each star, plus its period bin.
    """
    grouped = sectors.sort_values("sector").groupby("tic_id")["sector"]
    sector_lists = {}
    for tic, values in grouped:
        sector_lists[tic] = ";".join(str(s) for s in values)
    stars = stars.copy()
    stars["spoc_sectors"] = stars["tic_id"].map(sector_lists).fillna("")
    counts = []
    for value in stars["spoc_sectors"]:
        if value:
            counts.append(len(value.split(";")))
        else:
            counts.append(0)
    stars["n_spoc_sectors"] = counts
    stars["period_bin"] = pd.cut(stars["period"], bins=period_bins, labels=period_labels, right=False)
    return stars


def load_tars_control(seed: int) -> pd.DataFrame:
    """
    Draw short-period control stars from TARS: P < 10 d, no multiple-period or binary flag.
    TARS is used only as short-period truth (see CLAUDE.md); a fixed seed keeps the draw reproducible.
    """
    tars = pd.read_feather(tars_path, columns=[
        "TICID", "ra", "dec", "Tmag", "adopted_period", "flag_multiple_periods", "flag_possible_binary",
    ])
    clean = tars[
        (tars["adopted_period"] < tars_max_period)
        & ~tars["flag_multiple_periods"].astype(bool)
        & ~tars["flag_possible_binary"].astype(bool)
    ]
    draw = clean.sample(n=tars_n_candidates, random_state=seed)
    return pd.DataFrame({
        "sample": "tars",
        "twomass": "",
        "ra": draw["ra"].values,
        "dec": draw["dec"].values,
        "period": draw["adopted_period"].values,
        "grade": "",
        "amp_mag": np.nan,
        "contaminated": False,
        "tic_id": draw["TICID"].astype(int).values,
        "tmag": draw["Tmag"].values,
        "match_dist_arcsec": 0.0,
        "match_flag": "tars_ticid",
        "n_tic_neighbors_10as": np.nan,
    })


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """
    Count stars per sample and period bin: all, with >= 1 SPOC 2-min sector, and median sectors among those.
    """
    rows = []
    for (sample, period_bin), group in table.groupby(["sample", "period_bin"], observed=False):
        covered = group[group["n_spoc_sectors"] > 0]
        rows.append({
            "sample": sample,
            "period_bin": period_bin,
            "n_stars": len(group),
            "n_with_spoc_2min": len(covered),
            "median_sectors_if_covered": covered["n_spoc_sectors"].median(),
            "max_sectors": covered["n_spoc_sectors"].max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mearth = load_mearth()
    matches = xmatch_tic(mearth)
    mearth = pick_tic(mearth, matches)
    logger.info(f"TIC match flags: {mearth['match_flag'].value_counts(dropna=False).to_dict()}")
    mearth = mearth[mearth["tic_id"].notna()].copy()
    mearth["tic_id"] = mearth["tic_id"].astype(int)

    tars = load_tars_control(args.seed)
    stars = pd.concat([mearth, tars], ignore_index=True)

    sectors = spoc_2min_sectors(stars["tic_id"].tolist(), cache_dir / "mast_spoc_2min.csv")
    stars = attach_sectors(stars, sectors)

    tars_covered = stars[(stars["sample"] == "tars") & (stars["n_spoc_sectors"] > 0)]
    tars_keep = tars_covered.head(tars_n_keep)
    logger.info(f"TARS control: {len(tars_covered)} of {tars_n_candidates} candidates have SPOC 2-min; keeping {len(tars_keep)}")
    table = pd.concat([stars[stars["sample"] != "tars"], tars_keep], ignore_index=True)

    table.to_csv(results_dir / "step1_truth_table.csv", index=False)
    summary = summarize(table)
    summary.to_csv(results_dir / "step1_summary.csv", index=False)
    print(summary.to_string(index=False))

    long_covered = table[table["sample"].str.startswith("mearth") & (table["period"] >= 13.7) & (table["n_spoc_sectors"] > 0)]
    print(f"MEarth stars with P >= 13.7 d and >= 1 SPOC 2-min sector: {len(long_covered)}")


if __name__ == "__main__":
    main()
