"""
Step 2 of docs/ROADMAP.md, part 1: download the SPOC 2-min light-curve FITS for every star-sector in the Step 1 table.
Files are kept in data/fits/ (Step 3 needs SAP_FLUX from the same files).
Progress goes to data/fits/progress.csv with status done / error, so --resume-style reruns only fetch what is missing.
Per-file bytes and seconds are recorded so storage and time can be extrapolated for Theissen.
"""

import argparse
import logging
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

repo = Path(__file__).resolve().parents[1]
truth_path = repo / "results" / "step1_truth_table.csv"
obs_path = repo / "data" / "step1" / "mast_spoc_2min.csv"
fits_dir = repo / "data" / "fits"
progress_path = fits_dir / "progress.csv"
mast_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{obs_id}_lc.fits"
flush_every = 25

logger = logging.getLogger("step2_download")


@retry(
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, socket.timeout)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=5, max=60),
    reraise=True,
)
def fetch(url: str, dest: Path) -> int:
    """
    Stream one file to a temporary name, then rename, so a killed download never leaves a truncated .fits behind.
    Returns the number of bytes written.
    """
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def file_is_valid(path: Path) -> bool:
    """
    A downloaded light curve counts as valid if astropy opens it and it has the LIGHTCURVE extension.
    """
    if not path.exists():
        return False
    try:
        with fits.open(path) as hdul:
            return "LIGHTCURVE" in hdul and len(hdul["LIGHTCURVE"].data) > 0
    except (OSError, ValueError, KeyError):
        return False


def download_one(tic_id: int, sector: int, obs_id: str) -> dict:
    """
    Worker: fetch and verify one star-sector; never raises, returns a progress row instead.
    """
    socket.setdefaulttimeout(300)
    dest = fits_dir / f"{obs_id}_lc.fits"
    start = time.perf_counter()
    try:
        n_bytes = fetch(mast_url.format(obs_id=obs_id), dest)
        if not file_is_valid(dest):
            raise ValueError("downloaded file does not open as a SPOC light curve")
        status = "done"
        error_msg = ""
    except Exception as exc:
        n_bytes = 0
        status = "error"
        error_msg = repr(exc)[:500]
    return {
        "tic_id": tic_id,
        "sector": sector,
        "obs_id": obs_id,
        "path": str(dest.relative_to(repo)),
        "bytes": n_bytes,
        "seconds": time.perf_counter() - start,
        "status": status,
        "error_msg": error_msg,
    }


def select_jobs(n_stars: int | None, seed: int, truth_path: Path, obs_path: Path) -> pd.DataFrame:
    """
    Star-sectors to download: all covered stars, or a pilot of n_stars stars drawn evenly across samples and period bins.
    """
    truth = pd.read_csv(truth_path)
    truth = truth[truth["n_spoc_sectors"] > 0]
    if n_stars is not None:
        groups = truth.groupby(["sample", "period_bin"])
        per_group = -(-n_stars // groups.ngroups)  # ceil, so the pilot has at least n_stars stars
        picks = []
        for _, group in groups:
            picks.append(group.sample(n=min(per_group, len(group)), random_state=seed))
        truth = pd.concat(picks)
    obs = pd.read_csv(obs_path)
    obs = obs[obs["status"] == "done"].drop_duplicates(["tic_id", "sector"])
    return obs[obs["tic_id"].isin(truth["tic_id"])][["tic_id", "sector", "obs_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-stars", type=int, default=None, help="download only this many stars")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth", type=Path, default=truth_path, help="truth table with tic_id, sample, period_bin, n_spoc_sectors")
    parser.add_argument("--obs", type=Path, default=obs_path, help="MAST SPOC 2-min cache matching --truth")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    fits_dir.mkdir(parents=True, exist_ok=True)

    jobs = select_jobs(args.pilot_stars, args.seed, args.truth, args.obs)
    if progress_path.exists():
        progress = pd.read_csv(progress_path, keep_default_na=False)
    else:
        progress = pd.DataFrame(columns=["tic_id", "sector", "obs_id", "path", "bytes", "seconds", "status", "error_msg"])

    done_ids = set()
    for _, row in progress[progress["status"] == "done"].iterrows():
        if (repo / row["path"]).exists():
            done_ids.add(row["obs_id"])
    todo = jobs[~jobs["obs_id"].isin(done_ids)]
    logger.info(f"{len(jobs)} star-sectors requested, {len(jobs) - len(todo)} already on disk, {len(todo)} to fetch")

    new_rows = []
    wall_start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for _, row in todo.iterrows():
                futures.append(pool.submit(download_one, int(row["tic_id"]), int(row["sector"]), row["obs_id"]))
            for i, future in enumerate(tqdm(as_completed(futures), desc="download SPOC FITS", total=len(futures))):
                new_rows.append(future.result())
                if (i + 1) % flush_every == 0:
                    progress = pd.concat([progress, pd.DataFrame(new_rows)], ignore_index=True)
                    progress.drop_duplicates("obs_id", keep="last").to_csv(progress_path, index=False)
                    new_rows = []
    finally:
        if new_rows:
            progress = pd.concat([progress, pd.DataFrame(new_rows)], ignore_index=True)
        progress = progress.drop_duplicates("obs_id", keep="last")
        progress.to_csv(progress_path, index=False)

    wall = time.perf_counter() - wall_start
    this_run = progress[progress["obs_id"].isin(todo["obs_id"])]
    ok = this_run[this_run["status"] == "done"]
    n_error = (this_run["status"] == "error").sum()
    logger.info(f"this run: {len(ok)} done, {n_error} error, wall {wall} s with {args.workers} workers")
    if len(ok):
        mb = ok["bytes"].astype(float) / 1e6
        logger.info(f"MB per star-sector: median {np.median(mb)}, mean {mb.mean()}; seconds per file (one worker): median {ok['seconds'].astype(float).median()}")


if __name__ == "__main__":
    main()
