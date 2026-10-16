"""
Steps 2 and 3 of docs/ROADMAP.md: Lomb-Scargle baseline on SPOC 2-min light curves.
--flux pdcsap is Step 2, --flux sap is Step 3; same stars, same code, only the flux column changes.
Two modes per star:
- sector: one periodogram per sector.
- combined: all sectors together, with a free flux offset per sector (Theissen: sectors have flux offsets).
Both use the same floating-offset LS: power = fraction of variance, after removing per-sector means, explained by a sinusoid.
Writes results/ls_{flux}.csv (one row per star-sector plus one combined row per star) and results/ls_{flux}_summary.csv.
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm

repo = Path(__file__).resolve().parents[1]
truth_path = repo / "results" / "step1_truth_table.csv"
progress_path = repo / "data" / "fits" / "progress.csv"
results_dir = repo / "results"

bin_days = 30 / 1440  # 30-min bins
min_points = 100  # bins per sector; fewer and the sector is skipped
min_period, max_period = 0.1, 100.0  # d
samples_per_peak = {"sector": 50, "combined": 5}  # dense grid for one sector, else 27-100 d has only a few grid points
orbit_period = 13.7  # d, TESS orbit
orbit_window = 1.5  # d, a peak within this of 13.7 d is flagged as a likely systematic
tolerance = 0.1  # a recovered period counts if within 10% of truth
chunk_cells = 4e6  # frequencies x points per numpy chunk; ~32 MB per array (2e7 ran the machine out of RAM at 6 workers)
output_columns = [
    "tic_id", "sample", "period", "period_bin", "mode", "sector", "status", "error_msg",
    "peak_period", "peak_power", "peak_at_max_period", "near_orbit", "outcome",
    "n_bins", "baseline_days", "n_sectors", "seconds",
]  # fixed order, so per-star rows appended to the checkpoint stay aligned
period_bins = [0.0, 13.7, 27.0, np.inf]
period_labels = ["<13.7", "13.7-27", ">27"]

logger = logging.getLogger("step2_ls")


def read_sector(path: Path, flux: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one SPOC light curve, keep QUALITY == 0 and finite values, divide by the median, bin to 30 min.
    Returns (time in BTJD days, relative flux - 1), both per bin.
    """
    with fits.open(path) as hdul:
        data = hdul["LIGHTCURVE"].data
        names = {}
        for name in data.columns.names:
            names[name.lower()] = name
        t = np.asarray(data[names["time"]], dtype=float)
        f = np.asarray(data[names[f"{flux}_flux"]], dtype=float)
        q = np.asarray(data[names["quality"]])
    good = (q == 0) & np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if len(t) == 0:
        return t, f
    f = f / np.median(f) - 1.0
    idx = np.floor((t - t[0]) / bin_days).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    t_bin = np.bincount(idx, weights=t)[keep] / counts[keep]
    f_bin = np.bincount(idx, weights=f)[keep] / counts[keep]
    return t_bin, f_bin


def offset_ls(t: np.ndarray, y: np.ndarray, segment: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Lomb-Scargle with one free mean per segment (sector).
    For each frequency: regress y on [sin, cos] after subtracting per-segment means from y, sin and cos
    (equivalent to fitting the offsets jointly, by the Frisch-Waugh theorem).
    Returns power in [0, 1] = explained variance / total variance after offsets.
    """
    n_seg = segment.max() + 1
    onehot = np.zeros((len(t), n_seg))
    onehot[np.arange(len(t)), segment] = 1.0
    counts = onehot.sum(axis=0)  # (K,)
    y = y - (y @ onehot / counts)[segment]  # (N,)
    total = y @ y
    power = np.empty(len(freqs))
    step = max(1, int(chunk_cells // len(t)))
    for start in range(0, len(freqs), step):
        f = freqs[start:start + step]
        phase = 2 * np.pi * f[:, None] * t[None, :]  # (F, N)
        s = np.sin(phase)
        c = np.cos(phase)
        s -= (s @ onehot / counts)[:, segment]
        c -= (c @ onehot / counts)[:, segment]
        ss = np.einsum("fn,fn->f", s, s)
        cc = np.einsum("fn,fn->f", c, c)
        sc = np.einsum("fn,fn->f", s, c)
        sy = s @ y
        cy = c @ y
        det = ss * cc - sc ** 2
        explained = (cc * sy ** 2 - 2 * sc * sy * cy + ss * cy ** 2) / det
        power[start:start + step] = explained / total
    return power


def frequency_grid(baseline: float, mode: str) -> np.ndarray:
    """
    Uniform frequency grid from 1/100 to 1/0.1 per day, step 1/(samples_per_peak * baseline).
    """
    df = 1.0 / (samples_per_peak[mode] * baseline)
    return np.arange(1.0 / max_period, 1.0 / min_period + df, df)


def classify(peak: float, truth: float) -> str:
    """
    Score a peak against the truth period: hit, half (P/2 alias), double (2P alias) or miss.
    """
    if abs(peak - truth) < tolerance * truth:
        return "hit"
    if abs(peak - truth / 2) < tolerance * truth / 2:
        return "half"
    if abs(peak - 2 * truth) < tolerance * 2 * truth:
        return "double"
    return "miss"


def periodogram_row(t: np.ndarray, y: np.ndarray, segment: np.ndarray, mode: str, truth: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Run one periodogram and summarize its highest peak.
    """
    freqs = frequency_grid(t.max() - t.min(), mode)
    power = offset_ls(t, y, segment, freqs)
    best = int(np.argmax(power))
    peak = 1.0 / freqs[best]
    row = {
        "peak_period": peak,
        "peak_power": power[best],
        "peak_at_max_period": bool(best == 0),
        "near_orbit": bool(abs(peak - orbit_period) < orbit_window),
        "outcome": classify(peak, truth),
        "n_bins": len(t),
        "baseline_days": t.max() - t.min(),
    }
    return row, 1.0 / freqs, power


def process_star(star: dict, files: list[tuple[int, str]], flux: str) -> list[dict]:
    """
    Worker: all sectors of one star, one periodogram each, then the combined periodogram.
    """
    rows = []
    ts, ys, segs = [], [], []
    for sector, path in sorted(files):
        start = time.perf_counter()
        base = {"tic_id": star["tic_id"], "sample": star["sample"], "period": star["period"], "period_bin": star["period_bin"], "mode": "sector", "sector": str(sector)}
        try:
            t, y = read_sector(repo / path, flux)
        except (OSError, KeyError, ValueError) as exc:
            rows.append({**base, "status": "error", "error_msg": repr(exc)[:300]})
            continue
        if len(t) < min_points:
            rows.append({**base, "status": "too_few_points", "n_bins": len(t)})
            continue
        row, _, _ = periodogram_row(t, y, np.zeros(len(t), dtype=int), "sector", star["period"])
        rows.append({**base, **row, "status": "done", "seconds": time.perf_counter() - start})
        ts.append(t)
        ys.append(y)
        segs.append(np.full(len(t), len(segs)))

    base = {"tic_id": star["tic_id"], "sample": star["sample"], "period": star["period"], "period_bin": star["period_bin"], "mode": "combined", "sector": "all"}
    if len(ts) == 0:
        rows.append({**base, "status": "no_good_sector"})
        return rows
    start = time.perf_counter()
    t = np.concatenate(ts)
    order = np.argsort(t)
    row, _, _ = periodogram_row(t[order], np.concatenate(ys)[order], np.concatenate(segs)[order], "combined", star["period"])
    rows.append({**base, **row, "n_sectors": len(ts), "status": "done", "seconds": time.perf_counter() - start})
    return rows


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """
    Fractions of hit / half / double / near-13.7 d peaks per truth source, period bin and mode.
    Sector mode counts star-sectors; combined mode counts stars.
    """
    done = table[table["status"] == "done"].copy()
    done["truth"] = np.where(done["sample"] == "tars", "TARS", "MEarth")
    rows = []
    for (truth, period_bin, mode), group in done.groupby(["truth", "period_bin", "mode"], sort=False):
        rows.append({
            "truth": truth,
            "period_bin": period_bin,
            "mode": mode,
            "n": len(group),
            "frac_within_10pct": (group["outcome"] == "hit").mean(),
            "frac_half_alias": (group["outcome"] == "half").mean(),
            "frac_double_alias": (group["outcome"] == "double").mean(),
            "frac_peak_near_13.7d": group["near_orbit"].mean(),
            "frac_peak_at_100d_edge": group["peak_at_max_period"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary["period_bin"] = pd.Categorical(summary["period_bin"], categories=period_labels, ordered=True)
    return summary.sort_values(["mode", "truth", "period_bin"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke")
    parser.add_argument("--limit-stars", type=int, default=None, help="only the first N stars (smoke test)")
    args = parser.parse_args()

    log_path = repo / "data" / f"ls_{args.flux}{args.tag}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    truth = pd.read_csv(truth_path)
    progress = pd.read_csv(progress_path, keep_default_na=False)
    progress = progress[progress["status"] == "done"]
    files_by_star: dict[int, list[tuple[int, str]]] = {}
    for _, row in progress.iterrows():
        files_by_star.setdefault(int(row["tic_id"]), []).append((int(row["sector"]), row["path"]))
    stars = truth[truth["tic_id"].isin(files_by_star.keys())]
    if args.limit_stars is not None:
        stars = stars.head(args.limit_stars)
    logger.info(f"{len(stars)} stars with downloaded sectors, {len(progress)} star-sectors, flux = {args.flux}")

    checkpoint = repo / "data" / f"ls_{args.flux}{args.tag}.partial.csv"
    if checkpoint.exists():
        finished = set(pd.read_csv(checkpoint, usecols=["tic_id"])["tic_id"])
        logger.info(f"resuming: {len(finished)} stars already in {checkpoint.name}")
    else:
        finished = set()
    todo = stars[~stars["tic_id"].isin(finished)]
    logger.info(f"{len(todo)} stars to run; stars with many sectors take minutes each, so the bar can pause")

    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for star in todo.to_dict("records"):
            futures.append(pool.submit(process_star, star, files_by_star[star["tic_id"]], args.flux))
        for future in tqdm(as_completed(futures), desc=f"LS {args.flux}", total=len(futures)):
            rows = pd.DataFrame(future.result()).reindex(columns=output_columns)
            rows.to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
    wall = time.perf_counter() - wall_start

    table = pd.read_csv(checkpoint, dtype={"sector": str})
    table = table[table["tic_id"].isin(stars["tic_id"])]
    table.to_csv(results_dir / f"ls_{args.flux}{args.tag}.csv", index=False)
    summary = summarize(table)
    summary.to_csv(results_dir / f"ls_{args.flux}{args.tag}_summary.csv", index=False)
    logger.info("summary:\n" + summary.to_string(index=False))

    done = table[table["status"] == "done"]
    logger.info(f"status counts: {table['status'].value_counts().to_dict()}")
    logger.info(f"wall {wall} s on {args.workers} workers; CPU s per star-sector (median) {done[done['mode'] == 'sector']['seconds'].median()}, per combined fit (median / max) {done[done['mode'] == 'combined']['seconds'].median()} / {done[done['mode'] == 'combined']['seconds'].max()}")


if __name__ == "__main__":
    main()
