"""
Theissen 2026-10-16: where exactly does the LS hit rate drop past 13.7 d? MEarth has 0-6 stars per 2-day bin there (granular_mearth.py),
so the period axis is filled by injection-recovery instead: add a sinusoid of known period to real SPOC light curves and rerun the Step 2 LS.
Hosts: MEarth stars with P > 27 d (the target population; their own signal is recovered in ~2% of single sectors).
Injection: relative flux (1 + y) * (1 + A sin(2 pi t / P + phi)) - 1 on the 30-min bins from step2_ls.read_sector, phi random per (host, P),
same phi for PDCSAP and SAP so the two are paired. A = 0 is the control: how often the peak lands within 10% of P with nothing injected.
Limitation: injecting into PDCSAP happens after PDC, so this cannot measure signal that PDC itself removes; SAP injection is not affected.
Writes results/injection_recovery.csv, results/injection_recovery_summary.csv, results/plots/injection_recovery.png.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from granular_mearth import wilson
from step2_ls import min_points, orbit_period, periodogram_row, progress_path, read_sector, repo, results_dir, truth_path

inject_periods = np.concatenate([[3.0], np.arange(8.0, 20.01, 0.5)])  # d; 3 d is the sanity point (should be ~100%)
amplitudes = {"sector": [0.0, 0.001, 0.002, 0.005, 0.01], "combined": [0.0, 0.002, 0.01]}  # relative-flux semi-amplitude
fluxes = ["pdcsap", "sap"]
output_columns = ["tic_id", "mode", "sector", "flux", "amp", "inj_period", "peak_period", "peak_power", "peak_at_max_period", "near_orbit", "outcome", "n_bins", "baseline_days"]

logger = logging.getLogger("injection_recovery")


def inject(t: np.ndarray, y: np.ndarray, amp: float, period: float, phase: float) -> np.ndarray:
    return (1 + y) * (1 + amp * np.sin(2 * np.pi * t / period + phase)) - 1


def host_phases(seed: int) -> np.ndarray:
    """
    One random phase per injected period; run_host and example_figure share it so an example reproduces its csv row.
    """
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, len(inject_periods))


def host_seed(tic_id: int) -> int:
    return 1000 + tic_id % 100000


def run_host(tic_id: int, mode: str, files: list[tuple[int, str]], seed: int) -> list[dict]:
    """
    Worker: one host (a single star-sector in sector mode, all listed sectors of a star in combined mode).
    Reads both fluxes once, then one periodogram per (flux, amplitude, injected period).
    """
    phases = host_phases(seed)
    sector_label = str(files[0][0]) if mode == "sector" else "all"
    rows = []
    for flux in fluxes:
        ts, ys, segs = [], [], []
        for _, path in sorted(files):
            t, y = read_sector(repo / path, flux)
            if len(t) < min_points:
                continue
            ts.append(t)
            ys.append(y)
            segs.append(np.full(len(t), len(segs)))
        if len(ts) == 0:
            continue
        t = np.concatenate(ts)
        order = np.argsort(t)
        t, y, segment = t[order], np.concatenate(ys)[order], np.concatenate(segs)[order]
        for amp in amplitudes[mode]:
            for period, phase in zip(inject_periods, phases):
                row, _, _ = periodogram_row(t, inject(t, y, amp, period, phase), segment, mode, period)
                rows.append({"tic_id": tic_id, "mode": mode, "sector": sector_label, "flux": flux, "amp": amp, "inj_period": period, **row})
    return rows


def pick_hosts(n_sector: int, n_combined: int, max_combined_sectors: int) -> list[tuple[int, str, list[tuple[int, str]]]]:
    """
    Random hosts (seed 0) among MEarth P > 27 d stars with downloaded SPOC sectors.
    Combined hosts need 2 to max_combined_sectors sectors (one sector would repeat sector mode; many sectors make the LS grid huge).
    """
    truth = pd.read_csv(truth_path)
    hosts = set(truth[truth["sample"].str.startswith("mearth") & (truth["period"] > 27)]["tic_id"])
    progress = pd.read_csv(progress_path, keep_default_na=False)
    progress = progress[(progress["status"] == "done") & progress["tic_id"].isin(hosts)]
    rng = np.random.default_rng(0)
    star_sectors = progress.iloc[rng.permutation(len(progress))[:n_sector]]
    picks = []
    for row in star_sectors.itertuples():
        picks.append((int(row.tic_id), "sector", [(int(row.sector), row.path)]))
    by_star = progress.groupby("tic_id")
    eligible = []
    for tic_id, group in by_star:
        if 2 <= len(group) <= max_combined_sectors:
            eligible.append(tic_id)
    for tic_id in rng.permutation(np.array(eligible))[:n_combined]:
        group = by_star.get_group(tic_id)
        picks.append((int(tic_id), "combined", list(zip(group["sector"].astype(int), group["path"]))))
    return picks


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, flux, amp, period), group in table.groupby(["mode", "flux", "amp", "inj_period"]):
        hits = int((group["outcome"] == "hit").sum())
        low, high = wilson(hits, len(group))
        rows.append({
            "mode": mode,
            "flux": flux,
            "amp": amp,
            "inj_period": period,
            "n": len(group),
            "recovery_rate": hits / len(group),
            "wilson68_low": low,
            "wilson68_high": high,
            "frac_peak_near_13.7d": group["near_orbit"].mean(),
            "frac_peak_at_100d_edge": group["peak_at_max_period"].mean(),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> plt.Figure:
    """
    Recovery rate vs injected period, one panel per mode; MEarth 2-day bins (granular_mearth.csv) as open markers for comparison.
    """
    mearth_path = results_dir / "granular_mearth.csv"
    mearth = pd.read_csv(mearth_path) if mearth_path.exists() else None
    colors = {"pdcsap": "#2a78d6", "sap": "#eb6834"}  # same flux colours as granular_mearth.py
    styles = {0.01: "-", 0.002: "--", 0.0: ":"}  # plotted amplitudes; 0.1% and 0.5% stay in the summary csv
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4), sharey=True)
    for ax, mode in zip(axes, ["sector", "combined"]):
        part = summary[(summary["mode"] == mode) & (summary["inj_period"] >= 8)]
        for flux in fluxes:
            for amp in styles:
                line = part[(part["flux"] == flux) & (part["amp"] == amp)]
                label = f"{flux.upper()}, A = {round(100 * amp, 1)}%" if amp > 0 else f"{flux.upper()}, nothing injected"
                ax.plot(line["inj_period"], line["recovery_rate"], linestyle=styles[amp], color=colors[flux], linewidth=1.4, label=label)
        if mearth is not None:
            m = mearth[(mearth["mode"] == mode) & (mearth["bin_high"] <= 22)]
            for flux, marker in [("pdcsap", "s"), ("sap", "D")]:
                mf = m[m["flux"] == flux]
                ax.plot(mf["bin_low"] + 1, mf["hit_rate"], marker, color=colors[flux], markerfacecolor="none", markersize=6, label=f"MEarth real stars, {flux.upper()}")
        ax.axvline(orbit_period, color="#52514e", linewidth=0.8, linestyle="--")
        n_hosts = int(part["n"].max()) if len(part) else 0
        ax.set_title(f"{mode} mode, {n_hosts} hosts", fontsize=10)
        ax.set_xlabel("period (d); dashed = 13.7 d")
        ax.set_ylim(-0.03, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("LS peak within 10% of the period")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def example_figure(tic_id: int, path: str, period: float, amp: float) -> plt.Figure:
    """
    One host sector, one injection (same phase as in the run): light curve before / after injection (left) and the two periodograms (right),
    one row per flux. Dotted vertical line = injected period.
    """
    colors = {"before": "#52514e", "after": "#eb6834"}
    phase = host_phases(host_seed(tic_id))[int(np.argmin(np.abs(inject_periods - period)))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    for row, flux in zip(axes, fluxes):
        t, y = read_sector(repo / path, flux)
        segment = np.zeros(len(t), dtype=int)
        curves = {"before": y, "after": inject(t, y, amp, period, phase)}
        for name, curve in curves.items():
            row[0].plot(t - t[0], curve, ".", markersize=1.5, color=colors[name], label=f"{name} injection")
            _, periods, power = periodogram_row(t, curve, segment, "sector", period)
            row[1].plot(periods, power, color=colors[name], linewidth=1)
        row[1].axvline(period, color="black", linewidth=0.8, linestyle=":")
        row[1].axvline(orbit_period, color="#52514e", linewidth=0.8, linestyle="--")
        row[1].set_xscale("log")
        row[1].set_xlim(0.5, 100)
        row[0].set_ylabel(f"{flux.upper()} relative flux")
        row[1].set_ylabel("LS power")
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8, markerscale=6)
    axes[1][0].set_xlabel("days since sector start")
    axes[1][1].set_xlabel("period (d); dotted = injected, dashed = 13.7 d")
    fig.suptitle(f"{Path(path).name}: injected P = {period} d, A = {round(100 * amp, 1)}%", fontsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sector", type=int, default=120, help="host star-sectors for sector mode")
    parser.add_argument("--n-combined", type=int, default=50, help="host stars for combined mode")
    parser.add_argument("--max-combined-sectors", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke")
    parser.add_argument("--summarize-only", action="store_true", help="skip unfinished hosts; summarize and plot what is in the checkpoint")
    args = parser.parse_args()

    log_path = repo / "data" / f"injection_recovery{args.tag}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    hosts = pick_hosts(args.n_sector, args.n_combined, args.max_combined_sectors)

    checkpoint = repo / "data" / f"injection_recovery{args.tag}.partial.csv"
    finished = set()
    if checkpoint.exists():
        done = pd.read_csv(checkpoint, usecols=["tic_id", "mode", "sector"], dtype={"sector": str})
        finished = set(zip(done["tic_id"], done["mode"], done["sector"]))
        logger.info(f"resuming: {len(finished)} hosts already in {checkpoint.name}")
    todo = []
    for tic_id, mode, files in hosts:
        sector_label = str(files[0][0]) if mode == "sector" else "all"
        if (tic_id, mode, sector_label) not in finished:
            todo.append((tic_id, mode, files))
    if args.summarize_only:
        logger.info(f"--summarize-only: skipping {len(todo)} unfinished hosts")
        todo = []
    logger.info(f"{len(hosts)} hosts, {len(todo)} to run; {len(inject_periods)} periods per amplitude; combined hosts take minutes each")

    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for tic_id, mode, files in todo:
            futures[pool.submit(run_host, tic_id, mode, files, host_seed(tic_id))] = (tic_id, mode)
        for future in tqdm(as_completed(futures), desc="inject", total=len(futures)):
            try:
                rows = future.result()
            except (OSError, KeyError, ValueError) as exc:
                logger.warning(f"host {futures[future]} failed: {exc!r}")
                continue
            pd.DataFrame(rows).reindex(columns=output_columns).to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
    logger.info(f"wall {time.perf_counter() - wall_start} s on {args.workers} workers")

    table = pd.read_csv(checkpoint, dtype={"sector": str})
    table.to_csv(results_dir / f"injection_recovery{args.tag}.csv", index=False)
    summary = summarize(table)
    summary.to_csv(results_dir / f"injection_recovery{args.tag}_summary.csv", index=False)
    logger.info("summary:\n" + summary.to_string(index=False))
    fig = plot(summary)
    fig.savefig(results_dir / "plots" / f"injection_recovery{args.tag}.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote results/injection_recovery{args.tag}*.csv and results/plots/injection_recovery{args.tag}.png")


if __name__ == "__main__":
    main()
