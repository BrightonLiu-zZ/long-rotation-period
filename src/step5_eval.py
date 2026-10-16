"""
Step 5 of docs/ROADMAP.md, evaluation on held-out stars (each star is scored by the fold model that never saw it).
Two separate questions, two separate tables (ROADMAP honesty rule: never mix them):
1. Forecast: half 1 --> half 2 of the same sector. Masked MAE in units of half 1's robust std, per method.
   Methods: mean (0), last value, persistence (copy half 1 forward 14 d), fitted sinusoid, L1-only generator, GAN (1 sample; mean of 8).
2. Extension + periodogram: continue each sector 2 x 14 d past its 28-day grid (autoregressive), then single-sector LS.
Generator outputs come from step5_generate.py (env swm); this script never imports torch (env astro: two OpenMP runtimes crash).
   Rows: (a) none, (b) fitted sinusoid, L1-only generator, GAN (1 seeded sample). Scored against MEarth / TARS like Step 2, plus a shuffle chance rate.
The extended part is model output, not data; any gain must beat row (b).
Writes results/step5_forecast.csv, step5_forecast_summary.csv, step5_ls_extension.csv, step5_ls_summary.csv, plots/step5_examples.png.
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from step2_ls import bin_days, frequency_grid, offset_ls, period_labels, periodogram_row, results_dir, tolerance
from step5_common import data_dir, half, load_data, normalise, repo

n_shuffles = 1000
forecast_methods = ["mean", "last_value", "persistence", "sinusoid", "l1", "gan_1", "gan_mean8"]
ls_methods = ["none", "sinusoid", "l1", "gan"]
ls_columns = ["tic_id", "sector", "sample", "period", "period_bin", "fold", "method", "peak_period", "peak_power", "peak_at_max_period", "near_orbit", "outcome", "n_bins", "baseline_days"]

logger = logging.getLogger("step5_eval")


def fit_sinusoid(t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Best single sinusoid: frequency = LS peak on the Step 2 grid, then least squares for [1, sin, cos] at that frequency.
    Returns (frequency per day, coefficients) for predict_sinusoid.
    """
    freqs = frequency_grid(t.max() - t.min(), "sector")
    power = offset_ls(t, y, np.zeros(len(t), dtype=int), freqs)
    freq = freqs[int(np.argmax(power))]
    design = np.stack([np.ones_like(t), np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return freq, coef


def predict_sinusoid(freq: float, coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * np.sin(2 * np.pi * freq * t) + coef[2] * np.cos(2 * np.pi * freq * t)


def score_sector(info: dict, flux: np.ndarray, mask: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """
    Worker: forecast errors and extended periodograms for one star-sector.
    Forecast error is in units of half 1's robust std (unclipped), averaged over observed bins of half 2.
    """
    grid_t = info["t0"] + np.arange(len(flux)) * bin_days
    centre, scale = normalise(flux[None, :half], mask[None, :half])
    centre, scale = centre[0], scale[0]
    t1, y1 = grid_t[:half][mask[:half]], flux[:half][mask[:half]]
    t2, y2 = grid_t[half:][mask[half:]], flux[half:][mask[half:]]

    persistence = flux[:half].copy()
    persistence[~mask[:half]] = centre
    freq, coef = fit_sinusoid(t1, y1)
    forecasts = {
        "mean": np.full(half, centre),
        "last_value": np.full(half, y1[-1]),
        "persistence": persistence,
        "sinusoid": predict_sinusoid(freq, coef, grid_t[half:]),
        "l1": preds["l1"],
        "gan_1": preds["gan_1"],
        "gan_mean8": preds["gan_mean8"],
    }
    keys = {"tic_id": info["tic_id"], "sector": info["sector"], "sample": info["sample"], "period": info["period"], "period_bin": info["period_bin"], "fold": info["fold"]}
    forecast_rows = []
    for method in forecast_methods:
        error = np.abs(forecasts[method][mask[half:]] - y2) / scale
        forecast_rows.append({**keys, "method": method, "mae": error.mean()})

    t_obs, y_obs = grid_t[mask], flux[mask]
    t_ext = info["t0"] + (len(flux) + np.arange(len(preds["ext_gan"]))) * bin_days
    freq, coef = fit_sinusoid(t_obs, y_obs)
    extensions = {
        "sinusoid": predict_sinusoid(freq, coef, t_ext),
        "l1": preds["ext_l1"],
        "gan": preds["ext_gan"],
    }
    ls_rows = []
    for method in ls_methods:
        if method == "none":
            t, y = t_obs, y_obs
        else:
            t = np.concatenate([t_obs, t_ext])
            y = np.concatenate([y_obs, extensions[method]])
        row, _, _ = periodogram_row(t, y, np.zeros(len(t), dtype=int), "sector", info["period"])
        ls_rows.append({**keys, "method": method, **row})
    return forecast_rows, ls_rows


def chance_rates(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle chance, as in step3_chance.py: MEarth periods permuted among stars of the same bin, hit rate recomputed 1000 times.
    One row per (method, period bin).
    """
    rows = []
    mearth = table[table["sample"].str.startswith("mearth")]
    for (method, period_bin), group in mearth.groupby(["method", "period_bin"]):
        peak = group["peak_period"].values
        truth_by_star = group.groupby("tic_id")["period"].first()
        shuffled_rates = []
        for _ in range(n_shuffles):
            shuffled = pd.Series(rng.permutation(truth_by_star.values), index=truth_by_star.index)
            truth = group["tic_id"].map(shuffled).values  # all sectors of a star share one shuffled period
            shuffled_rates.append(np.mean(np.abs(peak - truth) < tolerance * truth))
        shuffled_rates = np.array(shuffled_rates)
        observed = np.mean(group["outcome"] == "hit")
        rows.append({
            "truth": "MEarth",
            "method": method,
            "period_bin": period_bin,
            "chance_hit_rate_mean": shuffled_rates.mean(),
            "p_value": (np.sum(shuffled_rates >= observed) + 1) / (n_shuffles + 1),
        })
    return pd.DataFrame(rows)


def summarize(forecast: pd.DataFrame, ls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast: median and mean MAE per method, per truth source and period bin, plus an all-sectors row.
    LS: hit / alias / near-13.7 d fractions per method, truth source and period bin, joined with the chance rate.
    """
    forecast = forecast.copy()
    forecast["truth"] = np.where(forecast["sample"] == "tars", "TARS", "MEarth")
    everything = forecast.assign(truth="all", period_bin="all")
    rows = []
    for (truth, period_bin, method), group in pd.concat([forecast, everything]).groupby(["truth", "period_bin", "method"]):
        rows.append({"truth": truth, "period_bin": period_bin, "method": method, "n": len(group), "median_mae": group["mae"].median(), "mean_mae": group["mae"].mean()})
    forecast_summary = pd.DataFrame(rows)
    forecast_summary["method"] = pd.Categorical(forecast_summary["method"], categories=forecast_methods, ordered=True)
    forecast_summary = forecast_summary.sort_values(["truth", "period_bin", "method"]).reset_index(drop=True)

    ls = ls.copy()
    ls["truth"] = np.where(ls["sample"] == "tars", "TARS", "MEarth")
    rows = []
    for (truth, period_bin, method), group in ls.groupby(["truth", "period_bin", "method"]):
        rows.append({
            "truth": truth,
            "period_bin": period_bin,
            "method": method,
            "n": len(group),
            "frac_within_10pct": (group["outcome"] == "hit").mean(),
            "frac_half_alias": (group["outcome"] == "half").mean(),
            "frac_double_alias": (group["outcome"] == "double").mean(),
            "frac_peak_near_13.7d": group["near_orbit"].mean(),
            "frac_peak_at_100d_edge": group["peak_at_max_period"].mean(),
        })
    ls_summary = pd.DataFrame(rows).merge(chance_rates(ls, np.random.default_rng(0)), on=["truth", "method", "period_bin"], how="left")
    ls_summary["period_bin"] = pd.Categorical(ls_summary["period_bin"], categories=period_labels, ordered=True)
    ls_summary["method"] = pd.Categorical(ls_summary["method"], categories=ls_methods, ordered=True)
    ls_summary = ls_summary.sort_values(["truth", "period_bin", "method"]).reset_index(drop=True)
    return forecast_summary, ls_summary


def plot_examples(flux: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, preds: dict[str, np.ndarray], forecast: pd.DataFrame, out_path: Path) -> None:
    """
    Four held-out sectors (one MEarth star per period bin plus one TARS star, median-error GAN case in each):
    observed light curve, then the sinusoid, L1 and GAN forecasts of half 2.
    """
    gan = forecast[forecast["method"] == "gan_1"].copy()
    gan["group"] = np.where(gan["sample"] == "tars", "TARS <13.7", "MEarth " + gan["period_bin"])
    picks = []
    for group_name in ["MEarth <13.7", "MEarth 13.7-27", "MEarth >27", "TARS <13.7"]:
        group = gan[gan["group"] == group_name].sort_values("mae")
        if len(group) > 0:
            picks.append((group_name, group.iloc[len(group) // 2]))
    colors = {"sinusoid": "#2a78d6", "l1": "#eb6834", "gan_1": "#1baf7a"}  # categorical slots 1-3 of the dataviz reference palette
    labels = {"sinusoid": "fitted sinusoid", "l1": "L1-only generator", "gan_1": "GAN (1 sample)"}
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.6 * len(picks)), sharex=True)
    for ax, (group_name, pick) in zip(np.atleast_1d(axes), picks):
        row = int(np.flatnonzero((meta["tic_id"] == pick["tic_id"]).values & (meta["sector"] == pick["sector"]).values)[0])
        days = np.arange(flux.shape[1]) * bin_days
        ax.plot(days[mask[row]], flux[row][mask[row]], ".", color="#52514e", markersize=2, label="observed")
        freq, coef = fit_sinusoid(meta.loc[row, "t0"] + days[:half][mask[row][:half]], flux[row][:half][mask[row][:half]])
        curves = {"sinusoid": predict_sinusoid(freq, coef, meta.loc[row, "t0"] + days[half:]), "l1": preds["l1"][row], "gan_1": preds["gan_1"][row]}
        for method, curve in curves.items():
            ax.plot(days[half:], curve, color=colors[method], linewidth=1.2, label=labels[method])
        ax.axvline(days[half], color="#52514e", linewidth=0.8, linestyle="--")
        low, high = np.percentile(flux[row][mask[row]], [0.5, 99.5])
        ax.set_ylim(low - 0.2 * (high - low), high + 0.2 * (high - low))  # flares would otherwise flatten the curve
        ax.set_title(f"{group_name}: TIC {pick['tic_id']} sector {pick['sector']}, true P = {pick['period']} d, GAN MAE {pick['mae']}", fontsize=9)
        ax.set_ylabel("relative flux")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("days since sector start (dashed: half 1 | half 2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--data-tag", default="", help="suffix of the step5_data.py output, e.g. _smoke")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="only the first N star-sectors (smoke test)")
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke")
    args = parser.parse_args()

    log_path = repo / "data" / f"step5_eval{args.tag}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    flux, mask, meta = load_data(args.flux, args.data_tag)
    if args.limit is not None:
        flux, mask, meta = flux[:args.limit], mask[:args.limit], meta.head(args.limit)
    outputs = np.load(data_dir / f"{args.flux}{args.data_tag}{args.tag}_outputs.npz")
    assert (outputs["tic_id"] == meta["tic_id"].values).all() and (outputs["sector"] == meta["sector"].values).all(), "outputs npz rows do not match the meta rows; rerun step5_generate.py with the same --limit / --tag"
    preds = {}
    for key in ["l1", "gan_1", "gan_mean8", "ext_l1", "ext_gan"]:
        preds[key] = outputs[key]
    logger.info(f"{len(meta)} star-sectors from {meta['tic_id'].nunique()} stars; generator outputs from step5_generate.py")

    forecast_ckpt = repo / "data" / f"step5_forecast{args.tag}.partial.csv"
    ls_ckpt = repo / "data" / f"step5_ls{args.tag}.partial.csv"
    finished = set()
    if ls_ckpt.exists():
        done = pd.read_csv(ls_ckpt, usecols=["tic_id", "sector"])
        finished = set(zip(done["tic_id"], done["sector"]))
        logger.info(f"resuming: {len(finished)} star-sectors already in {ls_ckpt.name}")
    todo = []
    for i, info in enumerate(meta.to_dict("records")):
        if (info["tic_id"], info["sector"]) not in finished:
            todo.append(i)
    logger.info(f"{len(todo)} star-sectors to score; each runs 4 periodograms (~seconds each)")

    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for i in todo:
            row_preds = {}
            for key, value in preds.items():
                row_preds[key] = value[i]
            futures.append(pool.submit(score_sector, meta.iloc[i].to_dict(), flux[i], mask[i], row_preds))
        for future in tqdm(as_completed(futures), desc="score", total=len(futures)):
            forecast_rows, ls_rows = future.result()
            pd.DataFrame(forecast_rows).to_csv(forecast_ckpt, mode="a", header=not forecast_ckpt.exists(), index=False)
            pd.DataFrame(ls_rows).reindex(columns=ls_columns).to_csv(ls_ckpt, mode="a", header=not ls_ckpt.exists(), index=False)
    logger.info(f"scoring wall {time.perf_counter() - wall_start} s on {args.workers} workers")

    forecast = pd.read_csv(forecast_ckpt)
    ls = pd.read_csv(ls_ckpt)
    forecast.to_csv(results_dir / f"step5_forecast{args.tag}.csv", index=False)
    ls.to_csv(results_dir / f"step5_ls_extension{args.tag}.csv", index=False)
    forecast_summary, ls_summary = summarize(forecast, ls)
    forecast_summary.to_csv(results_dir / f"step5_forecast_summary{args.tag}.csv", index=False)
    ls_summary.to_csv(results_dir / f"step5_ls_summary{args.tag}.csv", index=False)
    logger.info("forecast summary (MAE in units of half-1 robust std):\n" + forecast_summary.to_string(index=False))
    logger.info("LS after extension:\n" + ls_summary.to_string(index=False))
    plot_examples(flux, mask, meta, preds, forecast, results_dir / "plots" / f"step5_examples{args.tag}.png")
    logger.info(f"wrote results/step5_*{args.tag}.csv and results/plots/step5_examples{args.tag}.png")


if __name__ == "__main__":
    main()
