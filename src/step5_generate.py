"""
Step 5 of docs/ROADMAP.md, model outputs: run each fold's L1-only and GAN generators on that fold's held-out star-sectors.
Separate from step5_eval.py because torch and the periodogram code cannot share a process in env astro (two OpenMP runtimes).
Run in env swm after step5_gan.py has written all checkpoints. Inference only, about a minute.
Writes data/gan/{flux}{tag}_outputs.npz, row order = data/gan/{flux}{tag}_meta.csv, all in flux units:
- l1, gan_1, gan_mean8: forecasts of half 2 from half 1, shape (N, 672).
- ext_l1, ext_gan: extension past the 28-day grid, n_ext_steps x 14 d, shape (N, n_ext_steps * 672).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from step5_common import data_dir, half, load_data, model_input, normalise, repo
from step5_gan import Generator, generate

n_ext_steps = 2  # autoregressive 14-d steps appended after the 28-d grid
n_gan_samples = 8  # samples averaged for the "GAN mean of 8" forecast row


def run_generator(model: Generator, flux: np.ndarray, mask: np.ndarray, seed: int, device: torch.device) -> np.ndarray:
    """
    Continue each conditioning segment (N, 672) by 672 bins in flux units.
    The segment is normalised by its own robust centre / scale (as in training), passed through the generator, then mapped back.
    """
    centre, scale = normalise(flux, mask)
    cond = torch.tensor(model_input(flux, mask, centre, scale), device=device)
    with torch.no_grad(): # disable gradient tracking, not needed at inference
        out = generate(model, cond, seed=seed)[:, 0].cpu().numpy()  # (N, 672)
    return out * scale[:, None] + centre[:, None]


def extend(model: Generator, flux: np.ndarray, mask: np.ndarray, seed: int, device: torch.device) -> np.ndarray:
    """
    Autoregressive extension: condition on the real half 2, generate 14 d, then condition on that output, n_ext_steps times.
    Returns (N, n_ext_steps * 672) flux for grid bins 1344 onwards.
    """
    cond_flux, cond_mask = flux[:, half:], mask[:, half:]
    pieces = []
    for step in range(n_ext_steps):
        piece = run_generator(model, cond_flux, cond_mask, seed + step, device)
        pieces.append(piece)
        cond_flux = piece
        cond_mask = np.ones_like(piece, dtype=bool)
    return np.concatenate(pieces, axis=1)


def model_outputs(flux: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, ckpt_dir: Path, device: torch.device) -> dict[str, np.ndarray]:
    """
    Fill full-dataset arrays fold by fold, each fold with the generators that never saw its stars.
    """
    n = len(meta)
    out = {
        "l1": np.zeros((n, half)),
        "gan_1": np.zeros((n, half)),
        "gan_mean8": np.zeros((n, half)),
        "ext_l1": np.zeros((n, n_ext_steps * half)),
        "ext_gan": np.zeros((n, n_ext_steps * half)),
    }
    for fold in sorted(meta["fold"].unique()):
        rows = np.flatnonzero(meta["fold"].values == fold)
        f, m = flux[rows], mask[rows]
        for variant in ["l1", "gan"]:
            ckpt = torch.load(ckpt_dir / f"{variant}_fold{fold}.pt", map_location=device)
            model = Generator().to(device)
            model.load_state_dict(ckpt["generator"])
            model.eval() # inference mode; this network has no dropout or batchnorm, kept for clarity
            if variant == "l1":
                out["l1"][rows] = run_generator(model, f[:, :half], m[:, :half], 0, device)
            else:
                samples = []
                for seed in range(n_gan_samples):
                    samples.append(run_generator(model, f[:, :half], m[:, :half], seed, device))
                out["gan_1"][rows] = samples[0]
                out["gan_mean8"][rows] = np.mean(samples, axis=0)
            out[f"ext_{variant}"][rows] = extend(model, f, m, 100, device)
        print(f"fold {fold}: outputs for {len(rows)} held-out star-sectors", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--data-tag", default="", help="suffix of the step5_data.py output, e.g. _smoke")
    parser.add_argument("--ckpt-dir", default=str(repo / "checkpoints" / "step5"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N star-sectors (smoke test)")
    parser.add_argument("--tag", default="", help="suffix for the output name, e.g. _smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flux, mask, meta = load_data(args.flux, args.data_tag)
    if args.limit is not None:
        flux, mask, meta = flux[:args.limit], mask[:args.limit], meta.head(args.limit)
    out = model_outputs(flux, mask, meta, Path(args.ckpt_dir), device)
    out_path = data_dir / f"{args.flux}{args.data_tag}{args.tag}_outputs.npz"
    np.savez_compressed(out_path, tic_id=meta["tic_id"].values, sector=meta["sector"].values, **out)
    print(f"wrote {out_path} on {device}")


if __name__ == "__main__":
    main()
