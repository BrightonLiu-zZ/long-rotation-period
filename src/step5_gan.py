"""
Step 5 of docs/ROADMAP.md, training: a small conditional GAN that writes the second half of a sector from the first half.
Generator: 1D conv encoder-decoder (U-Net skips) on [flux, mask, noise] of half 1 --> flux of half 2.
Discriminator: 1D conv patch critic on (half 1, half 2) pairs, real or generated.
Loss (pix2pix style): masked L1 x lambda_l1 + adversarial x lambda_adv; --lambda-adv 0 trains the same generator with L1 only.
For each fold k, trains on the other four folds' stars; fold k's stars are never seen (held-out stars, not held-out windows).
10% of the training stars are set aside for a validation curve in W&B; the checkpoint is always the last epoch.
Needs numpy + torch only (env swm); reads data/gan/{flux}.npz written by step5_data.py.
Writes checkpoints/step5/{variant}_fold{k}.pt.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from step5_common import half, load_data, model_input, normalise, repo

val_fraction = 0.1  # of training stars, for the W&B validation curve only

logger = logging.getLogger("step5_gan")


def tensors(flux: np.ndarray, mask: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn gridded sectors (N, 1344) into training tensors: conditioning half 1, target half 2 and its mask.
    Both halves are normalised with half 1's centre and scale, so the target is in the units the model sees.
    """
    f1, m1 = flux[:, :half], mask[:, :half]
    f2, m2 = flux[:, half:], mask[:, half:]
    centre, scale = normalise(f1, m1)
    cond = torch.tensor(model_input(f1, m1, centre, scale), device=device)  # (N, 2, 672)
    target = torch.tensor(model_input(f2, m2, centre, scale)[:, :1], device=device)  # (N, 1, 672)
    target_mask = torch.tensor(m2[:, None, :], dtype=torch.float32, device=device)  # (N, 1, 672)
    return cond, target, target_mask


def masked_l1(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - target) * target_mask).sum() / target_mask.sum().clamp(min=1.0)


def down(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, 5, stride=2, padding=2), nn.LeakyReLU(0.2))  # halves length


def up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(c_in, c_out, 5, padding=2), nn.LeakyReLU(0.2))  # doubles length


class Generator(nn.Module):
    """
    Writes half 2 from half 1.
    The encoder shrinks 672 bins to 42 in four stride-2 steps; a pooled vector adds sector-wide context at the bottom;
    the decoder grows back to 672 and reuses each encoder level through skip connections (same position 14 d earlier).
    Noise enters as an extra input channel, so repeated calls give different continuations.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, width, 7, padding=3), nn.LeakyReLU(0.2))  # (B, 32, 672)
        self.d1 = down(width, 2 * width)  # (B, 64, 336)
        self.d2 = down(2 * width, 4 * width)  # (B, 128, 168)
        self.d3 = down(4 * width, 4 * width)  # (B, 128, 84)
        self.d4 = down(4 * width, 4 * width)  # (B, 128, 42)
        self.context = nn.Linear(4 * width, 4 * width)
        self.u4 = up(4 * width, 4 * width)  # (B, 128, 84)
        self.u3 = up(8 * width, 4 * width)  # (B, 128, 168), input = u4 + d3 skip
        self.u2 = up(8 * width, 2 * width)  # (B, 64, 336), input = u3 + d2 skip
        self.u1 = up(4 * width, width)  # (B, 32, 672), input = u2 + d1 skip
        self.head = nn.Conv1d(2 * width, 1, 7, padding=3)  # (B, 1, 672), input = u1 + stem skip

    def forward(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s = self.stem(torch.cat([cond, noise], dim=1))  # (B, 32, 672)
        e1 = self.d1(s)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)  # (B, 128, 42)
        e4 = e4 + self.context(e4.mean(dim=2))[:, :, None]  # broadcast sector-wide summary to every position
        h = self.u4(e4)
        h = self.u3(torch.cat([h, e3], dim=1))
        h = self.u2(torch.cat([h, e2], dim=1))
        h = self.u1(torch.cat([h, e1], dim=1))
        return self.head(torch.cat([h, s], dim=1))  # (B, 1, 672)


class Discriminator(nn.Module):
    """
    Scores (half 1, half 2) pairs: one real/fake logit per ~8-bin patch of half 2.
    Spectral norm keeps each layer's gain bounded, which steadies GAN training.
    """

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(4, width, 5, stride=2, padding=2)), nn.LeakyReLU(0.2),  # (B, 64, 336)
            sn(nn.Conv1d(width, 2 * width, 5, stride=2, padding=2)), nn.LeakyReLU(0.2),  # (B, 128, 168)
            sn(nn.Conv1d(2 * width, 2 * width, 5, stride=2, padding=2)), nn.LeakyReLU(0.2),  # (B, 128, 84)
            sn(nn.Conv1d(2 * width, 1, 5, padding=2)),  # (B, 1, 84) patch logits
        )

    def forward(self, cond: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, target * target_mask, target_mask], dim=1))


def generate(generator: Generator, cond: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """
    One continuation per row of cond (B, 2, 672), in normalised units. A seed makes the noise reproducible.
    """
    if seed is None:
        noise = torch.randn(len(cond), 1, half, device=cond.device)
    else:
        gen = torch.Generator(device=cond.device).manual_seed(seed)
        noise = torch.randn(len(cond), 1, half, device=cond.device, generator=gen)
    return generator(cond, noise)


def train_fold(flux: np.ndarray, mask: np.ndarray, meta: pd.DataFrame, fold: int, args: argparse.Namespace, device: torch.device) -> Path:
    """
    Train one generator (and discriminator, if lambda_adv > 0) on all stars outside the fold, for a fixed number of epochs.
    No early stopping and no test-set monitoring: the last epoch is the checkpoint, so the held-out fold never steers training.
    """
    if args.lambda_adv > 0:
        variant = "gan"
    else:
        variant = "l1"
    torch.manual_seed(fold)
    train_rows = np.flatnonzero(meta["fold"].values != fold)
    stars = meta.loc[train_rows, "tic_id"].unique()
    if args.limit_stars is not None:
        stars = stars[:args.limit_stars]
    val_stars = np.random.default_rng(fold).choice(stars, size=max(1, int(val_fraction * len(stars))), replace=False)
    val_rows = train_rows[meta.loc[train_rows, "tic_id"].isin(val_stars).values]
    train_rows = train_rows[meta.loc[train_rows, "tic_id"].isin(stars).values & ~meta.loc[train_rows, "tic_id"].isin(val_stars).values]
    cond, target, target_mask = tensors(flux[train_rows], mask[train_rows], device)
    val_cond, val_target, val_mask = tensors(flux[val_rows], mask[val_rows], device)
    logger.info(f"fold {fold} {variant}: {len(train_rows)} training star-sectors, {len(val_rows)} validation star-sectors (curve only, never used to pick a checkpoint)")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project="long-rotation-period", name=f"step5_{variant}_fold{fold}", mode=args.wandb_mode, config={**vars(args), "fold": fold, "variant": variant, "n_train": len(train_rows)}, reinit="finish_previous")

    for epoch in tqdm(range(args.epochs), desc=f"{variant} fold {fold}", total=args.epochs):
        order = torch.randperm(len(cond), device=device)
        sums = {"l1": 0.0, "adv": 0.0, "d": 0.0}
        for start in range(0, len(cond), args.batch_size):
            batch = order[start:start + args.batch_size]
            c, y, ym = cond[batch], target[batch], target_mask[batch]
            fake = generate(generator, c)

            if args.lambda_adv > 0:
                d_real = discriminator(c, y, ym)
                d_fake = discriminator(c, fake.detach(), ym) # detach: discriminator step must not update the generator
                d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                opt_d.zero_grad() # clear gradients from the previous batch
                d_loss.backward() # compute gradients via backpropagation
                opt_d.step() # update discriminator weights
                sums["d"] += d_loss.item() * len(batch)

            l1 = masked_l1(fake, y, ym)
            g_loss = args.lambda_l1 * l1
            if args.lambda_adv > 0:
                d_on_fake = discriminator(c, fake, ym)
                adv = F.binary_cross_entropy_with_logits(d_on_fake, torch.ones_like(d_on_fake)) # non-saturating: generator wants "real"
                g_loss = g_loss + args.lambda_adv * adv
                sums["adv"] += adv.item() * len(batch)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            sums["l1"] += l1.item() * len(batch)

        means = {}
        for key, value in sums.items():
            means[key] = value / len(cond)
        generator.eval()
        with torch.no_grad(): # disable gradient tracking for the validation pass
            val_l1 = masked_l1(generate(generator, val_cond, seed=0), val_target, val_mask).item()
        generator.train()
        if run is not None:
            run.log({"epoch": epoch, "train/l1": means["l1"], "train/adv": means["adv"], "train/d_loss": means["d"], "val/l1": val_l1})
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            logger.info(f"fold {fold} {variant} epoch {epoch}: train l1 {means['l1']} val l1 {val_l1} adv {means['adv']} d {means['d']}")

    out = Path(args.out) / f"{variant}_fold{fold}.pt"
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "args": vars(args), "fold": fold, "variant": variant}, out)
    if run is not None:
        run.finish()
    logger.info(f"saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", choices=["pdcsap", "sap"], default="pdcsap")
    parser.add_argument("--data-tag", default="", help="suffix of the step5_data.py output, e.g. _smoke")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--lambda-adv", type=float, default=1.0, help="0 = L1-only generator, no discriminator")
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit-stars", type=int, default=None, help="only the first N training stars (smoke test)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--out", default=str(repo / "checkpoints" / "step5"))
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(Path(args.out) / "train.log", mode="a", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device {device}; loading data/gan/{args.flux}{args.data_tag}.npz")
    flux, mask, meta = load_data(args.flux, args.data_tag)

    wall_start = time.perf_counter()
    for fold in args.folds:
        train_fold(flux, mask, meta, fold, args, device)
    logger.info(f"all folds done in {time.perf_counter() - wall_start} s")


if __name__ == "__main__":
    main()
