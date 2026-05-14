"""WGAN-GP shared critic trainer for pairwise medical-image evaluation.

Usage
-----
python train.py \\
    --real_dir  /data/real \\
    --fake_dirs /data/gen_A /data/gen_B \\
    --arch cnn --isize 128 --ndim 3 \\
    --epochs 200 --out_dir ./run_cnn

The script trains ONE critic on real vs. the pooled fake distribution, saves
the checkpoint with the highest training W-distance, then evaluates it on
held-out sets to produce per-model scores.  The model with the smaller
W-distance is closer to the real distribution.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MixedSubset, NiftiDataset, check_data_range
from losses import calculate_gradient_penalty
from models import build_discriminator

# ---------------------------------------------------------------------------
# Per-architecture training hyperparameters
# ---------------------------------------------------------------------------

_ARCH_HPARAMS = {
    # CNN: inductive bias (translation invariance) keeps gradients stable.
    # β1=0.0 is WGAN-standard; β2=0.9 is sufficient for local conv features.
    "cnn":         {"lr": 5e-5, "gp_lambda": 30.0, "betas": (0.0, 0.9)},
    # Transformer: no local bias, attention weights are scale-sensitive.
    # β1 must stay 0 for unbiased instant gradients; β2=0.99 smooths the
    # second-moment estimate across attention heads to prevent gradient spikes.
    "transformer": {"lr": 5e-5, "gp_lambda": 10.0, "betas": (0.0, 0.99)},
    # Hybrid shares the Transformer encoder so inherits its β2 requirement.
    "hybrid":      {"lr": 5e-5, "gp_lambda": 10.0, "betas": (0.0, 0.99)},
}

# Convergence thresholds — shared across all architectures.
# All three must hold for every epoch in the patience window to declare convergence.
_CONV_GN_TOL  = 0.20   # GradNorm must stay within [0.90, 1.10]
_CONV_W_CV    = 0.05   # W-dist coefficient of variation < 5 %
_CONV_GP_FRAC = 0.05   # mean GP < 5 % of gp_lambda (near zero)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(dataset, train_ratio: float):
    n = len(dataset)
    n_train = int(train_ratio * n)
    return Subset(dataset, range(n_train)), Subset(dataset, range(n_train, n))


def _eval_scores(discriminator, dataset, batch_size: int, device) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.append(discriminator(batch.to(device)).cpu())
    return torch.cat(scores)


def _save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def _cv(values) -> float:
    """Coefficient of variation (std / |mean|) of a sequence."""
    a = np.asarray(values, dtype=np.float64)
    mu = np.abs(a.mean())
    return float(a.std() / mu) if mu > 1e-8 else float("inf")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    hp = _ARCH_HPARAMS[args.arch]

    # --- Datasets & splits ---------------------------------------------------
    real_ds = NiftiDataset(args.real_dir, args.isize, args.ndim)
    n_real = len(real_ds)

    fake_dss = []
    for d in args.fake_dirs:
        ds = NiftiDataset(d, args.isize, args.ndim)
        fake_dss.append(Subset(ds, range(min(len(ds), n_real))))

    real_train_ds, real_eval_ds = _split(real_ds, args.real_train_ratio)
    fake_train_dss, fake_eval_dss = [], []
    for ds in fake_dss:
        tr, ev = _split(ds, args.fake_train_ratio)
        fake_train_dss.append(tr)
        fake_eval_dss.append(ev)

    n_models = len(fake_train_dss)
    n_real_train = len(real_train_ds)
    fake_per_epoch = n_real_train // n_models

    print(f"Real  — train: {n_real_train}  eval: {len(real_eval_ds)}")
    for i, (tr, ev) in enumerate(zip(fake_train_dss, fake_eval_dss)):
        print(f"Fake {i} — train: {len(tr)}  eval: {len(ev)}  dir: {args.fake_dirs[i]}")
    print(f"fake_per_epoch per model: {fake_per_epoch}")

    # --- Data range check ----------------------------------------------------
    check_data_range(real_ds, "real", n_samples=1)
    for i, ds in enumerate(fake_dss):
        name = os.path.basename(args.fake_dirs[i].rstrip("/\\"))
        check_data_range(ds.dataset, f"fake/{name}", n_samples=1)
    print()

    # --- Model & optimiser ---------------------------------------------------
    override = {}
    if args.ndf        is not None: override["ndf"]        = args.ndf
    if args.patch_size is not None: override["patch_size"] = args.patch_size
    if args.d_model    is not None: override["d_model"]    = args.d_model
    if args.n_heads    is not None: override["n_heads"]    = args.n_heads
    if args.n_layers   is not None: override["n_layers"]   = args.n_layers

    disc = build_discriminator(
        args.arch, args.ndim, args.in_channels, args.isize, **override
    ).to(device)

    n_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator ({args.arch}): {n_params:,} parameters\n")

    lr        = args.lr        if args.lr        is not None else hp["lr"]
    gp_lambda = args.gp_lambda if args.gp_lambda is not None else gp_lambda

    opt = torch.optim.Adam(
        disc.parameters(),
        lr=lr,
        betas=hp["betas"],
        weight_decay=1e-4,
    )

    # --- Tracking state ------------------------------------------------------
    patience = args.patience
    best_train_w = -float("inf")
    best_ckpt_path = os.path.join(args.out_dir, "best.pt")
    ckpt_saved = False  # tracks whether any checkpoint was saved this run

    w_window  = deque(maxlen=patience)
    gn_window = deque(maxlen=patience)
    gp_window = deque(maxlen=patience)
    conv_streak  = 0
    gn_div_streak = 0
    stop_reason = None

    # --- Epoch loop ----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        disc.train()

        mixed_fake_ds = MixedSubset(fake_train_dss, fake_per_epoch)
        real_loader = DataLoader(
            real_train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers, pin_memory=True,
        )
        fake_loader = DataLoader(
            mixed_fake_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers, pin_memory=True,
        )
        real_iter = iter(real_loader)
        fake_iter = iter(fake_loader)
        steps = min(len(real_loader), len(fake_loader))

        epoch_w, epoch_gn, epoch_gp = [], [], []

        for _ in range(steps):
            real_b = next(real_iter).to(device)
            fake_b = next(fake_iter).to(device)

            opt.zero_grad()
            w = disc(real_b).mean() - disc(fake_b).mean()
            gp, gn = calculate_gradient_penalty(disc, real_b, fake_b, gp_lambda)
            (-w + gp).backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            opt.step()

            epoch_w.append(w.item())
            epoch_gn.append(gn)
            epoch_gp.append(gp.item())

        mean_w  = float(np.mean(epoch_w))
        mean_gn = float(np.mean(epoch_gn))
        mean_gp = float(np.mean(epoch_gp))
        elapsed = time.time() - t0

        # --- Checkpoint: save only when GN is valid and trainW improves --------
        # GN far from 1.0 means the Lipschitz constraint is broken — the
        # W-distance estimate is unreliable regardless of its magnitude.
        gn_valid = abs(mean_gn - 1.0) <= _CONV_GN_TOL
        if gn_valid and mean_w > best_train_w:
            best_train_w = mean_w
            _save_checkpoint(
                {"epoch": epoch, "state_dict": disc.state_dict(),
                 "optimizer": opt.state_dict(), "train_w": mean_w},
                best_ckpt_path,
            )
            ckpt_saved = True
            marker = " *"
        else:
            marker = ""

        print(
            f"[{epoch:4d}/{args.epochs}] "
            f"trainW={mean_w:.4f}  GN={mean_gn:.3f}  GP={mean_gp:.3f}"
            f"  ({elapsed:.1f}s){marker}"
        )

        # --- Early-stop bookkeeping ------------------------------------------
        w_window.append(mean_w)
        gn_window.append(mean_gn)
        gp_window.append(mean_gp)

        # 1. Convergence: stable W, GN≈1, GP≈0 held for `patience` epochs
        if len(w_window) == patience:
            cv_ok     = _cv(w_window) < _CONV_W_CV
            gn_ok_all = all(abs(g - 1.0) <= _CONV_GN_TOL for g in gn_window)
            gp_ok     = float(np.mean(gp_window)) < _CONV_GP_FRAC * gp_lambda
            if cv_ok and gn_ok_all and gp_ok:
                conv_streak += 1
            else:
                conv_streak = 0
            if conv_streak >= patience:
                stop_reason = "convergence"
                break

        # 2. GN divergence: GN > 2.0 for `patience` consecutive epochs
        if mean_gn > 2.0:
            gn_div_streak += 1
        else:
            gn_div_streak = 0
        if gn_div_streak >= patience:
            stop_reason = "GradNorm divergence"
            break

    # --- Final evaluation on held-out sets -----------------------------------
    print()
    if stop_reason:
        print(f"Early stop: {stop_reason}")

    if not ckpt_saved:
        print("No checkpoint saved this run — GN never entered the valid range [0.90, 1.10].")
        print("Suggestions: increase GP_LAMBDA, lower LR, or reduce NDF.")
        return
    if not os.path.exists(best_ckpt_path):
        print("No checkpoint file found.")
        return

    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
    print(f"Loading best checkpoint (epoch {ckpt['epoch']}, trainW={ckpt['train_w']:.4f})")

    disc.load_state_dict(ckpt["state_dict"])
    disc.eval().to(device)

    real_scores = _eval_scores(disc, real_eval_ds, args.batch_size, device)
    eval_ws = []
    for fake_eval_ds in fake_eval_dss:
        fake_scores = _eval_scores(disc, fake_eval_ds, args.batch_size, device)
        eval_ws.append((real_scores.mean() - fake_scores.mean()).item())

    best_w = min(eval_ws)
    # os.path.basename returns '' for paths with trailing slash — strip it first
    names = [os.path.basename(p.rstrip("/\\")) for p in args.fake_dirs]

    print(f"\nFinal scores  (epoch {ckpt['epoch']}):")
    print(f"  {'Model':<40}  {'W-distance':>12}  {'Ratio':>8}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*8}")
    for name, w in zip(names, eval_ws):
        ratio = w / best_w if best_w != 0 else float("nan")
        tag  = "  ← closer to real" if w == best_w else ""
        warn = "  WARNING: negative — critic may not have converged" if w < 0 else ""
        print(f"  {name:<40}  {w:>12.4f}  {ratio:>8.3f}{tag}{warn}")

    print()
    print("Interpretation: lower W-distance = closer to real.  Ratio = W / best_W.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train a WGAN-GP shared critic and report per-model W-distances."
    )

    # Required
    p.add_argument("--real_dir",  required=True,
                   help="Directory containing real .nii/.nii.gz images.")
    p.add_argument("--fake_dirs", required=True, nargs="+",
                   help="One directory per generator model (at least one).")

    # Architecture
    p.add_argument("--arch",        default="cnn", choices=["cnn", "transformer", "hybrid"])
    p.add_argument("--isize",       type=int, default=64,
                   help="Isotropic spatial size images are resampled to.")
    p.add_argument("--ndim",        type=int, default=3, choices=[2, 3])
    p.add_argument("--in_channels", type=int, default=1)

    # Architecture overrides (auto-scaled when omitted)
    p.add_argument("--ndf",        type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--d_model",    type=int, default=None)
    p.add_argument("--n_heads",    type=int, default=None)
    p.add_argument("--n_layers",   type=int, default=None)

    # Training
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--patience",         type=int,   default=10)
    p.add_argument("--real_train_ratio", type=float, default=0.6)
    p.add_argument("--fake_train_ratio", type=float, default=0.6)
    # Override arch-default lr / gp_lambda from the command line
    p.add_argument("--lr",        type=float, default=None,
                   help="Learning rate (overrides arch default).")
    p.add_argument("--gp_lambda", type=float, default=None,
                   help="Gradient-penalty coefficient (overrides arch default).")

    # Infrastructure
    p.add_argument("--out_dir",     default="./checkpoints")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    train(args)
