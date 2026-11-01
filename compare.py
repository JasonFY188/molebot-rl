"""Plot SAC vs TD3 vs PPO learning curves for a task (paper-style).

Reads every run under runs/<task>/<algo>/: the folder itself and any sub-folders
(e.g. from `--tag s0 --tag s1 ...`). Each folder with monitor CSVs = one run/seed;
its per-worker monitor_*.csv files are merged. With several seeds per algo it draws
a mean +/- std band; with one, a single smoothed line.

    # one run per algo -> three clean lines
    python compare.py --task cross

    # paper-style bands: train each algo with a few seeds first, e.g.
    #   python train.py --algo sac --task cross --n-envs 6 --grains 8,8,4 \
    #                   --timesteps 100000 --seed 0 --tag s0
    #   ... --seed 1 --tag s1  ,  ... --seed 2 --tag s2
    # then:
    python compare.py --task cross --window 50
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < 2 or w <= 1:
        return y
    w = min(w, len(y))
    return np.convolve(y, np.ones(w) / w, mode="valid")


def run_series(run_dir: str):
    """Merge monitor*.csv in one dir -> (cum_steps, reward) episode arrays."""
    frames = []
    for p in sorted(glob.glob(os.path.join(run_dir, "monitor*.csv"))):
        try:
            frames.append(pd.read_csv(p, skiprows=1))   # skip the '#{...}' header line
        except Exception:  # noqa: BLE001
            pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    if len(df) == 0:
        return None
    return df["l"].cumsum().to_numpy(), df["r"].to_numpy()


def find_runs(algo_dir: str):
    """Every directory under algo_dir (incl. itself) that holds monitor CSVs = one run."""
    runs = []
    for root, _, files in os.walk(algo_dir):
        if any(f.startswith("monitor") and f.endswith(".csv") for f in files):
            s = run_series(root)
            if s is not None:
                runs.append(s)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cross", choices=["cross", "straight"])
    ap.add_argument("--runs", default=None, help="runs dir (default: runs/<task>)")
    ap.add_argument("--algos", default="sac,td3,ppo")
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                    help="plot exactly one run directory; repeat for each line, e.g. "
                         "--run SAC=runs/cross/sac/s0 --run PPO=runs/cross/ppo/s0")
    ap.add_argument("--window", type=int, default=30, help="episode smoothing window")
    ap.add_argument("--bins", type=int, default=120, help="x-grid resolution for bands")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = args.runs or os.path.join(os.path.dirname(__file__), "runs", args.task)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    if args.run:
        for spec in args.run:
            if "=" not in spec:
                ap.error(f"--run must be LABEL=DIR, got: {spec}")
            label, run_dir = spec.split("=", 1)
            series = run_series(run_dir)
            if series is None:
                print(f"  skip {label}: no monitor data under {run_dir}")
                continue
            xs, ys = series
            ys_s = smooth(ys, args.window)
            xs_s = xs[len(xs) - len(ys_s):]
            ax.plot(xs_s, ys_s, label=label, linewidth=2)
            print(f"  {label}: {len(ys)} episodes from {run_dir}")
            plotted += 1
    else:
        for algo in [a.strip() for a in args.algos.split(",")]:
            algo_dir = os.path.join(base, algo)
            if not os.path.isdir(algo_dir):
                print(f"  skip {algo}: no folder {algo_dir}")
                continue
            runs = find_runs(algo_dir)
            if not runs:
                print(f"  skip {algo}: no monitor data under {algo_dir}")
                continue

            n = len(runs)
            label = f"{algo.upper()} ({n} seed{'s' if n > 1 else ''})"
            if n == 1:
                xs, ys = runs[0]
                ys_s = smooth(ys, args.window)
                xs_s = xs[len(xs) - len(ys_s):]
                ax.plot(xs_s, ys_s, label=label, linewidth=2)
            else:
                xmax = min(xs.max() for xs, _ in runs)          # common range all seeds cover
                grid = np.linspace(0, xmax, args.bins)
                curves = []
                for xs, ys in runs:
                    ys_s = smooth(ys, args.window)
                    xs_s = xs[len(xs) - len(ys_s):]
                    curves.append(np.interp(grid, xs_s, ys_s))
                arr = np.asarray(curves)
                mean, std = arr.mean(0), arr.std(0)
                line, = ax.plot(grid, mean, label=label, linewidth=2)
                ax.fill_between(grid, mean - std, mean + std, alpha=0.2, color=line.get_color())
            print(f"  {algo}: {n} run(s)")
            plotted += 1

    if plotted == 0:
        print("Nothing to plot. Train some models first (train.py).")
        return

    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return (moving avg, w={args.window})")
    title = (f"Molebot '{args.task}': selected training runs"
             if args.run else f"Molebot '{args.task}': SAC vs TD3 vs PPO")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    out = args.out or os.path.join(base, f"compare_{args.task}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
