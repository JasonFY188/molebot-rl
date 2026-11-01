"""
Draws docs/figures/result_summary.png from the committed logs only:
  left : PPO training on the 'cross' task — success rate (moving window of 100 episodes) vs environment steps,
         from results/ppo_cross_easy/monitor_*.csv (Stable-Baselines3 Monitor logs, 6 parallel simulators)
  right: deterministic evaluation of the final policy, from results/eval/*.txt (output of evaluate.py), if present

Usage:  python docs/make_figure.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "results", "ppo_cross_easy")
EVAL = os.path.join(ROOT, "results", "eval")
OUT = os.path.join(ROOT, "docs", "figures", "result_summary.png")


def load_training():
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in glob.glob(os.path.join(RUN, "monitor_*.csv"))])
    df = df.sort_values("t").reset_index(drop=True)          # t = wall-clock seconds since start
    df["success"] = df.is_success.astype(str).str.lower().eq("true")
    df["steps"] = df.l.cumsum()
    return df


def load_eval():
    rows = []
    for f in sorted(glob.glob(os.path.join(EVAL, "*.txt"))):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        eps = re.findall(r"success=(True|False)", txt)
        grains = re.search(r"grains_(\d+)x(\d+)x(\d+)", os.path.basename(f))
        if eps and grains:
            n = int(grains.group(1)) * int(grains.group(2)) * int(grains.group(3))
            rows.append(dict(label=f"{grains.group(1)}×{grains.group(2)}×{grains.group(3)}\n({n} balls)",
                             n=len(eps), k=sum(e == "True" for e in eps)))
    return rows


def main():
    df = load_training()
    k = len(df) // 5
    print(f"episodes {len(df)} | steps {int(df.l.sum())} | successes {int(df.success.sum())} | "
          f"success first 20% {df.success.iloc[:k].mean():.3f} last 20% {df.success.iloc[-k:].mean():.3f}")
    ev = load_eval()
    for r in ev:
        print(f"eval {r['label'].splitlines()[0]}: {r['k']}/{r['n']} = {r['k'] / r['n']:.0%}")
    fig, ax = plt.subplots(1, 2 if ev else 1, figsize=(12 if ev else 7, 4.2), squeeze=False)
    a = ax[0, 0]
    roll = df.success.rolling(100, min_periods=20).mean() * 100
    a.plot(df.steps, roll, color="#8a5a2b", lw=2)
    a.set(xlabel="environment steps", ylabel="success rate, last 100 episodes (%)",
          title="PPO training: reaching the goal through the ball pile")
    a.grid(alpha=0.3)
    if ev:
        b = ax[0, 1]
        x = np.arange(len(ev))
        rate = [100 * r["k"] / r["n"] for r in ev]
        b.bar(x, rate, color="#8a5a2b", width=0.55)
        for i, r in enumerate(ev):
            b.text(i, rate[i] + 2, f"{r['k']}/{r['n']}", ha="center")
        b.set(xticks=x, xticklabels=[r["label"] for r in ev], ylim=(0, 100), ylabel="success rate (%)",
              title="Final policy, deterministic evaluation")
        b.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)


if __name__ == "__main__":
    main()
