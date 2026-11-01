"""Roll out a trained policy (watch it in the GUI and print episode stats).

    # open CoppeliaSim GUI + load scene, then:
    python evaluate.py --model runs/cross/sac/sac_final.zip --attach --episodes 5

    # or launch a GUI automatically:
    python evaluate.py --model runs/cross/sac/sac_final.zip
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import default_config
from molebot_env import MolebotEnv
from stable_baselines3 import SAC, TD3, PPO

ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a saved .zip")
    ap.add_argument("--algo", default=None, choices=list(ALGOS))
    ap.add_argument("--task", default=None, choices=["cross", "straight"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--grains", default=None, help="pile size 'nx,ny,nz'")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of deterministic")
    args = ap.parse_args()

    # infer algo/task from meta.json (written by train.py) or the filename
    algo, task = args.algo, args.task
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.model)), "meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        algo = algo or meta.get("algo")
        task = task or meta.get("task")
    if algo is None:
        for k in ALGOS:
            if k in os.path.basename(args.model).lower():
                algo = k
                break
    if algo is None:
        ap.error("could not infer --algo; pass it explicitly")
    task = task or "cross"

    cfg = default_config()
    cfg.task = task
    if args.scene:
        cfg.sim.scene = args.scene
    if args.port:
        cfg.sim.port = args.port
    cfg.sim.launch = not args.attach
    cfg.sim.headless = args.headless
    if args.grains:
        cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = (int(x) for x in args.grains.split(","))

    print(f"[eval] algo={algo} task={task} model={args.model}")
    env = MolebotEnv(cfg, task=task)
    model = ALGOS[algo].load(args.model, device="auto")

    returns, successes = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, ret, steps, succ = False, 0.0, 0, False
        info = {}
        while not done:
            act, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rew, term, trunc, info = env.step(act)
            ret += rew
            steps += 1
            succ = succ or bool(info.get("is_success", False))
            done = term or trunc
        returns.append(ret)
        successes.append(succ)
        print(f"  ep{ep}: return={ret:8.2f}  steps={steps:4d}  success={succ}  "
              f"info={ {k: round(v,3) if isinstance(v,(int,float)) else v for k,v in info.items()} }")

    print(f"[eval] mean return={np.mean(returns):.2f} +/- {np.std(returns):.2f}   "
          f"success_rate={np.mean(successes):.2f}")
    env.close()


if __name__ == "__main__":
    main()
