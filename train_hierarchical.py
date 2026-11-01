"""Trainer for the two hierarchical SAC policies (locomotion, drilling).

    # locomotion: pure point-to-point nav (no pile), learns forward/back/turn
    python train_hierarchical.py --policy locomotion --timesteps 40000 --n-envs 6

    # drilling: robot pre-positioned at the pile face, learns stab/clear/retract
    python train_hierarchical.py --policy drilling --timesteps 30000 --n-envs 6 --grains 6,6,3

Each saves to runs/hier/<policy>/. Once both are trained, run
evaluate_hierarchical.py to watch them work together (rule-based switcher)
attempt a full crossing.

    # watch it train (only with --n-envs 1):  add --gui
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config
from hierarchical_env import LocomotionEnv, DrillingEnv

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ENVS = {"locomotion": LocomotionEnv, "drilling": DrillingEnv}


def build_model(env, seed: int, tb: str, n_envs: int):
    # Smaller net + buffer than the flat trainer: these are short macro-action
    # episodes (<=25 decisions) over a small obs/action space.
    return SAC(policy="MlpPolicy", env=env, verbose=1, seed=seed, tensorboard_log=tb,
              device="auto", gamma=0.98, policy_kwargs=dict(net_arch=[128, 128]),
              learning_rate=3e-4, buffer_size=100_000, batch_size=256, tau=0.005,
              train_freq=1, gradient_steps=max(1, n_envs), learning_starts=500,
              ent_coef="auto")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=list(ENVS), required=True)
    ap.add_argument("--timesteps", type=int, default=40_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--gui", action="store_true", help="render (only with --n-envs 1)")
    ap.add_argument("--grains", default=None, help="drilling only: pile size 'nx,ny,nz'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=5_000)
    args = ap.parse_args()

    n_envs = max(1, args.n_envs)
    EnvCls = ENVS[args.policy]
    save_dir = args.save_dir or os.path.join(os.path.dirname(__file__), "runs", "hier", args.policy)
    os.makedirs(save_dir, exist_ok=True)
    tb = os.path.join(save_dir, "tb")

    def make_worker(rank: int):
        def _init():
            cfg = default_config()
            cfg.sim.headless = not args.gui
            cfg.sim.launch = True
            if args.grains and args.policy == "drilling":
                cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = (int(x) for x in args.grains.split(","))
            e = EnvCls(cfg)
            name = "monitor.csv" if n_envs == 1 else f"monitor_{rank}.csv"
            return Monitor(e, os.path.join(save_dir, name), info_keywords=("is_success",))
        return _init

    if n_envs > 1 and args.gui:
        print("[train_hier] --gui is ignored with --n-envs > 1 (workers run headless)")

    print(f"[train_hier] policy={args.policy} timesteps={args.timesteps} n_envs={n_envs}")
    print(f"[train_hier] launching {n_envs} CoppeliaSim worker(s)...")
    VecCls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    venv = VecCls([make_worker(i) for i in range(n_envs)])
    print(f"[train_hier] obs_dim={venv.observation_space.shape[0]} action_dim={venv.action_space.shape[0]}")

    model = build_model(venv, args.seed, tb, n_envs)
    if args.init_from:
        if not (os.path.isfile(args.init_from) or os.path.isfile(args.init_from + ".zip")):
            venv.close()
            raise SystemExit(f"[train_hier] --init-from not found: {args.init_from}")
        model.set_parameters(args.init_from)
        print(f"[train_hier] warm-started from {args.init_from}")

    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_freq // n_envs, 1),
                              save_path=save_dir, name_prefix=args.policy)
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=ckpt)
    finally:
        final = os.path.join(save_dir, f"{args.policy}_final")
        model.save(final)
        meta = dict(policy=args.policy, timesteps=args.timesteps, seed=args.seed, n_envs=n_envs,
                    obs_dim=int(venv.observation_space.shape[0]),
                    action_dim=int(venv.action_space.shape[0]),
                    wall_time_s=round(time.time() - t0, 1))
        json.dump(meta, open(os.path.join(save_dir, "meta.json"), "w"), indent=2)
        venv.close()
        print(f"[train_hier] saved -> {final}.zip ({meta['wall_time_s']}s)")


if __name__ == "__main__":
    main()
