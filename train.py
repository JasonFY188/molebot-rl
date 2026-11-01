"""Unified trainer for SAC / TD3 / PPO on the molebot.

All three algorithms share the exact same env + reward, so the learning curves
are directly comparable.  Examples:

    # Phase 1 - dig through the pile (single sim)
    python train.py --algo sac --task cross --timesteps 300000

    # Parallel: 6 headless CoppeliaSim workers -> uses many CPU cores, trains far faster
    python train.py --algo sac --task cross --timesteps 600000 --n-envs 6 --grains 4,4,2

    # Phase 2 - straight line, warm-started from the cross checkpoint (same algo)
    python train.py --algo sac --task straight --timesteps 200000 \
                    --init-from runs/cross/sac/sac_final.zip --n-envs 6

    # watch it train (only with --n-envs 1):  add  --gui
    # attach to a running CoppeliaSim:          add  --attach
"""
from __future__ import annotations

import os
import sys
import copy
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import default_config
from molebot_env import MolebotEnv

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}

# Each parallel worker gets its own ZMQ port, spaced by 5 (CoppeliaSim also uses port+1).
PORT_BASE = 23100
PORT_STRIDE = 5


def build_model(algo: str, env, seed: int, tb: str, n_envs: int, net):
    # batch_size larger + gradient_steps scaled with n_envs -> keeps the GPU busier and
    # keeps the update:sample ratio ~1:1 when collecting from many envs at once.
    common = dict(policy="MlpPolicy", env=env, verbose=1, seed=seed,
                  tensorboard_log=tb, device="auto", gamma=0.99,
                  policy_kwargs=dict(net_arch=net))
    if algo == "sac":
        return SAC(**common, learning_rate=3e-4, buffer_size=500_000, batch_size=512,
                   tau=0.005, train_freq=1, gradient_steps=max(1, n_envs),
                   learning_starts=1000, ent_coef="auto")
    if algo == "td3":
        na = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(na), 0.1 * np.ones(na))
        if n_envs > 1:
            noise = VectorizedActionNoise(noise, n_envs)
        return TD3(**common, learning_rate=1e-3, buffer_size=500_000, batch_size=512,
                   tau=0.005, train_freq=1, gradient_steps=max(1, n_envs),
                   learning_starts=1000, action_noise=noise)
    if algo == "ppo":
        # Six workers collect 3,072 transitions per rollout here, so PPO improves
        # its policy regularly instead of waiting for a very large 12,288-step batch.
        return PPO(**common, learning_rate=3e-4, n_steps=512, batch_size=512,
                   n_epochs=10, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=list(ALGOS), required=True)
    ap.add_argument("--task", choices=["cross", "straight"], default="cross")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel CoppeliaSim workers (uses more CPU cores; 4-8 is sensible)")
    ap.add_argument("--big-net", action="store_true",
                    help="use a larger [512,512] network (more GPU use; rarely helps this task)")
    ap.add_argument("--scene", default=None, help="override scene .ttt path")
    ap.add_argument("--gui", action="store_true", help="render the simulator (only with --n-envs 1)")
    ap.add_argument("--attach", action="store_true",
                    help="connect to an already-running CoppeliaSim (only with --n-envs 1)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start policy weights from this checkpoint (same algo)")
    ap.add_argument("--grains", default=None, help="pile size 'nx,ny,nz' e.g. 4,4,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="sub-folder label for this run (e.g. 's0'); use different tags "
                         "per seed so compare.py can draw mean±std bands")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=25_000)
    args = ap.parse_args()

    n_envs = max(1, args.n_envs)
    cfg = default_config()
    cfg.task = args.task
    cfg.seed = args.seed
    if args.scene:
        cfg.sim.scene = args.scene
    cfg.sim.headless = not args.gui
    cfg.sim.launch = not args.attach
    if args.grains:
        cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = (int(x) for x in args.grains.split(","))

    if n_envs > 1 and (args.gui or args.attach):
        print("[train] --gui/--attach are ignored with --n-envs > 1 (workers run headless & launched)")
        cfg.sim.headless = True
        cfg.sim.launch = True

    save_dir = args.save_dir or os.path.join(os.path.dirname(__file__), "runs", args.task, args.algo)
    if args.tag and not args.save_dir:
        save_dir = os.path.join(save_dir, args.tag)   # e.g. runs/cross/sac/s0
    os.makedirs(save_dir, exist_ok=True)
    tb = os.path.join(save_dir, "tb")
    info_kw = ("is_success",) if args.task == "cross" else ()

    def make_worker(rank: int):
        def _init():
            c = copy.deepcopy(cfg)
            if n_envs > 1:                      # deterministic, collision-free ports per worker
                c.sim.auto_port = False
                c.sim.port = PORT_BASE + rank * PORT_STRIDE
            e = MolebotEnv(c, task=args.task)
            name = "monitor.csv" if n_envs == 1 else f"monitor_{rank}.csv"
            return Monitor(e, os.path.join(save_dir, name), info_keywords=info_kw)
        return _init

    print(f"[train] algo={args.algo} task={args.task} timesteps={args.timesteps} "
          f"n_envs={n_envs} headless={cfg.sim.headless}")
    print(f"[train] launching {n_envs} CoppeliaSim worker(s)... (this takes ~15-30s)")

    VecCls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    venv = VecCls([make_worker(i) for i in range(n_envs)])

    try:
        joint_alias = venv.get_attr("joint_alias")[0]
        print(f"[train] discovered {len(joint_alias)} joints: {joint_alias}")
    except Exception as e:  # noqa: BLE001
        joint_alias = None
        print(f"[train] (could not read joint aliases via get_attr: {e})")
    print(f"[train] obs_dim={venv.observation_space.shape[0]} "
          f"action_dim={venv.action_space.shape[0]}")

    net = [512, 512] if args.big_net else [256, 256]
    model = build_model(args.algo, venv, args.seed, tb, n_envs, net)
    if args.init_from:
        if not (os.path.isfile(args.init_from) or os.path.isfile(args.init_from + ".zip")):
            venv.close()
            raise SystemExit(
                f"[train] --init-from file not found: {args.init_from}\n"
                f"        Omit --init-from to train from scratch, or point it at a real "
                f"checkpoint, e.g. runs/cross/{args.algo}/{args.algo}_final.zip")
        model.set_parameters(args.init_from)
        print(f"[train] warm-started weights from {args.init_from}")

    # CheckpointCallback counts vec-steps; divide the env-step budget by n_envs.
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_freq // n_envs, 1),
                              save_path=save_dir, name_prefix=args.algo)

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=ckpt)
    finally:
        final = os.path.join(save_dir, f"{args.algo}_final")
        model.save(final)
        meta = dict(algo=args.algo, task=args.task, timesteps=args.timesteps, seed=args.seed,
                    n_envs=n_envs, net_arch=net,
                    obs_dim=int(venv.observation_space.shape[0]),
                    n_act=int(venv.action_space.shape[0]),
                    joint_alias=joint_alias, wall_time_s=round(time.time() - t0, 1))
        json.dump(meta, open(os.path.join(save_dir, "meta.json"), "w"), indent=2)
        venv.close()
        print(f"[train] saved final model -> {final}.zip  ({meta['wall_time_s']}s)")


if __name__ == "__main__":
    main()
