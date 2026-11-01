"""Quick sanity check before committing to a long training run.

Launches (or attaches to) CoppeliaSim, discovers the robot's joints, verifies the
grain-field signals are alive, drives all wheels forward to confirm the forward
axis, then runs a short random rollout.

    # easiest: open CoppeliaSim GUI, load the scene, then:
    python smoke_test.py --attach

    # or let it launch a GUI itself:
    python smoke_test.py
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import default_config
from molebot_env import MolebotEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--attach", action="store_true",
                    help="connect to a running CoppeliaSim (default: launch a GUI)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--task", default="cross", choices=["cross", "straight"])
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    cfg = default_config()
    if args.scene:
        cfg.sim.scene = args.scene
    if args.port:
        cfg.sim.port = args.port
    cfg.sim.launch = not args.attach
    cfg.sim.headless = args.headless  # default False -> you can watch it

    print("Bringing up environment...")
    env = MolebotEnv(cfg, task=args.task)
    print(f"  base object    : {cfg.robot.base_path}")
    print(f"  joints ({env.n_act}) : {env.joint_alias}")
    print(f"  is_wheel        : {env.is_wheel}")
    print(f"  max_vel         : {np.round(env.max_vel, 3).tolist()}")
    print(f"  obs_dim={env.obs_dim}  action_dim={env.n_act}")

    obs, _ = env.reset()
    assert obs.shape == (env.obs_dim,), obs.shape
    st = env._read_state()
    print(f"\ngrain signals   : disturbed={st['grain_disturbed']:.3f}  com={np.round(st['grain_com'],3).tolist()}")
    if not np.any(st["grain_com"]):
        print("  !! grain COM is all-zero. Did you paste scene/grain_field.lua onto the 'dummy' script?")

    # ---- forward-axis check: drive every wheel +1, arms 0 ----
    a = np.zeros(env.n_act, np.float32)
    for i, w in enumerate(env.is_wheel):
        if w:
            a[i] = 1.0
    p0 = env._read_state()["pos"][:2].copy()
    for _ in range(20):
        env.step(a)
    p1 = env._read_state()["pos"][:2]
    disp = p1 - p0
    fwd = env.forward0
    left = env.left0
    along = float(np.dot(disp, fwd))
    lateral = float(np.dot(disp, left))
    print(f"\nall-wheels-forward test:")
    print(f"  world displacement : {np.round(disp,3).tolist()} m")
    print(f"  along forward axis : {along:+.3f} m   lateral: {lateral:+.3f} m")
    if along < 0.02:
        print("  NOTE: not moving forward. If it drove backward -> flip robot.forward_axis in config.py.")
        print("        If it spun in place -> left/right wheels spin opposite; see README 'skid-steer signs'.")
    else:
        print("  OK: forward axis looks correct.")

    # ---- short random rollout ----
    obs, _ = env.reset()
    ret, steps = 0.0, 0
    for _ in range(args.steps):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        ret += rew
        steps += 1
        if term or trunc:
            break
    print(f"\nrandom rollout: return={ret:.2f} over {steps} steps  last_info={info}")
    print("Smoke test done.")
    env.close()


if __name__ == "__main__":
    main()
