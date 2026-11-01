"""Run the trained locomotion + drilling policies TOGETHER, on ONE shared
robot/simulator, with a simple rule-based switcher -- to see whether the
molebot can cross the pile end to end.

    python evaluate_hierarchical.py --loco runs/hier/locomotion/locomotion_final.zip \
                                    --drill runs/hier/drilling/drilling_final.zip

Switcher logic: run locomotion toward the far-side goal; if a forward attempt
stalls (barely moved -> blocked by the pile), switch to drilling until it
picks DONE, then switch back to locomotion. Repeat until the goal is reached,
the robot tips/diverges, or --max-switches is hit.
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config
from hierarchical_env import LocomotionEnv, DrillingEnv
from stable_baselines3 import SAC


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loco", required=True)
    ap.add_argument("--drill", required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--grains", default="6,6,3")
    ap.add_argument("--max-switches", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    cfg = default_config()
    cfg.sim.headless = args.headless
    cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = (int(x) for x in args.grains.split(","))

    print(f"[eval_hier] loco={args.loco}  drill={args.drill}  grains={args.grains}")
    loco_env = LocomotionEnv(cfg, standalone=False)       # keep the real pile (owns + launches the sim)
    drill_env = DrillingEnv(cfg, ctrl=loco_env.ctrl)      # SAME robot, SAME simulator
    loco_model = SAC.load(args.loco, device="auto")
    drill_model = SAC.load(args.drill, device="auto")

    obs, _ = loco_env.reset()
    ctrl = loco_env.ctrl
    p0, _ = ctrl.base_pose()
    print(f"start pos={p0[:2].round(3)}  goal={loco_env.goal_xy.round(3)}")

    mode = "locomotion"
    switches = 0
    total_reward = 0.0
    outcome = "max_switches_reached"
    while switches <= args.max_switches:
        if mode == "locomotion":
            act, _ = loco_model.predict(obs, deterministic=not args.stochastic)
            obs, rew, term, trunc, info = loco_env.step(act)
            total_reward += rew
            print(f"  [loco]  {info['primitive']:10s} dist_to_goal={info['dist_to_goal']:.2f} "
                  f"stalled={info['stalled']}  r={rew:6.2f}")
            if info["is_success"]:
                outcome = "reached_goal"
                break
            if term:  # tipped / diverged
                outcome = "locomotion_failed"
                break
            if info["stalled"]:
                print("  -> stalled (blocked by pile): switching to DRILLING")
                mode = "drilling"
                switches += 1
                dobs, _ = drill_env.begin_episode()
        else:
            act, _ = drill_model.predict(dobs, deterministic=not args.stochastic)
            dobs, rew, term, trunc, info = drill_env.step(act)
            total_reward += rew
            print(f"  [drill] {info['primitive']:10s} grain_disturbed={info['grain_disturbed']:.2f}  "
                  f"r={rew:6.2f}")
            if term or trunc:
                if term and not trunc and info["primitive"] != "done":
                    outcome = "drilling_failed"  # tipped/diverged mid-drilling
                    break
                print("  -> drilling phase ended: switching back to LOCOMOTION")
                mode = "locomotion"
                obs = loco_env._obs()

    print(f"\noutcome={outcome}  total_reward={total_reward:.2f}  switches={switches}")
    loco_env.close()  # closes the shared simulator (drill_env does not own it)


if __name__ == "__main__":
    main()
