"""Watch ONE turn at a time to confirm left/right direction.

    python turn_check.py            # GUI
Resets between each so you see the robot turn purely left, then purely right.
"""
from __future__ import annotations
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import default_config
from controller import MolebotController

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--deg", type=float, default=90.0)
args = ap.parse_args()

cfg = default_config()
cfg.sim.rpc_timeout_s = 120.0
cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = 4, 4, 2
ctrl = MolebotController(cfg, gui=not args.headless)
ctrl.reset()
ctrl.turn_sign, _ = ctrl.calibrate_turn(20)

ctrl.reset()
print(f"\n>>> TURN LEFT {args.deg:.0f} deg  (watch: should rotate LEFT)", flush=True)
print("   ", ctrl.turn_left(args.deg, sign=ctrl.turn_sign))
time.sleep(2.5)

ctrl.reset()
print(f"\n>>> TURN RIGHT {args.deg:.0f} deg (watch: should rotate RIGHT)", flush=True)
print("   ", ctrl.turn_right(args.deg, sign=ctrl.turn_sign))
time.sleep(2.5)

print("\nIf LEFT went right / RIGHT went left, tell me and I'll flip it back.")
ctrl.close()
