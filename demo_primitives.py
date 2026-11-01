"""Watch the 7 scripted primitives run, one after another, in the CoppeliaSim GUI.

    python demo_primitives.py            # GUI, watch it
    python demo_primitives.py --headless # fast, no window (just checks it runs)
    python demo_primitives.py --grains 8,8,4

Sequence: forward, backward, turn-left, turn-right, drive-to-pile,
stab (both arms), clear (both arms), retract, one more stab.
"""
from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config
import coppelia as cp
from controller import MolebotController


EXPECTED_STRUCTURE = {
    "R": {
        "branch": ["robot_body", "shoulder_R", "knee_R", "extender_R", "arm_R", "wrist_R", "half_cone_R"],
        "joints": {"shoulder": "shoulder_R", "prismatic": "extender_R", "wrist": "wrist_R"},
        "cone": "half_cone_R",
    },
    "L": {
        "branch": ["robot_body", "shoulder_L", "knee_L", "extender_L", "arm_L", "wrist_L", "half_cone_L"],
        "joints": {"shoulder": "shoulder_L", "prismatic": "extender_L", "wrist": "wrist_L"},
        "cone": "half_cone_L",
    },
}


def _alias(ctrl, handle):
    """Return the readable CoppeliaSim alias for one object handle."""
    return cp.object_alias(ctrl.sim, handle)


def _branch_from_base(ctrl, cone):
    """Return the selected cone's hierarchy, from robot_body to the cone."""
    branch = []
    handle = cone
    while handle is not None and handle >= 0:
        branch.append(_alias(ctrl, handle))
        if handle == ctrl.base:
            break
        handle = ctrl.sim.getObjectParent(handle)
    return list(reversed(branch))


def verify_scene_structure(ctrl):
    """Print and validate the robot hierarchy used by the motion primitives.

    The shape objects knee_* and arm_* are intentionally listed too. The controller
    does not actuate shapes, but seeing the complete branch makes it easy to compare
    this output with the CoppeliaSim scene hierarchy.
    """
    errors = []
    print("\n=== Molebot structure check ===")
    print("Expected: robot_body -> shoulder -> knee -> extender -> arm -> wrist -> half_cone")

    base_name = _alias(ctrl, ctrl.base)
    print(f"Base: {base_name}")
    if base_name != "robot_body":
        errors.append(f"base is '{base_name}', expected 'robot_body'")

    for side, expected in EXPECTED_STRUCTURE.items():
        arm = ctrl.arms.get(side)
        if arm is None:
            errors.append(f"{side} arm was not discovered")
            continue

        actual_branch = _branch_from_base(ctrl, arm["cone"])
        print(f"{side} expected: {' -> '.join(expected['branch'])}")
        print(f"{side} found:    {' -> '.join(actual_branch)}")
        if actual_branch != expected["branch"]:
            errors.append(f"{side} branch does not match the expected hierarchy")

        for key, expected_name in expected["joints"].items():
            handle = arm[key]
            actual_name = _alias(ctrl, handle) if handle is not None else "<missing>"
            print(f"  {side} {key:10s}: {actual_name}")
            if actual_name != expected_name:
                errors.append(f"{side} {key} is '{actual_name}', expected '{expected_name}'")

        cone_name = _alias(ctrl, arm["cone"])
        print(f"  {side} cone       : {cone_name}")
        if cone_name != expected["cone"]:
            errors.append(f"{side} cone is '{cone_name}', expected '{expected['cone']}'")

        for key in ("shoulder", "wrist"):
            if ctrl.sim.getJointType(arm[key]) == ctrl.C["pris"]:
                errors.append(f"{side} {key} must be a revolute joint, not prismatic")
        if ctrl.sim.getJointType(arm["prismatic"]) != ctrl.C["pris"]:
            errors.append(f"{side} extender must be a prismatic joint")

    left_wheels = sorted(_alias(ctrl, h) for h in ctrl.wheels_left)
    right_wheels = sorted(_alias(ctrl, h) for h in ctrl.wheels_right)
    print(f"Wheels left : {', '.join(left_wheels) or '<missing>'}")
    print(f"Wheels right: {', '.join(right_wheels) or '<missing>'}")
    if set(left_wheels) != {"front_left_wheel_joint", "back_left_wheel_joint"}:
        errors.append("left wheel joint names do not match the current robot")
    if set(right_wheels) != {"front_right_wheel_joint", "back_right_wheel_joint"}:
        errors.append("right wheel joint names do not match the current robot")

    if errors:
        raise RuntimeError("Scene structure check failed:\n- " + "\n- ".join(errors))
    print("Structure check passed. The controller is using the intended joints.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--attach", action="store_true",
                    help="use the CoppeliaSim GUI scene already open on port 23000")
    ap.add_argument("--check-only", action="store_true",
                    help="print the robot hierarchy check, then exit without starting simulation")
    ap.add_argument("--scene", default=None,
                    help="scene file to launch instead of the configured default")
    ap.add_argument("--grains", default=None, help="pile size 'nx,ny,nz'")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between primitives")
    args = ap.parse_args()

    cfg = default_config()
    cfg.sim.rpc_timeout_s = 120.0   # generous: a heavy contact step just pauses, never crashes
    if args.attach:
        cfg.sim.launch = False
        cfg.sim.load_scene_on_attach = False
    if args.scene:
        cfg.sim.scene = args.scene
    if args.grains:
        cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = (int(x) for x in args.grains.split(","))
    else:
        cfg.grain.nx, cfg.grain.ny, cfg.grain.nz = 4, 4, 2   # small pile = stable demo
    ctrl = MolebotController(cfg, gui=not args.headless)
    try:
        verify_scene_structure(ctrl)
    except Exception:
        ctrl.close()
        raise
    if args.check_only:
        print("Check-only mode: scene was not started or changed.")
        ctrl.close()
        return
    ctrl.reset()
    # confirm turn direction once
    ctrl.turn_sign, dy = ctrl.calibrate_turn(20)
    print(f"turn calibration: {dy:+.0f} deg -> sign={ctrl.turn_sign}")
    ctrl.reset()

    def step(msg, fn):
        print(f"\n>>> {msg}", flush=True)
        try:
            print("   ", fn())
        except Exception as e:  # noqa: BLE001
            print("   !! primitive error:", type(e).__name__, str(e)[:80])
        time.sleep(args.pause)

    step("DRIVE FORWARD 0.8 m", lambda: ctrl.drive(0.8))
    step("DRIVE BACKWARD 0.5 m", lambda: ctrl.drive(-0.5))
    step("TURN LEFT 60 deg", lambda: ctrl.turn_left(60, sign=ctrl.turn_sign))
    step("TURN RIGHT 60 deg", lambda: ctrl.turn_right(60, sign=ctrl.turn_sign))
    # drive gently UP TO the pile face (not ramming through it)
    step("DRIVE UP TO THE PILE", lambda: ctrl.drive(0.5, speed=2.5))
    step("STAB R: extender_R pushes half_cone_R forward", lambda: ctrl.stab("R"))
    step("STAB L: extender_L pushes half_cone_L forward", lambda: ctrl.stab("L"))
    step("CLEAR R: shoulder_R sweeps half_cone_R outward", lambda: ctrl.clear("R", 40))
    step("CLEAR L: shoulder_L sweeps half_cone_L outward", lambda: ctrl.clear("L", 40))
    step("RETRACT: shoulders, extenders, and wrists return home", lambda: (ctrl.retract("R"), ctrl.retract("L")))
    step("RESET BALLS to the original pile", lambda: ctrl.reset_balls())
    step("STAB again (next peck)", lambda: ctrl.stab("R"))

    print("\nDemo complete.")
    time.sleep(2.0)
    ctrl.close()


if __name__ == "__main__":
    main()
