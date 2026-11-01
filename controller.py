"""Scripted motion-primitive library for the molebot.

This is the low-level layer under the (planned) two RL policies. It brings up one
CoppeliaSim instance, finds the wheels (left/right) and each arm's joints
(shoulder / prismatic / wrist / cone), and exposes 7 primitives:

  locomotion:  forward, backward, turn_left, turn_right   (wheels; turn = heading PID)
  drilling:    stab (刺す), clear (どける), retract         (arm; PID, wheels braked)

Kinematics (measured, forward = body -y):
  prismatic  -> cone moves 1:1 forward   => STAB
  shoulder   -> cone swings sideways      => CLEAR (sweep)
  wrist      -> cone swings sideways      => fine clear

Use demo_primitives.py to watch these run in the GUI.
"""
from __future__ import annotations

import time
import numpy as np

import coppelia as cp


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class MolebotController:
    def __init__(self, cfg, gui: bool | None = None):
        self.cfg = cfg
        if gui is not None:
            cfg.sim.headless = not gui
        self._proc = None
        self.client = None
        self.sim = None
        self.port = cfg.sim.port
        self._connect_simulator()

    def _connect_simulator(self):
        """(Re)launch + connect. Split out of __init__ so a crashed/timed-out worker
        can be replaced in place via restart_simulator() without losing the caller's
        reference to this MolebotController (evaluate_hierarchical.py shares one
        instance between LocomotionEnv and DrillingEnv)."""
        port = self.cfg.sim.port
        if self.cfg.sim.launch:
            if self.cfg.sim.auto_port:
                port = cp.find_free_port()
            self._proc = cp.launch_coppelia(self.cfg.sim.coppelia_exe, self.cfg.sim.scene,
                                            port, self.cfg.sim.headless)
        self.port = port
        self.client, self.sim = cp.connect(port, self.cfg.sim.connect_timeout,
                                            getattr(self.cfg.sim, "rpc_timeout_s", 20.0))
        self._resolve()
        self._discover()

    def can_recover(self, exc: BaseException) -> bool:
        """True if `exc` is a ZMQ RPC timeout on a worker we launched ourselves (so we
        can replace it) and recovery is enabled -- matches molebot_env.py's watchdog."""
        return bool(self.cfg.sim.launch
                    and getattr(self.cfg.sim, "restart_on_rpc_timeout", True)
                    and cp.is_rpc_timeout(exc))

    def restart_simulator(self, exc: BaseException):
        """Replace a launched worker after its ZMQ request timed out, so one flaky
        RPC hiccup does not kill a multi-hour training run."""
        print(f"[controller watchdog] RPC timed out on port {self.port}; "
              f"restarting worker ({exc}).", flush=True)
        cp.close_client(self.client)
        self.client = None
        self.sim = None
        if self._proc is not None:
            cp.terminate_process(self._proc)
            self._proc = None
        self._connect_simulator()

    # ------------------------------------------------------------------ setup
    def _resolve(self):
        s = self.sim
        self.C = dict(
            joint=cp.const(s, "object_joint_type"),
            shape=cp.const(s, "object_shape_type", default=0),
            all=cp.const(s, "handle_all"),
            pris=cp.const(s, "joint_prismatic_subtype", "joint_prismatic"),
            stopped=cp.const(s, "simulation_stopped"),
            dynmode=cp.const(s, "jointintparam_dynctrlmode", default=-1),
            velmode=cp.const(s, "jointdynctrl_velocity", default=-1),
            posmode=cp.const(s, "jointdynctrl_position", default=-1),
            respond=cp.const(s, "shapeintparam_respondable", default=-1),
        )

    def _discover(self):
        s, C = self.sim, self.C
        self.base = s.getObject(self.cfg.robot.base_path)
        tree = s.getObjectsInTree(self.base, C["all"], 0)

        self.wheels_left, self.wheels_right = [], []
        self.cone_shapes = []
        for h in tree:
            t = s.getObjectType(h)
            if t == C["joint"]:
                a = cp.object_alias(s, h).lower()
                if self.cfg.robot.wheel_name_hint in a:
                    (self.wheels_left if "left" in a else self.wheels_right).append(h)
            elif (t == C["shape"] and self.cfg.robot.disable_cone_collision
                  and self.cfg.robot.cone_name_hint in cp.object_alias(s, h).lower()):
                try:
                    _, idx, _ = s.getShapeMesh(h); ntri = len(idx) // 3
                except Exception:  # noqa: BLE001
                    ntri = 10 ** 9
                if ntri > self.cfg.robot.cone_max_tris:
                    self.cone_shapes.append(h)
        self.wheels = self.wheels_left + self.wheels_right

        # per-arm joints: walk cone -> base, chain = [wrist, prismatic, shoulder]
        self.arms = {}
        for side, name in (("R", "half_cone_R"), ("L", "half_cone_L")):
            try:
                cone = s.getObject("/" + name)
            except Exception:  # noqa: BLE001
                continue
            chain = []
            h = s.getObjectParent(cone)
            while h is not None and h >= 0 and h != self.base:
                if s.getObjectType(h) == C["joint"]:
                    chain.append(h)
                h = s.getObjectParent(h)
            # chain[0]=wrist(nearest cone), find prismatic, shoulder(nearest base)
            pris = next((j for j in chain if s.getJointType(j) == C["pris"]), None)
            revs = [j for j in chain if s.getJointType(j) != C["pris"]]
            self.arms[side] = dict(cone=cone, wrist=chain[0] if chain else None,
                                   prismatic=pris, shoulder=revs[-1] if revs else None,
                                   all=chain)
        self.arm_joints = [j for a in self.arms.values() for j in a["all"] if j is not None]

    # ------------------------------------------------------------------ helpers
    def _vel_mode(self, joints):
        s, C = self.sim, self.C
        for h in joints:
            if C["dynmode"] != -1 and C["velmode"] != -1:
                try: s.setObjectInt32Param(h, C["dynmode"], C["velmode"])
                except Exception: pass

    def reset(self):
        s, C = self.sim, self.C
        if s.getSimulationState() != C["stopped"]:
            s.stopSimulation()
            while s.getSimulationState() != C["stopped"]:
                time.sleep(0.02)
        g = self.cfg.grain
        for k, v in (("grain_nx", g.nx), ("grain_ny", g.ny), ("grain_nz", g.nz)):
            cp.set_int_signal(s, k, v)
        for k, v in (("grain_d", g.diameter), ("grain_spacing", g.spacing),
                     ("grain_cx", g.center_x), ("grain_cy", g.center_y),
                     ("grain_floorz", g.floor_z), ("grain_disturb_thresh", g.disturb_thresh)):
            cp.set_float_signal(s, k, v)
        if C["respond"] != -1:
            for h in self.cone_shapes:
                try: s.setObjectInt32Param(h, C["respond"], 0)
                except Exception: pass
        try: s.setBoolParam(s.boolparam_realtime_simulation, False)
        except Exception: pass
        s.setStepping(True)
        s.startSimulation()
        self._vel_mode(self.wheels + self.arm_joints)
        wforce = getattr(self.cfg.robot, "wheel_locomotion_force", self.cfg.robot.wheel_max_force)
        for h in self.wheels:
            try: s.setJointTargetForce(h, wforce, False)
            except Exception: pass
        for h in self.arm_joints:
            try: s.setJointTargetForce(h, self.cfg.robot.arm_max_force, False)
            except Exception: pass
        self.brake_wheels()
        self._lock_arms()
        for _ in range(self.cfg.sim.step_settle):
            s.step()

    def _step(self, n=1):
        for _ in range(n):
            self.sim.step()

    def base_pose(self):
        p = self.sim.getObjectPose(self.base)
        R = cp.quat_to_matrix(p[3:7])
        _, _, yaw = cp.roll_pitch_yaw(R, self.cfg.robot.forward_axis)
        return np.array(p[:3]), yaw

    def yaw(self):
        return self.base_pose()[1]

    def forward_vec(self):
        y = self.yaw()
        return np.array([np.cos(y), np.sin(y)])

    def jpos(self, h):
        return float(self.sim.getJointPosition(h))

    def jforce(self, h):
        try: return float(self.sim.getJointForce(h))
        except Exception: return 0.0

    def _set_wheels(self, left_v, right_v):
        for h in self.wheels_left:
            try: self.sim.setJointTargetVelocity(h, float(left_v))
            except Exception: pass
        for h in self.wheels_right:
            try: self.sim.setJointTargetVelocity(h, float(right_v))
            except Exception: pass

    def brake_wheels(self):
        self._set_wheels(0.0, 0.0)

    def _hold_arms(self):
        for h in self.arm_joints:
            try: self.sim.setJointTargetVelocity(h, 0.0)
            except Exception: pass

    def _lock_arms(self, target=0.0):
        """Brake the arms (velocity 0 with a firm force) so the cones don't drift
        while the body accelerates/turns -- the arm's 'do nothing' state. A velocity
        brake (not position control) avoids PID over-actuation / blow-ups."""
        s, C = self.sim, self.C
        hf = getattr(self.cfg.robot, "arm_hold_force", 60.0)
        for h in self.arm_joints:
            if C["dynmode"] != -1 and C["velmode"] != -1:
                try: s.setObjectInt32Param(h, C["dynmode"], C["velmode"])
                except Exception: pass
            try: s.setJointTargetForce(h, hf, False)
            except Exception: pass
            try: s.setJointTargetVelocity(h, 0.0)
            except Exception: pass

    def _ensure_vel(self, joints, force=None):
        """Put joints back into velocity-control mode (needed after a position lock)."""
        s, C = self.sim, self.C
        for h in joints:
            if C["dynmode"] != -1 and C["velmode"] != -1:
                try: s.setObjectInt32Param(h, C["dynmode"], C["velmode"])
                except Exception: pass
            if force is not None:
                try: s.setJointTargetForce(h, force, False)
                except Exception: pass

    # ------------------------------------------------------------------ locomotion
    def drive(self, distance=0.5, speed=None, max_steps=400):
        """Drive straight; +distance = forward (body -y), -distance = backward."""
        v = (speed if speed is not None else self.cfg.robot.wheel_max_vel) * np.sign(distance)
        p0, _ = self.base_pose()
        fwd = self.forward_vec()
        self._lock_arms()
        moved = 0.0
        for i in range(max_steps):
            self._set_wheels(v, v)
            self._step()
            p, _ = self.base_pose()
            if not np.all(np.isfinite(p)) or float(np.max(np.abs(p[:2]))) > 50.0:
                self.brake_wheels()
                return {"moved": moved, "steps": i + 1, "diverged": True}
            moved = float(np.dot((p - p0)[:2], fwd))
            if abs(moved) >= abs(distance):
                break
        self.brake_wheels()
        return {"moved": moved, "steps": i + 1}

    def turn(self, dyaw_deg=45.0, kp=2.0, tol_deg=3.0, max_steps=400, sign=1.0):
        """Turn in place by dyaw_deg (heading PID, differential wheels).
        `sign` sets which wheel side leads; calibrate once with calibrate_turn()."""
        target = _wrap(self.yaw() + np.deg2rad(dyaw_deg))
        tol = np.deg2rad(tol_deg)
        self._lock_arms()
        err = 1e9
        for i in range(max_steps):
            p, yaw = self.base_pose()
            if not np.all(np.isfinite(p)) or float(np.max(np.abs(p[:2]))) > 50.0:
                self.brake_wheels()
                return {"yaw_err_deg": float(np.rad2deg(err)), "steps": i + 1, "diverged": True}
            err = _wrap(target - yaw)
            if abs(err) < tol:
                break
            turn_v = min(self.cfg.robot.wheel_max_vel, 3.5)   # gentler turn = less arm drift + smoother
            cmd = float(np.clip(kp * err, -1.0, 1.0)) * turn_v * sign
            self._set_wheels(-cmd, cmd)     # differential (skid steer)
            self._step()
        self.brake_wheels()
        return {"yaw_err_deg": float(np.rad2deg(err)), "steps": i + 1}

    def turn_left(self, deg=45.0, **kw):
        return self.turn(+abs(deg), **kw)

    def turn_right(self, deg=45.0, **kw):
        return self.turn(-abs(deg), **kw)

    def calibrate_turn(self, steps=25):
        """Return +1/-1: the sign that makes turn() rotate the correct way."""
        y0 = self.yaw()
        for _ in range(steps):
            self._set_wheels(-self.cfg.robot.wheel_max_vel, self.cfg.robot.wheel_max_vel)
            self._step()
        dy = _wrap(self.yaw() - y0)
        self.brake_wheels()
        return 1.0 if dy > 0 else -1.0, float(np.rad2deg(dy))

    # ------------------------------------------------------------------ drilling
    def stab(self, side="R", speed=None, force=None, max_steps=60, stop_force=None):
        """Extend the prismatic (cone forward into pile). `force` (N) is the thrust
        force cap = how hard it pushes the balls. Optionally stop at a resistance
        threshold (stop_force, N) read from the joint."""
        j = self.arms[side]["prismatic"]
        v = speed if speed is not None else self.cfg.robot.arm_pris_max_vel
        f = force if force is not None else getattr(self.cfg.robot, "arm_stab_force", 30.0)
        self._ensure_vel([j], force=f)   # restore velocity control (arm may be position-locked)
        self.brake_wheels()
        f = 0.0
        for i in range(max_steps):
            self.sim.setJointTargetVelocity(j, +abs(v))
            self._step()
            f = self.jforce(j)
            if stop_force is not None and abs(f) > stop_force:
                break
        self.sim.setJointTargetVelocity(j, 0.0)
        return {"pos": self.jpos(j), "contact_force": f, "steps": i + 1}

    def clear(self, side="R", angle_deg=35.0, kp=3.0, tol_deg=4.0, max_steps=120):
        """Sweep the near-body shoulder OUTWARD to push balls aside. Each cone stays on
        its OWN side (R -> robot's right, L -> robot's left) instead of crossing over the
        centre into the other cone (which collides -> crash). angle_deg>0 = further out."""
        outward = -1.0 if side == "L" else 1.0    # measured: shoulder+ moves cone +x (left)
        target = np.deg2rad(angle_deg) * outward
        return self._pid_joint(self.arms[side]["shoulder"], target,
                               self.cfg.robot.arm_rev_max_vel, kp, tol_deg=tol_deg,
                               max_steps=max_steps, brake=True)

    def retract(self, side="R", kp=4.0, max_steps=200, tol=8e-3):
        """Home prismatic + shoulder + wrist to 0 all at once (fast). Uses a firmer
        force than clear so it can pull the cone back out of the balls."""
        a = self.arms[side]
        joints = [j for j in (a["prismatic"], a["shoulder"], a["wrist"]) if j is not None]
        force = max(getattr(self.cfg.robot, "arm_stab_force", 30.0), 30.0)
        self._ensure_vel(joints, force=force)
        pv = max(self.cfg.robot.arm_pris_max_vel * 3, 0.10)
        steps = max_steps
        for i in range(max_steps):
            self.brake_wheels()
            done = True
            for j in joints:
                mv = pv if j == a["prismatic"] else self.cfg.robot.arm_rev_max_vel
                err = 0.0 - self.jpos(j)
                self.sim.setJointTargetVelocity(j, float(np.clip(kp * err, -mv, mv)))
                if abs(err) > tol:
                    done = False
            self._step()
            if done:
                steps = i + 1
                break
        for j in joints:
            self.sim.setJointTargetVelocity(j, 0.0)
        return {"pris_pos": self.jpos(a["prismatic"]), "steps": steps}

    def _pid_joint(self, h, target, max_vel, kp, ki=0.0, tol=None, tol_deg=None,
                   max_steps=120, brake=False):
        if tol is None:
            tol = np.deg2rad(tol_deg) if tol_deg is not None else 1e-2
        # restore velocity control at a gentle force (not the strong hold-brake force)
        self._ensure_vel([h], force=self.cfg.robot.arm_max_force)
        integ = 0.0
        err = 1e9
        for i in range(max_steps):
            if brake:
                self.brake_wheels()
            err = target - self.jpos(h)
            integ += err
            cmd = float(np.clip(kp * err + ki * integ, -max_vel, max_vel))
            self.sim.setJointTargetVelocity(h, cmd)
            self._step()
            if abs(err) < tol:
                break
        self.sim.setJointTargetVelocity(h, 0.0)
        return {"pos": self.jpos(h), "err": float(err), "steps": i + 1}

    def grain_disturbed(self):
        return cp.get_float_signal(self.sim, "grain_disturbed") or 0.0

    def reset_balls(self):
        """Restore the original pile. CoppeliaSim won't let a script move/recreate
        dynamic shapes mid-simulation, so a partial in-place reset does not stick -
        the only reliable way is a full sim reset (this also returns the robot to
        its start pose)."""
        self.reset()
        return {"grain_disturbed": self.grain_disturbed()}

    def close(self):
        try:
            if self.sim.getSimulationState() != self.C["stopped"]:
                self.sim.stopSimulation()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            cp.terminate_process(self._proc)
            self._proc = None
