"""Hierarchical macro-action Gymnasium envs built on the scripted primitive
controller (controller.py). Two SEPARATE SAC policies are trained here, each
with its own focused objective:

  LocomotionEnv - choose among forward / backward / turn-left / turn-right
                  (+ magnitude) to reach a goal point. Trained with NO pile
                  present, so it learns clean point-to-point navigation. The
                  combined evaluator (evaluate_hierarchical.py) reuses it
                  against the far-side goal and lets it naturally stall
                  against the pile, which triggers a switch to drilling.

  DrillingEnv   - choose among stab (刺す) / clear (どける) / retract / done
                  (+ stab force, stab speed) to clear a path through the pile
                  immediately in front of the robot. Wheels stay braked
                  throughout (via the primitives themselves), so the body
                  does not move during drilling.

Both share BaseControllerEnv for simulator setup / termination checks.
train_hierarchical.py trains either one standalone; evaluate_hierarchical.py
runs BOTH together, on one shared robot, with a rule-based switcher.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import coppelia as cp
from controller import MolebotController


class BaseControllerEnv(gym.Env):
    """Common simulator setup + goal-relative / divergence helpers.

    Pass an existing `ctrl` to REUSE a running controller (same robot, same
    simulator) instead of launching a new CoppeliaSim instance -- this is how
    evaluate_hierarchical.py runs locomotion and drilling on one shared robot.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg, ctrl: MolebotController | None = None):
        super().__init__()
        self.cfg = cfg
        self.ctrl = ctrl if ctrl is not None else MolebotController(cfg)
        self._owns_ctrl = ctrl is None
        self.step_i = 0
        self.start_pos = np.zeros(3)
        self.start_yaw = 0.0
        self.forward0 = np.array([1.0, 0.0])
        self.left0 = np.array([0.0, 1.0])
        self.goal_xy = np.zeros(2)
        self.prev_dist = 0.0

    def _new_episode_frame(self, goal_distance: float):
        pos, yaw = self.ctrl.base_pose()
        self.start_pos = pos.copy()
        self.start_yaw = yaw
        self.forward0 = np.array([np.cos(yaw), np.sin(yaw)])
        self.left0 = np.array([-self.forward0[1], self.forward0[0]])
        self.goal_xy = self.start_pos[:2] + self.forward0 * goal_distance
        self.prev_dist = float(np.linalg.norm(self.goal_xy - self.start_pos[:2]))

    def _goal_features(self):
        pos, yaw = self.ctrl.base_pose()
        delta = self.goal_xy - pos[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        gx_b = c * delta[0] + s * delta[1]
        gy_b = -s * delta[0] + c * delta[1]
        dist = float(np.linalg.norm(delta))
        head_err = float(np.arctan2(gy_b, gx_b))
        return pos, yaw, dist, gx_b, gy_b, head_err

    def _roll_pitch(self):
        pose = self.ctrl.sim.getObjectPose(self.ctrl.base)
        R = cp.quat_to_matrix(pose[3:7])
        roll, pitch, _ = cp.roll_pitch_yaw(R, self.cfg.robot.forward_axis)
        return roll, pitch

    def _divergence_or_tip(self, pos):
        ep = self.cfg.episode
        if not np.all(np.isfinite(pos)) or float(np.max(np.abs(pos))) > ep.divergence_bound:
            return True, True
        roll, pitch = self._roll_pitch()
        tipped = abs(roll) > ep.tip_angle or abs(pitch) > ep.tip_angle
        fell = (pos[2] - self.start_pos[2]) < ep.min_height
        return False, bool(tipped or fell)

    @staticmethod
    def _clean(o: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0).astype(np.float32)

    def close(self):
        if self._owns_ctrl:
            try:
                self.ctrl.close()
            except Exception:  # noqa: BLE001
                pass


class LocomotionEnv(BaseControllerEnv):
    """Macro-action nav: pick forward / backward / turn-left / turn-right + magnitude.

    action = [fwd_score, bwd_score, left_score, right_score, magnitude] in [-1,1]^5
    One RL step = one full primitive call (drive() or turn()), which itself loops
    many simulator steps internally.
    """
    PRIMS = ("forward", "backward", "turn_left", "turn_right")

    def __init__(self, cfg, ctrl=None, standalone: bool = True):
        super().__init__(cfg, ctrl)
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        # gx_b, gy_b, dist, sin(head_err), cos(head_err), roll, pitch, stalled_flag
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self._turn_calibrated = False
        self.last_stalled = 0.0
        # standalone=True (training): no pile, pure point-to-point nav.
        # standalone=False (combined evaluator): keep the real configured pile so a
        # forward attempt can actually stall against it and trigger a drilling switch.
        self.standalone = standalone

    def reset(self, *, seed=None, options=None):
        try:
            return self._reset_once(seed=seed)
        except Exception as exc:  # noqa: BLE001
            if not self.ctrl.can_recover(exc):
                raise
            self.ctrl.restart_simulator(exc)
            self._turn_calibrated = False
            return self._reset_once(seed=seed)

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        h = self.cfg.hier
        if self.standalone:
            self.cfg.grain.nx = self.cfg.grain.ny = self.cfg.grain.nz = 0
        self.ctrl.reset()
        if not self._turn_calibrated:
            self.ctrl.turn_sign, _ = self.ctrl.calibrate_turn(15)
            self.ctrl.reset()
            self._turn_calibrated = True
        self._new_episode_frame(h.loco_goal_distance)
        self.step_i = 0
        self.last_stalled = 0.0
        return self._obs(), {}

    def _obs(self):
        _, _, dist, gx_b, gy_b, head_err = self._goal_features()
        roll, pitch = self._roll_pitch()
        h = self.cfg.hier
        o = np.array([
            gx_b / h.loco_goal_distance, gy_b / h.loco_goal_distance,
            dist / h.loco_goal_distance,
            np.sin(head_err), np.cos(head_err),
            roll, pitch, self.last_stalled,
        ])
        return self._clean(o)

    def step(self, action):
        try:
            return self._step_once(action)
        except Exception as exc:  # noqa: BLE001
            if not self.ctrl.can_recover(exc):
                raise
            self.ctrl.restart_simulator(exc)
            self._turn_calibrated = False
            info = {"is_success": False, "sim_timeout": True}
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -self.cfg.hier.loco_tip_penalty, True, False, info

    def _step_once(self, action):
        h = self.cfg.hier
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        idx = int(np.argmax(action[:4]))
        mag = float((action[4] + 1.0) / 2.0)
        requested = None

        ms = h.loco_primitive_max_steps
        if idx in (0, 1):
            dist = h.loco_fwd_dist_range[0] + mag * (h.loco_fwd_dist_range[1] - h.loco_fwd_dist_range[0])
            res = self.ctrl.drive(dist if idx == 0 else -dist, speed=h.loco_drive_speed, max_steps=ms)
            requested = dist
        else:
            deg = h.loco_turn_deg_range[0] + mag * (h.loco_turn_deg_range[1] - h.loco_turn_deg_range[0])
            sign = getattr(self.ctrl, "turn_sign", 1.0)
            res = (self.ctrl.turn_left(deg, sign=sign, max_steps=ms) if idx == 2
                  else self.ctrl.turn_right(deg, sign=sign, max_steps=ms))

        pos, _ = self.ctrl.base_pose()
        diverged, tipped = self._divergence_or_tip(pos)
        diverged = diverged or bool(res.get("diverged", False))

        _, _, dist_now, *_ = self._goal_features()
        progress = self.prev_dist - dist_now
        self.prev_dist = dist_now
        self.last_stalled = 1.0 if (requested is not None
                                    and abs(res.get("moved", 0.0)) < h.loco_stall_ratio * requested) else 0.0

        reward = h.loco_progress_reward * progress - h.loco_energy_penalty - h.loco_time_penalty
        success = dist_now < self.cfg.episode.goal_radius
        info = {"is_success": bool(success), "dist_to_goal": dist_now, "primitive": self.PRIMS[idx],
                "stalled": bool(self.last_stalled)}
        if success:
            reward += h.loco_success_bonus
        if diverged or tipped:
            reward -= h.loco_tip_penalty
        self.step_i += 1
        terminated = bool(success or diverged or tipped)
        truncated = self.step_i >= h.loco_max_macro_steps
        return self._obs(), float(reward), terminated, truncated, info


class DrillingEnv(BaseControllerEnv):
    """Macro-action drilling: pick stab / clear / retract / done (+ stab force, speed).

    action = [stab_score, clear_score, retract_score, done_score, force_param, speed_param]
    Stab is applied symmetrically to both cones (same force/speed); clear/retract
    likewise, each staying on its own side (see controller.clear()).
    """
    PRIMS = ("stab", "clear", "retract", "done")

    def __init__(self, cfg, ctrl=None):
        super().__init__(cfg, ctrl)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        # disturbed, mean_disp, pris_R, pris_L, shoulder_R, shoulder_L, contact_R, contact_L, steps_frac
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.last_contact = {"R": 0.0, "L": 0.0}
        self.prev_disturbed = 0.0

    def reset(self, *, seed=None, options=None):
        try:
            return self._reset_once(seed=seed)
        except Exception as exc:  # noqa: BLE001
            if not self.ctrl.can_recover(exc):
                raise
            self.ctrl.restart_simulator(exc)
            return self._reset_once(seed=seed)

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        h = self.cfg.hier
        self.ctrl.reset()
        self.ctrl.drive(h.drill_approach_distance, speed=self.cfg.hier.loco_drive_speed)
        return self.begin_episode()

    def begin_episode(self):
        """Reset only the drilling bookkeeping, WITHOUT touching the simulator.
        Used by evaluate_hierarchical.py to (re)start a drilling phase on a robot
        that locomotion has already driven up to the pile."""
        self.step_i = 0
        self.last_contact = {"R": 0.0, "L": 0.0}
        self.prev_disturbed = self.ctrl.grain_disturbed()
        return self._obs(), {}

    def _obs(self):
        a = self.ctrl.arms
        h = self.cfg.hier
        o = np.array([
            self.ctrl.grain_disturbed(),
            cp.get_float_signal(self.ctrl.sim, "grain_mean_disp") or 0.0,
            self.ctrl.jpos(a["R"]["prismatic"]) / 0.5, self.ctrl.jpos(a["L"]["prismatic"]) / 0.5,
            self.ctrl.jpos(a["R"]["shoulder"]) / np.pi, self.ctrl.jpos(a["L"]["shoulder"]) / np.pi,
            np.tanh(self.last_contact["R"] / 30.0), np.tanh(self.last_contact["L"] / 30.0),
            self.step_i / max(h.drill_max_macro_steps, 1),
        ])
        return self._clean(o)

    def step(self, action):
        try:
            return self._step_once(action)
        except Exception as exc:  # noqa: BLE001
            if not self.ctrl.can_recover(exc):
                raise
            self.ctrl.restart_simulator(exc)
            info = {"is_success": False, "sim_timeout": True}
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -self.cfg.hier.drill_tip_penalty, True, False, info

    def _step_once(self, action):
        h = self.cfg.hier
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        idx = int(np.argmax(action[:4]))
        force = h.drill_force_range[0] + (action[4] + 1) / 2 * (h.drill_force_range[1] - h.drill_force_range[0])
        speed = h.drill_speed_range[0] + (action[5] + 1) / 2 * (h.drill_speed_range[1] - h.drill_speed_range[0])

        ms = h.drill_primitive_max_steps
        done_chosen = False
        if idx == 0:  # STAB (刺す)
            for side in ("R", "L"):
                res = self.ctrl.stab(side, force=float(force), speed=float(speed), max_steps=ms)
                self.last_contact[side] = abs(res.get("contact_force", 0.0))
        elif idx == 1:  # CLEAR (どける)
            for side in ("R", "L"):
                self.ctrl.clear(side, angle_deg=h.drill_clear_angle_deg, max_steps=ms)
        elif idx == 2:  # RETRACT
            for side in ("R", "L"):
                self.ctrl.retract(side, max_steps=ms)
        else:  # DONE
            done_chosen = True

        pos, _ = self.ctrl.base_pose()
        diverged, tipped = self._divergence_or_tip(pos)
        disturbed = self.ctrl.grain_disturbed()
        dig = max(0.0, disturbed - self.prev_disturbed)
        self.prev_disturbed = disturbed

        reward = h.drill_dig_reward * dig - h.drill_energy_penalty - h.drill_time_penalty
        success = disturbed >= h.drill_success_disturbed
        info = {"is_success": bool(success), "grain_disturbed": disturbed, "primitive": self.PRIMS[idx]}
        if done_chosen:
            reward += h.drill_success_bonus if success else -h.drill_premature_done_penalty
        if diverged or tipped:
            reward -= h.drill_tip_penalty
        self.step_i += 1
        terminated = bool(done_chosen or diverged or tipped)
        truncated = self.step_i >= h.drill_max_macro_steps
        return self._obs(), float(reward), terminated, truncated, info
