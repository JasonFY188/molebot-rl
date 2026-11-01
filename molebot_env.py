"""Gymnasium environment that wraps the CoppeliaSim molebot.

One env == one CoppeliaSim instance (launched headless, or attached to a running
GUI).  The action space is target velocities for every actuated joint the robot
has (4 wheels + the two cone-arm chains -> ~10 DoF, auto-discovered).  Two tasks
share the *same* observation/action layout so a straight-line policy can be
initialised from a cross-the-pile checkpoint:

    task="cross"    -> dig/push through the grain pile to reach the far side
    task="straight" -> drive as straight as possible along the initial heading
"""
from __future__ import annotations

import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import coppelia as cp


class MolebotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, task: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.task = task or cfg.task
        assert self.task in ("cross", "straight"), self.task

        # ---- bring up the simulator ----
        self._proc = None
        self.client = None
        self.sim = None
        self.port = cfg.sim.port
        self._connect_simulator()

        # per-episode state
        self.step_i = 0
        self.start_pos = np.zeros(3)
        self.start_yaw = 0.0
        self.forward0 = np.array([1.0, 0.0])
        self.left0 = np.array([0.0, 1.0])
        self.goal_xy = np.zeros(2)
        self.prev_dist = 0.0
        self.prev_disturbed = 0.0
        self.prev_action = None
        self._m = {}  # cached metrics for the reward fn

    # ------------------------------------------------------------------ setup
    def _connect_simulator(self):
        """Launch/connect and resolve scene handles for this simulator instance."""
        port = self.cfg.sim.port
        if self.cfg.sim.launch:
            # Grab a free port so we never hijack an already-running GUI on 23000.
            if self.cfg.sim.auto_port:
                port = cp.find_free_port()
            self._proc = cp.launch_coppelia(self.cfg.sim.coppelia_exe, self.cfg.sim.scene,
                                            port, self.cfg.sim.headless)
        self.port = port
        self.client, self.sim = cp.connect(
            port, self.cfg.sim.connect_timeout, self.cfg.sim.rpc_timeout_s)
        if (not self.cfg.sim.launch) and self.cfg.sim.load_scene_on_attach and self.cfg.sim.scene:
            self.sim.loadScene(self.cfg.sim.scene)
        self._resolve_constants()
        self._discover_joints()
        self._build_spaces()

    def _can_recover(self, exc: BaseException) -> bool:
        return bool(self.cfg.sim.launch and self.cfg.sim.restart_on_rpc_timeout
                    and cp.is_rpc_timeout(exc))

    def _restart_simulator(self, exc: BaseException):
        """Replace one launched worker after its ZMQ request timed out."""
        print(f"[env watchdog] RPC timed out on port {self.port}; restarting worker ({exc}).",
              flush=True)
        cp.close_client(self.client)
        self.client = None
        self.sim = None
        if self._proc is not None:
            cp.terminate_process(self._proc)
            self._proc = None
        self._connect_simulator()
        self.prev_action = None
        self._m = {}

    def _resolve_constants(self):
        s = self.sim
        self.C = dict(
            joint_type=cp.const(s, "object_joint_type"),
            shape_type=cp.const(s, "object_shape_type", default=0),
            handle_all=cp.const(s, "handle_all"),
            prismatic=cp.const(s, "joint_prismatic_subtype", "joint_prismatic"),
            sim_stopped=cp.const(s, "simulation_stopped"),
            dynctrl_param=cp.const(s, "jointintparam_dynctrlmode", default=-1),
            dynctrl_velocity=cp.const(s, "jointdynctrl_velocity", default=-1),
            respondable=cp.const(s, "shapeintparam_respondable", default=-1),
        )

    def _discover_joints(self):
        s, C = self.sim, self.C
        try:
            base = s.getObject(self.cfg.robot.base_path)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Could not find robot base '{self.cfg.robot.base_path}'. "
                f"Open the scene and check the object alias. ({e})")
        self.base = base

        tree = s.getObjectsInTree(base, C["handle_all"], 0)
        wheels, arms = [], []
        self.cone_shapes = []
        for h in tree:
            t = s.getObjectType(h)
            if t == C["joint_type"]:
                alias = cp.object_alias(s, h)
                if self.cfg.robot.wheel_name_hint in alias.lower():
                    wheels.append((alias, h))
                else:
                    arms.append((alias, h))
            elif (t == C["shape_type"] and self.cfg.robot.disable_cone_collision
                  and self.cfg.robot.cone_name_hint in cp.object_alias(s, h).lower()):
                # only disable collision for the heavy original cones; low-poly rebuilt
                # cones (< cone_max_tris) keep colliding so they still drill.
                try:
                    _, idx, _ = s.getShapeMesh(h)
                    ntri = len(idx) // 3
                except Exception:  # noqa: BLE001
                    ntri = 10 ** 9
                if ntri > self.cfg.robot.cone_max_tris:
                    self.cone_shapes.append(h)
        wheels.sort()
        arms.sort()
        if not (wheels or arms):
            raise RuntimeError("No joints discovered under the robot base.")

        self.joint_alias = [a for a, _ in wheels] + [a for a, _ in arms]
        self.joints = [h for _, h in wheels] + [h for _, h in arms]
        self.is_wheel = [True] * len(wheels) + [False] * len(arms)
        self.arm_idx = [i for i, w in enumerate(self.is_wheel) if not w]
        self.n_act = len(self.joints)

        # per-joint max target velocity + limits (for normalising arm positions)
        self.max_vel = np.zeros(self.n_act)
        self.jpos_off = np.zeros(self.n_act)
        self.jpos_scale = np.ones(self.n_act)
        self.cyclic = [False] * self.n_act
        r = self.cfg.robot
        for i, h in enumerate(self.joints):
            jtype = s.getJointType(h)
            if self.is_wheel[i]:
                self.max_vel[i] = r.wheel_max_vel
            elif jtype == C["prismatic"]:
                self.max_vel[i] = r.arm_pris_max_vel
            else:
                self.max_vel[i] = r.arm_rev_max_vel
            try:
                cyclic, interval = s.getJointInterval(h)
                self.cyclic[i] = bool(cyclic)
                if not cyclic and interval and interval[1] > 1e-6:
                    self.jpos_off[i] = interval[0]
                    self.jpos_scale[i] = interval[1]
            except Exception:  # noqa: BLE001
                pass

    def _build_spaces(self):
        # obs = 14 base features + normalised velocity of every joint
        #        + normalised position of every arm joint
        self._n_base = 14
        self.obs_dim = self._n_base + self.n_act + len(self.arm_idx)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

    def _setup_motors(self):
        s, C = self.sim, self.C
        for i, h in enumerate(self.joints):
            if C["dynctrl_param"] != -1 and C["dynctrl_velocity"] != -1:
                try:
                    s.setObjectInt32Param(h, C["dynctrl_param"], C["dynctrl_velocity"])
                except Exception:  # noqa: BLE001
                    pass
            force = self.cfg.robot.wheel_max_force if self.is_wheel[i] else self.cfg.robot.arm_max_force
            try:
                s.setJointTargetForce(h, force, False)
            except Exception:  # noqa: BLE001
                pass
            try:
                s.setJointTargetVelocity(h, 0.0)
            except Exception:  # noqa: BLE001
                pass

    def _push_grain_signals(self):
        s, g = self.sim, self.cfg.grain
        cp.set_int_signal(s, "grain_nx", g.nx)
        cp.set_int_signal(s, "grain_ny", g.ny)
        cp.set_int_signal(s, "grain_nz", g.nz)
        cp.set_float_signal(s, "grain_d", g.diameter)
        cp.set_float_signal(s, "grain_spacing", g.spacing)
        cp.set_float_signal(s, "grain_cx", g.center_x)
        cp.set_float_signal(s, "grain_cy", g.center_y)
        cp.set_float_signal(s, "grain_floorz", g.floor_z)
        cp.set_float_signal(s, "grain_disturb_thresh", g.disturb_thresh)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        try:
            return self._reset_once(seed=seed)
        except Exception as exc:
            if not self._can_recover(exc):
                raise
            self._restart_simulator(exc)
            return self._reset_once(seed=seed)

    def _reset_once(self, *, seed=None):
        super().reset(seed=seed)
        s, C = self.sim, self.C

        if s.getSimulationState() != C["sim_stopped"]:
            s.stopSimulation()
            while s.getSimulationState() != C["sim_stopped"]:
                time.sleep(0.02)

        self._push_grain_signals()      # read by the grain script's sysCall_init
        # Disable collision on the huge cone meshes (~100x faster steps). Must be set
        # while stopped, every reset (stop restores the scene's initial respondable flag).
        if self.C["respondable"] != -1:
            for h in self.cone_shapes:
                try:
                    s.setObjectInt32Param(h, self.C["respondable"], 0)
                except Exception:  # noqa: BLE001
                    pass
        # RL wants as-fast-as-possible stepping, never wall-clock-throttled real time.
        try:
            s.setBoolParam(s.boolparam_realtime_simulation, False)
        except Exception:  # noqa: BLE001
            pass
        s.setStepping(True)
        s.startSimulation()
        self._setup_motors()
        for _ in range(self.cfg.sim.step_settle):
            s.step()

        st = self._read_state()
        self.start_pos = st["pos"].copy()
        self.start_yaw = st["yaw"]
        self.forward0 = np.array([np.cos(st["yaw"]), np.sin(st["yaw"])])
        self.left0 = np.array([-self.forward0[1], self.forward0[0]])
        self.goal_xy = self.start_pos[:2] + self.forward0 * self.cfg.episode.goal_distance
        self.prev_dist = float(np.linalg.norm(self.goal_xy - self.start_pos[:2]))
        self.prev_disturbed = st["grain_disturbed"]
        self.prev_action = np.zeros(self.n_act, dtype=np.float32)
        self.step_i = 0
        self._m = {}                     # per-episode reward bookkeeping

        return self._make_obs(st), {}

    def step(self, action):
        try:
            return self._step_once(action)
        except Exception as exc:
            if not self._can_recover(exc):
                raise
            self._restart_simulator(exc)
            # SubprocVecEnv sees this as a failed episode, then calls reset() on the
            # new simulator before asking this worker for another action.
            info = {"is_success": False, "sim_timeout": True}
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -self.cfg.reward.tip_penalty, True, False, info

    def _step_once(self, action):
        s = self.sim
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action)
        max_delta = float(getattr(self.cfg.robot, "action_smoothing", 1.0))
        if max_delta < 1.0:
            action = self.prev_action + np.clip(action - self.prev_action, -max_delta, max_delta)
            action = np.clip(action, -1.0, 1.0)
        self.prev_action = action.copy()
        for i, h in enumerate(self.joints):
            try:
                s.setJointTargetVelocity(h, float(action[i] * self.max_vel[i]))
            except Exception:  # noqa: BLE001
                pass
        for _ in range(self.cfg.episode.action_repeat):
            s.step()

        st = self._read_state()
        obs = self._make_obs(st)
        reward, terminated, info = self._reward(st, action)
        self.step_i += 1
        truncated = self.step_i >= self.cfg.episode.max_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            if self.sim.getSimulationState() != self.C["sim_stopped"]:
                self.sim.stopSimulation()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            cp.terminate_process(self._proc)
            self._proc = None

    # ------------------------------------------------------------------ state
    def _read_state(self) -> dict:
        s = self.sim
        pose = s.getObjectPose(self.base)          # [x y z qx qy qz qw]
        pos = np.asarray(pose[:3], float)
        R = cp.quat_to_matrix(pose[3:7])
        roll, pitch, yaw = cp.roll_pitch_yaw(R, self.cfg.robot.forward_axis)
        lin, ang = cp.object_velocity(s, self.base)
        vb = R.T @ lin                             # body-frame linear velocity

        jpos = np.array([s.getJointPosition(h) for h in self.joints], float)
        jvel = np.array([s.getJointVelocity(h) for h in self.joints], float)

        grain = {
            "com": np.array([cp.get_float_signal(s, "grain_com_x") or 0.0,
                             cp.get_float_signal(s, "grain_com_y") or 0.0,
                             cp.get_float_signal(s, "grain_com_z") or 0.0]),
            "disturbed": cp.get_float_signal(s, "grain_disturbed") or 0.0,
            "mean_disp": cp.get_float_signal(s, "grain_mean_disp") or 0.0,
        }
        return dict(pos=pos, R=R, roll=roll, pitch=pitch, yaw=yaw, lin=lin,
                    vb=vb, wz=float(ang[2]), jpos=jpos, jvel=jvel,
                    grain_com=grain["com"], grain_disturbed=grain["disturbed"],
                    grain_mean_disp=grain["mean_disp"])

    def _make_obs(self, st) -> np.ndarray:
        ep = self.cfg.episode
        delta = self.goal_xy - st["pos"][:2]
        yaw = st["yaw"]
        c, s_ = np.cos(yaw), np.sin(yaw)
        gx_b = c * delta[0] + s_ * delta[1]        # goal in body frame
        gy_b = -s_ * delta[0] + c * delta[1]
        dist = float(np.linalg.norm(delta))
        head_err = np.arctan2(gy_b, gx_b)          # bearing to goal in body frame

        grain_rel = st["grain_com"][:2] - st["pos"][:2]
        gr_x = c * grain_rel[0] + s_ * grain_rel[1]
        gr_y = -s_ * grain_rel[0] + c * grain_rel[1]

        base = np.array([
            gx_b / ep.goal_distance, gy_b / ep.goal_distance,
            dist / ep.goal_distance,
            np.sin(head_err), np.cos(head_err),
            st["vb"][0], st["vb"][1],
            st["wz"] / (2 * np.pi),
            st["roll"], st["pitch"],
            st["pos"][2] - self.start_pos[2],
            gr_x, gr_y,
            st["grain_disturbed"],
        ], dtype=np.float32)

        jvel_n = (st["jvel"] / np.maximum(self.max_vel, 1e-6)).astype(np.float32)

        arm_pos = st["jpos"][self.arm_idx]
        arm_off = self.jpos_off[self.arm_idx]
        arm_scale = self.jpos_scale[self.arm_idx]
        arm_cyc = np.array([self.cyclic[i] for i in self.arm_idx])
        pos_n = np.where(
            arm_cyc,
            (np.mod(arm_pos + np.pi, 2 * np.pi) - np.pi) / np.pi,   # wrap cyclic to [-1,1]
            2.0 * (arm_pos - arm_off) / np.maximum(arm_scale, 1e-6) - 1.0,
        ).astype(np.float32)

        obs = np.concatenate([base, jvel_n, pos_n]).astype(np.float32)
        # never hand NaN/Inf or absurd magnitudes to the network
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)

    # ------------------------------------------------------------------ reward
    def _reward(self, st, action):
        r, ep = self.cfg.reward, self.cfg.episode
        info = {"is_success": False}

        # physics blow-up: robot flung to infinity / NaN -> end episode, bounded penalty
        if (not np.all(np.isfinite(st["pos"]))
                or float(np.max(np.abs(st["pos"]))) > ep.divergence_bound
                or not np.all(np.isfinite(st["vb"]))):
            info["diverged"] = True
            return -r.tip_penalty, True, info

        tipped = abs(st["roll"]) > ep.tip_angle or abs(st["pitch"]) > ep.tip_angle
        fell = (st["pos"][2] - self.start_pos[2]) < ep.min_height

        energy = r.energy * float(np.mean(np.square(action)))

        if self.task == "cross":
            delta = st["pos"][:2] - self.start_pos[:2]
            along = float(np.dot(delta, self.forward0))
            lateral = float(np.dot(delta, self.left0))
            dist = float(np.linalg.norm(self.goal_xy - st["pos"][:2]))
            progress = self.prev_dist - dist
            self.prev_dist = dist
            dig = max(0.0, st["grain_disturbed"] - self.prev_disturbed)
            self.prev_disturbed = max(self.prev_disturbed, st["grain_disturbed"])
            forward_speed = float(np.dot(st["lin"][:2], self.forward0))
            reward = (r.cross_progress * progress
                      + r.cross_grain * dig
                      + r.cross_forward * max(0.0, forward_speed)
                      - energy - r.time_penalty)
            success = dist < ep.goal_radius
            if success:
                reward += r.cross_success
            info["is_success"] = bool(success)
            info["dist_to_goal"] = dist
            info["along"] = along
            info["lateral"] = lateral
            info["grain_disturbed"] = st["grain_disturbed"]
            reversed_too_far = along < -self.cfg.robot.cross_backward_limit
            sideways_too_far = abs(lateral) > self.cfg.robot.cross_lateral_limit
            if reversed_too_far or sideways_too_far:
                reward -= r.tip_penalty
                info["out_of_bounds"] = True
            terminated = bool(success or tipped or fell or reversed_too_far or sideways_too_far)
        else:  # straight
            delta = st["pos"][:2] - self.start_pos[:2]
            along = float(np.dot(delta, self.forward0))
            lateral = float(np.dot(delta, self.left0))
            yaw_err = float((st["yaw"] - self.start_yaw + np.pi) % (2 * np.pi) - np.pi)
            progress = along - self._m.get("along", 0.0)
            self._m["along"] = along
            reward = (r.straight_forward * progress
                      - r.straight_lateral * abs(lateral)
                      - r.straight_yaw * abs(yaw_err)
                      - r.straight_wz * abs(st["wz"])
                      - energy)
            info["along"] = along
            info["lateral"] = lateral
            info["yaw_err"] = yaw_err
            terminated = bool(tipped or fell or abs(lateral) > ep.max_lateral)

        if tipped or fell:
            reward -= r.tip_penalty
        return float(reward), terminated, info
