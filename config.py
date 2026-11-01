"""Central configuration for the molebot RL project.

Everything tunable lives here as dataclasses so the env, trainer and eval
scripts all share one source of truth.  CLI flags in train.py / evaluate.py
override individual fields at runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace  # noqa: F401 (replace re-exported)

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Simulator connection / launch
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    # Both can be overridden with environment variables (see .env.example).
    coppelia_exe: str = os.environ.get(
        "COPPELIASIM_EXE", r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe")
    # The Molebot scene shipped in scene/ (renamed joints: shoulder/knee/extender/arm/wrist/half_cone).
    scene: str = os.environ.get("MOLEBOT_SCENE", os.path.join(_HERE, "scene", "molebot.ttt"))
    port: int = 23000          # used when attaching (a GUI CoppeliaSim defaults to 23000)
    auto_port: bool = True     # when launching, grab a free port instead of `port`
    headless: bool = True      # -h : no rendering (fast). Set False to watch.
    launch: bool = True        # True -> spawn CoppeliaSim; False -> attach to a running instance.
    load_scene_on_attach: bool = False  # when attaching, (re)load the scene over the API
    connect_timeout: float = 90.0
    rpc_timeout_s: float = 45.0  # abandon/restart a launched worker if one RPC call stalls
                                 # (heavy contact steps when driving into the pile can be slow)
    restart_on_rpc_timeout: bool = True
    step_settle: int = 20      # sim.step() calls right after start to let things settle
                               # (grain_field.lua re-baselines "disturbed" after 15 of these)


# --------------------------------------------------------------------------- #
# Robot / actuation
# --------------------------------------------------------------------------- #
@dataclass
class RobotConfig:
    base_path: str = "/robot_body"     # top-level robot object (alias/path in the scene)
    wheel_name_hint: str = "wheel"     # joints whose alias contains this are treated as wheels
    cone_name_hint: str = "cone"       # shapes whose alias contains this are the cone tools
    # The original half-cones are ~40k-triangle meshes; their collision makes each sim step
    # ~100x slower (929 ms vs 9 ms). This disables collision ONLY for cones heavier than
    # cone_max_tris. After you rebuild them low-poly (scene/rebuild_cones.lua, ~1k tris),
    # they fall under the threshold and keep colliding -> the cones drill AND it's fast.
    disable_cone_collision: bool = True
    cone_max_tris: int = 4000
    # Measured on molebot.ttt: driving all wheels + rolls the body along its -y axis.
    forward_axis: str = "-y"           # robot-local forward direction: one of +x -x +y -y
    wheel_max_vel: float = 6.0         # rad/s   lower than demo speed; safer for random RL exploration
    arm_rev_max_vel: float = 0.7       # rad/s   gentle cone/arm motion avoids solver blow-ups
    arm_pris_max_vel: float = 0.04     # m/s
    wheel_max_force: float = 30.0      # N.m  wheel motor torque cap (old flat env)
    wheel_locomotion_force: float = 100.0  # N.m  primitives/controller: enough torque to skid-turn
    arm_max_force: float = 8.0         # N or N.m arm motor cap for clear/retract (gentle)
    arm_stab_force: float = 30.0       # N  prismatic thrust force during STAB -> harder push = stronger
                                       # contact force / more ball displacement (SAC will set this later)
    arm_hold_force: float = 120.0      # N  velocity-brake force that holds the arms still while driving,
                                       # so the cones don't drift when the body accelerates/turns
    action_smoothing: float = 0.25     # max action change per env step; prevents instant full-power flips
    cross_backward_limit: float = 0.35 # terminate cross episode before robot reverses off the plane
    cross_lateral_limit: float = 1.25  # terminate cross episode before robot drives sideways off the plane


# --------------------------------------------------------------------------- #
# Granular field (must match the grain child-script; sent to it via signals)
# --------------------------------------------------------------------------- #
@dataclass
class GrainConfig:
    nx: int = 6
    ny: int = 6
    nz: int = 3            # 6*6*3 = 108 grains for training (original demo was 8*8*4=256)
    diameter: float = 0.10
    # spawn spacing as a multiple of diameter. >= 2.0 keeps balls outside the grain
    # repulsion cutoff at spawn so the pile settles instead of exploding (which stalls
    # the physics solver). 1.1 = original (explodes at high ball counts).
    spacing: float = 2.1
    center_x: float = 0.0
    center_y: float = 0.0
    floor_z: float = 0.0
    disturb_thresh: float = 0.03   # a grain counts as "disturbed" once it moves > this (m)


# --------------------------------------------------------------------------- #
# Episode / termination
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeConfig:
    action_repeat: int = 3     # sim.step() calls per agent action (smaller = safer contact dynamics)
    max_steps: int = 300       # agent steps before truncation
    tip_angle: float = 1.0     # rad; terminate if |roll| or |pitch| exceeds this (~57 deg)
    min_height: float = -0.15  # terminate if body drops this far below its start height
    divergence_bound: float = 20.0  # |pos| beyond this (or non-finite) = solver blow-up -> end episode
    goal_distance: float = 2.5 # m ahead (along forward axis); robot starts ~1.87 m from the
                               # pile centre, so 2.5 m puts the goal well past the far side.
    goal_radius: float = 0.25  # m; success when body is within this of the goal
    max_lateral: float = 1.0   # straight task: terminate if |lateral deviation| exceeds


# --------------------------------------------------------------------------- #
# Reward weights
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    # ---- cross / dig-through ----
    cross_progress: float = 10.0   # per metre of progress toward the far-side goal
    cross_grain: float = 3.0       # per unit increase in disturbed-grain fraction (digging)
    cross_forward: float = 0.5     # per (m/s) of forward body velocity
    cross_success: float = 50.0    # one-off bonus for reaching the goal
    # ---- straight line ----
    straight_forward: float = 12.0 # per metre travelled along the initial heading
    straight_lateral: float = 5.0  # per metre of |lateral deviation|
    straight_yaw: float = 2.0      # per rad of |heading error|
    straight_wz: float = 0.20      # per (rad/s) of |yaw rate|
    # ---- shared ----
    energy: float = 0.01       # per unit mean(action^2)
    time_penalty: float = 0.01 # per agent step (encourages finishing)
    tip_penalty: float = 10.0  # one-off penalty on a tip/fall termination


# --------------------------------------------------------------------------- #
# Hierarchical (two-policy) training: macro-actions over the scripted primitives
# in controller.py. LocomotionEnv picks among forward/backward/turn-left/turn-
# right; DrillingEnv picks among stab/clear/retract/done. See hierarchical_env.py.
# --------------------------------------------------------------------------- #
@dataclass
class HierConfig:
    # ---- locomotion (trained standalone, no pile: pure point-to-point nav) ----
    loco_max_macro_steps: int = 25
    loco_goal_distance: float = 2.5        # m ahead along the start heading
    loco_fwd_dist_range: tuple = (0.2, 1.0)  # m per forward/backward call; SAC picks within this
    loco_turn_deg_range: tuple = (15.0, 90.0)  # deg per turn call; SAC picks within this
    loco_drive_speed: float = 4.0          # wheel speed used for macro drive/turn calls
    loco_progress_reward: float = 10.0
    loco_energy_penalty: float = 0.02
    loco_time_penalty: float = 0.05
    loco_tip_penalty: float = 10.0
    loco_success_bonus: float = 50.0
    loco_stall_ratio: float = 0.3          # moved/requested below this => "blocked" (switch trigger)
    # Bounds the WORST-CASE wall time of one macro-action. controller.py's own drive()/
    # turn() defaults (max_steps=400) are tuned for the scripted demo's correctness, not
    # RL training speed -- with a noisy early policy some primitives hunt for a long time
    # before converging, and in a synchronous VecEnv EVERY worker waits for the slowest
    # one each round, so a single straggler stalls the whole batch. ~100 steps is enough
    # for a normal drive/turn to converge (measured ~50-60 steps typical).
    loco_primitive_max_steps: int = 100

    # ---- drilling (trained standalone, robot pre-positioned at the pile face) ----
    drill_max_macro_steps: int = 16
    # Measured on the 6,6,3 pile: 0.9m leaves disturbed~=0.2 (real headroom before the
    # 0.6 success threshold). Driving further plows straight through and leaves nothing
    # for the drilling policy to actually learn from.
    drill_approach_distance: float = 0.9   # m driven up to the pile before a drilling episode starts
    drill_success_disturbed: float = 0.6   # fraction of grains disturbed => path considered cleared
    drill_force_range: tuple = (10.0, 60.0)   # N; SAC-controlled STAB thrust force ("how hard")
    drill_speed_range: tuple = (0.02, 0.15)   # m/s; SAC-controlled STAB thrust speed
    drill_clear_angle_deg: float = 40.0    # CLEAR sweep angle (fixed, PID-executed, not learned yet)
    drill_dig_reward: float = 20.0         # per unit increase in grain_disturbed
    drill_energy_penalty: float = 0.02
    drill_time_penalty: float = 0.05
    drill_tip_penalty: float = 10.0
    drill_success_bonus: float = 30.0
    drill_premature_done_penalty: float = 10.0  # picking DONE while the pile is still mostly intact
    # Same straggler-bounding purpose as loco_primitive_max_steps. 80 tightens clear's
    # (120) and retract's (200) own defaults; close to stab's own default (60). A
    # primitive that doesn't finish in one macro-step can just be issued again next step.
    drill_primitive_max_steps: int = 80


@dataclass
class Config:
    task: str = "cross"        # "cross" | "straight"
    seed: int = 0
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    grain: GrainConfig = field(default_factory=GrainConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    hier: HierConfig = field(default_factory=HierConfig)


def default_config() -> Config:
    return Config()
