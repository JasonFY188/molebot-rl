# Preparing the CoppeliaSim scene

The RL code talks to CoppeliaSim over the **ZMQ Remote API** (port 23000, enabled
by default in 4.10). It drives the robot's joints from Python; the scene only has
to provide (a) the molebot and (b) the grain-field child-script.

## 1. Robot base object
The env auto-discovers all joints in the subtree of one base object. Default is
the alias **`robot_body`** (`config.py -> RobotConfig.base_path = "/robot_body"`).
Open the scene, click the chassis in the *Scene hierarchy*, and confirm its alias
is `robot_body`. If it differs, either rename it (double-click the name) or set
`base_path` in `config.py`.

The four wheels must have `wheel` in their alias (they do:
`front_left_wheel_joint`, …) — that's how the env tells wheels from the cone-arm
joints. Everything else under `robot_body` (the two `Revolute_joint` chains, the
`Prismatic_joint`s, the cone `Revolute_joint`s) is treated as an arm joint.

## 2. Grain-field script
Replace the child script on the **`dummy`** object with
[`grain_field.lua`](grain_field.lua):

1. In the *Scene hierarchy*, double-click the script icon next to `dummy`.
2. Select-all, delete, paste the entire contents of `grain_field.lua`, save.

It is a superset of your original script — same grain physics — but it reads the
pile size from signals (so training can use fewer grains for speed) and publishes
`grain_com_*`, `grain_mean_disp`, `grain_disturbed` signals that the agent uses
for the dig-through reward. With no signals set it falls back to your 8×8×4 = 256
grains.

## 3. Joints must be dynamically motor-driven
For velocity control to move a joint it must be **dynamic** with its **motor
enabled**. The wheels already are. For the cone-arm joints, in the joint's
*Dynamics* dialog make sure "Motor enabled" is on (the env also requests velocity
control mode over the API each reset, but it can't enable dynamics on a purely
static joint).

## 4. Remove any competing controller
If the scene has its own child-script that sets wheel/arm velocities (e.g. a
`molebot_control` demo controller), disable or delete it — otherwise it fights the
RL commands. The base `molebot.ttt` is the clean choice.

## 5. Save
Save the scene, and point `config.py -> SimConfig.scene` (or `--scene`) at it.

## 6. Test
```
python smoke_test.py --attach      # with the GUI open + scene loaded
```
You want to see: all joints listed, `is_wheel` correct, grain COM non-zero, and
"all-wheels-forward" producing forward (+along) motion.
