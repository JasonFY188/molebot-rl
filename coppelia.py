"""Low-level CoppeliaSim helpers: launch, connect, constant resolution, small math.

Keeps every fragile "does this API constant exist in this version" decision in one
place so the env code stays readable.
"""
from __future__ import annotations

import os
import time
import socket
import subprocess

import numpy as np
import zmq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# --------------------------------------------------------------------------- #
# Launch / connect
# --------------------------------------------------------------------------- #
def find_free_port(start: int = 23050, count: int = 60) -> int:
    """Return a TCP port nothing is listening on (so our launched sim owns it,
    instead of silently attaching to an already-running CoppeliaSim on 23000)."""
    for p in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port found for CoppeliaSim")


def terminate_process(proc):
    """Kill CoppeliaSim and its whole child-process tree (terminate() alone leaks it)."""
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

def launch_coppelia(exe: str, scene: str, port: int, headless: bool) -> subprocess.Popen:
    """Spawn a CoppeliaSim instance (its ZMQ Remote API add-on auto-starts)."""
    if not os.path.isfile(exe):
        raise FileNotFoundError(f"CoppeliaSim executable not found: {exe}")
    args = [exe]
    if headless:
        args.append("-h")
    # Tell the remote-API add-on which port to listen on (default is 23000 anyway).
    args.append(f"-GzmqRemoteApi.rpcPort={port}")
    if scene:
        if not os.path.isfile(scene):
            raise FileNotFoundError(f"Scene file not found: {scene}")
        args.append(scene)
    # DETACHED so the sim keeps running independently; we terminate it in close().
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(args, creationflags=flags)


def set_rpc_timeout(client: RemoteAPIClient, timeout_s: float):
    """Bound a Remote API request so a stalled simulator cannot block RL forever."""
    ms = max(1, int(timeout_s * 1000))
    client.socket.setsockopt(zmq.RCVTIMEO, ms)
    client.socket.setsockopt(zmq.SNDTIMEO, ms)
    client.socket.setsockopt(zmq.LINGER, 0)


def close_client(client):
    """Dispose a timed-out ZMQ client; a REQ socket cannot be reused after a timeout."""
    if client is None:
        return
    try:
        client.socket.close(0)
    except Exception:  # noqa: BLE001
        pass
    try:
        client.context.term()
    except Exception:  # noqa: BLE001
        pass


def is_rpc_timeout(exc: BaseException) -> bool:
    return isinstance(exc, zmq.error.Again)


def connect(port: int, timeout: float, rpc_timeout_s: float = 20.0):
    """Return (client, sim), retrying until CoppeliaSim's ZMQ server answers."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            client = RemoteAPIClient(port=port)
            # Use a short timeout while the server is still coming up. Once connected,
            # use the configured watchdog timeout for normal simulation requests.
            set_rpc_timeout(client, min(rpc_timeout_s, 5.0))
            sim = client.require("sim")
            sim.getSimulationTime()  # ping
            set_rpc_timeout(client, rpc_timeout_s)
            return client, sim
        except Exception as e:  # noqa: BLE001 - server may not be up yet
            last = e
            close_client(locals().get("client"))
            time.sleep(0.5)
    raise RuntimeError(f"Could not connect to CoppeliaSim on port {port} within "
                       f"{timeout:.0f}s. Last error: {last}")


# --------------------------------------------------------------------------- #
# Version-robust constant / call helpers
# --------------------------------------------------------------------------- #
def const(sim, *names, default=None):
    """First existing attribute among *names (handles API renames across versions)."""
    for n in names:
        if hasattr(sim, n):
            return getattr(sim, n)
    if default is not None:
        return default
    raise AttributeError(f"None of {names} found on sim object")


def object_alias(sim, handle: int) -> str:
    """Best-effort human-readable alias for an object handle."""
    for args in ((handle,), (handle, 1), (handle, 0), (handle, -1)):
        try:
            a = sim.getObjectAlias(*args)
            if isinstance(a, str) and a:
                return a
        except Exception:  # noqa: BLE001
            continue
    return f"obj{handle}"


def object_velocity(sim, handle):
    """Return (lin[3], ang[3]) regardless of how the client packs the result."""
    v = sim.getObjectVelocity(handle)
    if len(v) == 2:               # [lin, ang]
        return np.asarray(v[0], float), np.asarray(v[1], float)
    return np.asarray(v[:3], float), np.asarray(v[3:6], float)  # flat [6]


def get_float_signal(sim, name: str):
    """Read a float signal, tolerating absence / API differences. Returns None if unset."""
    try:
        v = sim.getFloatSignal(name)
        return None if v is None else float(v)
    except Exception:  # noqa: BLE001
        return None


def set_float_signal(sim, name: str, value: float):
    try:
        sim.setFloatSignal(name, float(value))
    except Exception:  # noqa: BLE001
        pass


def set_int_signal(sim, name: str, value: int):
    try:
        sim.setInt32Signal(name, int(value))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Small rigid-body math (CoppeliaSim quaternion order is [x, y, z, w])
# --------------------------------------------------------------------------- #
def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,          xz + wy],
        [xy + wz,          1.0 - (xx + zz), yz - wx],
        [xz - wy,          yz + wx,          1.0 - (xx + yy)],
    ])


def roll_pitch_yaw(R: np.ndarray, forward_axis: str = "+x"):
    """roll, pitch (tilt of body-up from world-up) and yaw of the chosen forward axis.

    yaw is the heading of the robot's forward axis projected onto the world XY-plane.
    roll/pitch are reported as the tilt of the body Z (up) axis, which is what we use
    purely to detect tipping, so their exact convention does not matter.
    """
    fwd = axis_vector(forward_axis)
    f_world = R @ fwd
    yaw = float(np.arctan2(f_world[1], f_world[0]))
    up_world = R @ np.array([0.0, 0.0, 1.0])
    # tilt of body-up away from world-up, split into roll (about forward) & pitch
    pitch = float(np.arctan2(-up_world[0], up_world[2]))
    roll = float(np.arctan2(up_world[1], up_world[2]))
    return roll, pitch, yaw


def axis_vector(axis: str) -> np.ndarray:
    sign = -1.0 if axis.startswith("-") else 1.0
    a = axis[-1].lower()
    base = {"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}[a]
    return sign * np.asarray(base)
