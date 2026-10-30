from __future__ import annotations

import math
from types import MethodType

import mujoco
import numpy as np

from training.e2e_phase10_lidar_v7_curriculum import (
    train_stage7_v12_1_terminal_geometry_overnight as v12,
)

_PATCHED = False
_ORIGINAL_INIT = None
_ORIGINAL_WRITE = None

STEER_KP_SCALE = 2.0
FRONT_FREE = True
USE_ACKERMANN = True
USE_REAR_DIFFERENTIAL = True

STEER_ACTUATORS = (
    "left_front_steer_act",
    "right_front_steer_act",
)
DRIVE_ACTUATORS = (
    "left_rear_wheel_act",
    "right_rear_wheel_act",
)
FRONT_WHEEL_JOINTS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
)
WHEEL_BODIES = {
    "fl": "left_front_wheel",
    "fr": "right_front_wheel",
    "rl": "left_rear_wheel",
    "rr": "right_rear_wheel",
}


def _oid(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise RuntimeError(f"FULL_PLANT missing object: {name}")
    return int(idx)


def _geometry(env):
    model, data = env.model, env.data
    ids = {
        k: _oid(model, mujoco.mjtObj.mjOBJ_BODY, v)
        for k, v in WHEEL_BODIES.items()
    }
    p = {k: data.xpos[v][:2].copy() for k, v in ids.items()}
    front_center = 0.5 * (p["fl"] + p["fr"])
    rear_center = 0.5 * (p["rl"] + p["rr"])
    left_center = 0.5 * (p["fl"] + p["rl"])
    right_center = 0.5 * (p["fr"] + p["rr"])
    wheelbase = float(np.linalg.norm(front_center - rear_center))
    track = float(np.linalg.norm(left_center - right_center))
    return wheelbase, track


def _clip_ctrl(model, aid, value):
    lo, hi = map(float, model.actuator_ctrlrange[aid])
    return float(np.clip(value, lo, hi))


def _apply_static_full_plant(env):
    model = env.model

    # 1) Keep the already-verified Stage8 CONDIM3 contact model unchanged.
    active_relevant_condims = []
    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, bid
        ) or ""
        text = bname.lower()
        if (
            "wheel" in text
            or gtype == int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            if (
                int(model.geom_contype[gid]) != 0
                and int(model.geom_conaffinity[gid]) != 0
            ):
                active_relevant_condims.append(
                    int(model.geom_condim[gid])
                )

    if not active_relevant_condims:
        raise RuntimeError("FULL_PLANT could not audit wheel/plane condim")

    if any(x != 3 for x in active_relevant_condims):
        raise RuntimeError(
            "FULL_PLANT refuses to run: Stage8 active wheel/plane "
            f"condims are {sorted(set(active_relevant_condims))}, expected [3]"
        )

    # 2) Front free rolling, exactly as the successful plant A/B.
    if FRONT_FREE:
        for name in FRONT_WHEEL_JOINTS:
            jid = _oid(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            did = int(model.jnt_dofadr[jid])
            model.dof_damping[did] = 0.0
            model.dof_frictionloss[did] = 0.0
            model.dof_armature[did] = 1.0e-5

    # 3) Steering position servo gain x2.
    for name in STEER_ACTUATORS:
        aid = _oid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        base_kp = 1200.0
        kp = base_kp * STEER_KP_SCALE
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp

    mujoco.mj_forward(model, env.data)

    wheelbase, track = _geometry(env)
    env._fullplant_wheelbase = wheelbase
    env._fullplant_track = track
    env._fullplant_installed = True
    env._fullplant_name = "CONDIM3_ACKERMANN_REAR_DIFF_FRONT_FREE_KP_X2"


def _ackermann(center_steer, wheelbase, track):
    steer = float(center_steer)
    if abs(steer) < 1.0e-8:
        return 0.0, 0.0

    sign = 1.0 if steer > 0.0 else -1.0
    magnitude = abs(steer)
    radius = wheelbase / max(math.tan(magnitude), 1.0e-8)

    inner = math.atan2(
        wheelbase,
        max(radius - track / 2.0, 1.0e-6),
    )
    outer = math.atan2(
        wheelbase,
        radius + track / 2.0,
    )

    if sign > 0.0:  # LEFT
        return inner, outer
    return -outer, -inner


def _rear_diff(center_steer, wheel, wheelbase, track):
    steer = float(center_steer)
    wheel = float(wheel)

    if abs(steer) < 1.0e-8:
        return wheel, wheel

    sign = 1.0 if steer > 0.0 else -1.0
    radius = wheelbase / max(math.tan(abs(steer)), 1.0e-8)

    inner = wheel * max(radius - track / 2.0, 1.0e-6) / radius
    outer = wheel * (radius + track / 2.0) / radius

    if sign > 0.0:  # LEFT turn: left rear inner
        return inner, outer
    return outer, inner


def _fullplant_write_vehicle_ctrl(self, steer: float, wheel: float):
    if not getattr(self, "_fullplant_installed", False):
        _apply_static_full_plant(self)

    model, data = self.model, self.data
    L = float(self._fullplant_wheelbase)
    T = float(self._fullplant_track)

    sl = _oid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_front_steer_act")
    sr = _oid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_front_steer_act")
    dl = _oid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_rear_wheel_act")
    dr = _oid(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_rear_wheel_act")

    if USE_ACKERMANN:
        left_steer, right_steer = _ackermann(steer, L, T)
    else:
        left_steer = right_steer = float(steer)

    if USE_REAR_DIFFERENTIAL:
        left_wheel, right_wheel = _rear_diff(steer, wheel, L, T)
    else:
        left_wheel = right_wheel = float(wheel)

    data.ctrl[sl] = _clip_ctrl(model, sl, left_steer)
    data.ctrl[sr] = _clip_ctrl(model, sr, right_steer)
    data.ctrl[dl] = _clip_ctrl(model, dl, left_wheel)
    data.ctrl[dr] = _clip_ctrl(model, dr, right_wheel)


def _patched_init(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    _apply_static_full_plant(self)


def install_full_plant_patch():
    global _PATCHED, _ORIGINAL_INIT, _ORIGINAL_WRITE

    if _PATCHED:
        return

    cls = v12.V12BasicEnv
    _ORIGINAL_INIT = cls.__init__
    _ORIGINAL_WRITE = cls._write_vehicle_ctrl

    cls.__init__ = _patched_init
    cls._write_vehicle_ctrl = _fullplant_write_vehicle_ctrl

    _PATCHED = True

    print("FULL_PLANT_PATCH_INSTALLED = True")
    print("FULL_PLANT_NAME = CONDIM3_ACKERMANN_REAR_DIFF_FRONT_FREE_KP_X2")
    print("FULL_PLANT_ACKERMANN = True")
    print("FULL_PLANT_REAR_DIFF = True")
    print("FULL_PLANT_FRONT_FREE = True")
    print("FULL_PLANT_STEER_KP_SCALE = 2.0")


def full_plant_installed():
    return bool(_PATCHED)
