from __future__ import annotations

import contextvars
import math
from typing import Any

import mujoco
import numpy as np

# Stage9 Runtime FULL_KP_X2 Fix V5.2
#
# Root cause fixed from V5:
# V4 identified Phase10Stage1ADenseEnv.step as the actual runtime owner
# that executes mujoco.mj_step(). V5 incorrectly wrapped the deeper
# Phase10SceneE2ELidarEnv.step, which is not reached by the active bridge
# physics path. V5.2 wraps the exact Stage1Dense owner.
#
# Runtime design:
#   upstream wrappers
#       -> Phase10Stage1ADenseEnv.step(action)
#           -> base code writes PARALLEL front ctrl + equal rear ctrl
#           -> before every real mujoco.mj_step:
#                first raw parallel front pair -> center steer
#                Ackermann(center) -> LF/RF
#                raw rear mean -> rear differential LR/RR
#           -> real mujoco.mj_step
#
# No qpos/qvel overwrite.
# No RL action override.
# No steering floor.
# No speed override.
# No angle-limit expansion in this stage.

WHEELBASE_M = 0.60
TRACK_M = 0.54
STEER_KP = 2400.0
STEER_FORCE_LIMIT = 500.0

STEER_ACTUATORS = (
    "left_front_steer_act",
    "right_front_steer_act",
)
REAR_ACTUATORS = (
    "left_rear_wheel_act",
    "right_rear_wheel_act",
)
FRONT_WHEEL_JOINTS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
)
ALL_WHEEL_JOINTS = (
    "left_front_wheel_joint",
    "right_front_wheel_joint",
    "left_rear_wheel_joint",
    "right_rear_wheel_joint",
)

_REAL_MJ_STEP = mujoco.mj_step
_CONTEXT = contextvars.ContextVar(
    "stage9_v5_2_fullplant_context",
    default=None,
)

_INSTALLED = False
_ORIGINAL_DENSE_INIT = None
_ORIGINAL_DENSE_STEP = None

_LAST_STEP_RECORD: dict[int, dict[str, Any]] = {}


def _id(model: mujoco.MjModel, obj_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, obj_type, name))
    if value < 0:
        raise RuntimeError(f"missing MuJoCo object: {name}")
    return value


def _clip_ctrl(model: mujoco.MjModel, aid: int, value: float) -> float:
    lo = float(model.actuator_ctrlrange[aid, 0])
    hi = float(model.actuator_ctrlrange[aid, 1])
    return float(np.clip(float(value), lo, hi))


def _wheel_collision_geom(model: mujoco.MjModel, joint_name: str) -> int:
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    body = int(model.jnt_bodyid[jid])
    start = int(model.body_geomadr[body])
    count = int(model.body_geomnum[body])

    candidates = []
    for gid in range(start, start + count):
        if (
            int(model.geom_contype[gid]) == 0
            or int(model.geom_conaffinity[gid]) == 0
        ):
            continue
        if int(model.geom_type[gid]) in (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        ):
            candidates.append(int(gid))

    if len(candidates) != 1:
        raise RuntimeError(
            f"{joint_name}: expected one active wheel collision geom, "
            f"found {candidates}"
        )
    return int(candidates[0])


def apply_static_fullplant(env: Any) -> dict[str, Any]:
    """
    Restore only the already validated FULL_KP_X2 static plant.
    IMPORTANT: do not expand the old +/-0.45 rad angle limits here.
    """
    model = env.model

    wheel_geoms = {
        name: _wheel_collision_geom(model, name)
        for name in ALL_WHEEL_JOINTS
    }
    for gid in wheel_geoms.values():
        model.geom_condim[int(gid)] = 3

    plane_ids = []
    for gid in range(int(model.ngeom)):
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            model.geom_condim[gid] = 3
            plane_ids.append(int(gid))
    if not plane_ids:
        raise RuntimeError("no ground plane found")

    model.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    model.opt.noslip_iterations = 0

    for joint_name in FRONT_WHEEL_JOINTS:
        jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        did = int(model.jnt_dofadr[jid])
        model.dof_damping[did] = 0.0
        model.dof_frictionloss[did] = 0.0
        model.dof_armature[did] = 1.0e-5

    # Default remains the historically validated KP=2400, but calibration
    # environments may explicitly override it. This value is re-applied on
    # every env.step, so the override MUST be read here rather than written
    # only once at environment construction.
    steer_kp = float(
        getattr(env, "_stage9_steer_kp_override", STEER_KP)
    )
    if not np.isfinite(steer_kp) or steer_kp <= 0.0:
        raise RuntimeError(
            f"invalid _stage9_steer_kp_override={steer_kp}"
        )

    # R17G-I: force-limit override mirrors the already-supported KP override.
    # With no override attribute present, behavior is exactly unchanged.
    steer_force_limit = float(
        getattr(
            env,
            "_stage9_steer_force_limit_override",
            STEER_FORCE_LIMIT,
        )
    )
    if not np.isfinite(steer_force_limit) or steer_force_limit <= 0.0:
        raise RuntimeError(
            "invalid _stage9_steer_force_limit_override="
            f"{steer_force_limit}"
        )

    for actuator_name in STEER_ACTUATORS:
        aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        model.actuator_gainprm[aid, 0] = steer_kp
        model.actuator_biasprm[aid, 1] = -steer_kp
        model.actuator_forcelimited[aid] = 1
        model.actuator_forcerange[aid, 0] = -steer_force_limit
        model.actuator_forcerange[aid, 1] = +steer_force_limit

    env._stage9_v5_2_static_fullplant = True

    return {
        "wheel_condim": {
            name: int(model.geom_condim[gid])
            for name, gid in wheel_geoms.items()
        },
        "plane_condim": {
            gid: int(model.geom_condim[gid])
            for gid in plane_ids
        },
    }


def ackermann(
    center_steer: float,
    wheelbase: float = WHEELBASE_M,
    track: float = TRACK_M,
) -> tuple[float, float]:
    center = float(center_steer)
    mag = abs(center)
    if mag < 1.0e-12:
        return 0.0, 0.0

    radius = float(wheelbase) / math.tan(mag)
    inner = math.atan2(
        float(wheelbase),
        max(radius - float(track) / 2.0, 1.0e-9),
    )
    outer = math.atan2(
        float(wheelbase),
        radius + float(track) / 2.0,
    )

    if center > 0.0:
        return float(inner), float(outer)
    return float(-outer), float(-inner)


def rear_differential(
    center_steer: float,
    wheel_mean: float,
    wheelbase: float = WHEELBASE_M,
    track: float = TRACK_M,
) -> tuple[float, float]:
    center = float(center_steer)
    wheel = float(wheel_mean)

    if abs(center) < 1.0e-12 or abs(wheel) < 1.0e-12:
        return wheel, wheel

    radius = float(wheelbase) / math.tan(abs(center))
    inner = wheel * (radius - float(track) / 2.0) / radius
    outer = wheel * (radius + float(track) / 2.0) / radius

    if center > 0.0:
        return float(inner), float(outer)
    return float(outer), float(inner)


def _apply_runtime_fullplant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: dict[str, Any],
) -> None:
    sl = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_front_steer_act")
    sr = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_front_steer_act")
    dl = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_rear_wheel_act")
    dr = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_rear_wheel_act")

    raw_sl = float(data.ctrl[sl])
    raw_sr = float(data.ctrl[sr])
    raw_dl = float(data.ctrl[dl])
    raw_dr = float(data.ctrl[dr])

    call_index = int(ctx["physics_calls"])

    if call_index == 0:
        # V4 proved that the active Stage1Dense runtime writes equal LF/RF
        # parallel steering before physics. Derive the REAL center command
        # from that command itself instead of guessing from action[1].
        front_gap = abs(raw_sl - raw_sr)
        if front_gap > 1.0e-6:
            raise RuntimeError(
                "FIRST_RAW_FRONT_NOT_PARALLEL: "
                f"lf={raw_sl} rf={raw_sr} gap={front_gap}"
            )

        ctx["center_steer"] = 0.5 * (raw_sl + raw_sr)

        # Likewise the current base runtime writes equal rear commands.
        # The first raw mean is the center/rear speed command that must
        # be preserved by differential conversion.
        ctx["rear_mean"] = 0.5 * (raw_dl + raw_dr)

        ctx["first_raw"] = {
            "lf": raw_sl,
            "rf": raw_sr,
            "lr": raw_dl,
            "rr": raw_dr,
            "front_gap": front_gap,
            "rear_gap": abs(raw_dl - raw_dr),
        }

    center = float(ctx["center_steer"])

    # Keep the original rear mean for this env.step. This prevents repeated
    # mj_step substeps from recursively reinterpreting the already-converted
    # LR/RR pair.
    rear_mean = float(ctx["rear_mean"])

    lf, rf = ackermann(center)
    lr, rr = rear_differential(center, rear_mean)

    lf = _clip_ctrl(model, sl, lf)
    rf = _clip_ctrl(model, sr, rf)
    lr = _clip_ctrl(model, dl, lr)
    rr = _clip_ctrl(model, dr, rr)

    data.ctrl[sl] = lf
    data.ctrl[sr] = rf
    data.ctrl[dl] = lr
    data.ctrl[dr] = rr

    ctx["physics_calls"] = call_index + 1
    ctx["last_applied"] = {
        "lf": lf,
        "rf": rf,
        "lr": lr,
        "rr": rr,
        "rear_mean_after": 0.5 * (lr + rr),
    }


def _patched_mj_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *args,
    **kwargs,
):
    ctx = _CONTEXT.get()
    if (
        ctx is not None
        and id(model) == int(ctx["model_id"])
        and id(data) == int(ctx["data_id"])
    ):
        _apply_runtime_fullplant(model, data, ctx)

    return _REAL_MJ_STEP(model, data, *args, **kwargs)


def get_last_step_record(data: mujoco.MjData) -> dict[str, Any] | None:
    row = _LAST_STEP_RECORD.get(id(data))
    if row is None:
        return None

    # Return a JSON-friendly shallow copy.
    result = dict(row)
    if isinstance(result.get("first_raw"), dict):
        result["first_raw"] = dict(result["first_raw"])
    if isinstance(result.get("last_applied"), dict):
        result["last_applied"] = dict(result["last_applied"])
    return result


def install_runtime_fullplant_fix_v5_2() -> None:
    global _INSTALLED
    global _ORIGINAL_DENSE_INIT
    global _ORIGINAL_DENSE_STEP

    if _INSTALLED:
        return

    from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a_dense_env import (
        Phase10Stage1ADenseEnv,
    )

    _ORIGINAL_DENSE_INIT = Phase10Stage1ADenseEnv.__init__
    _ORIGINAL_DENSE_STEP = Phase10Stage1ADenseEnv.step

    def patched_init(self, *args, **kwargs):
        _ORIGINAL_DENSE_INIT(self, *args, **kwargs)
        apply_static_fullplant(self)

    def patched_step(self, action):
        # This is the exact runtime owner found by V4.
        apply_static_fullplant(self)

        ctx = {
            "model_id": id(self.model),
            "data_id": id(self.data),
            "physics_calls": 0,
            "center_steer": None,
            "rear_mean": None,
            "first_raw": None,
            "last_applied": None,
            "action_entering_dense_step": np.asarray(
                action,
                dtype=np.float64,
            ).reshape(-1).tolist(),
        }

        token = _CONTEXT.set(ctx)
        try:
            result = _ORIGINAL_DENSE_STEP(self, action)
        finally:
            _CONTEXT.reset(token)

        if int(ctx["physics_calls"]) <= 0:
            raise RuntimeError(
                "DENSE_STEP_EXECUTED_BUT_NO_MJ_STEP_INTERCEPTED"
            )

        _LAST_STEP_RECORD[id(self.data)] = {
            "physics_calls": int(ctx["physics_calls"]),
            "center_steer": float(ctx["center_steer"]),
            "rear_mean": float(ctx["rear_mean"]),
            "first_raw": dict(ctx["first_raw"]),
            "last_applied": dict(ctx["last_applied"]),
            "action_entering_dense_step": list(
                ctx["action_entering_dense_step"]
            ),
        }

        return result

    Phase10Stage1ADenseEnv.__init__ = patched_init
    Phase10Stage1ADenseEnv.step = patched_step
    mujoco.mj_step = _patched_mj_step

    _INSTALLED = True

    print("STAGE9_V5_2_RUNTIME_OWNER = Phase10Stage1ADenseEnv.step")
    print("STAGE9_V5_2_RUNTIME_FULLPLANT_FIX_INSTALLED = True")
    print("STAGE9_V5_2_ACKERMANN = True")
    print("STAGE9_V5_2_REAR_DIFF = True")
    print("STAGE9_V5_2_FRONT_FREE = True")
    print("STAGE9_V5_2_STEER_KP_DEFAULT = 2400")
    print("STAGE9_V5_2_STEER_KP_OVERRIDE_SUPPORTED = True")
    print("STAGE9_V5_2_ANGLE_LIMIT_EXPANSION = False")
    print("STAGE9_V5_2_QPOS_QVEL_OVERRIDE = False")
    print("STAGE9_V5_2_ACTION_OVERRIDE = False")

# >>> STAGE17_E1E_REAR_DIFF_GAP_SCALE_2_BEGIN >>>
# Formal persistent plant repair.
# Evidence:
#   E1B3 frozen-action causal audit:
#       rear-differential gap magnitude was the only strong positive intervention.
#   E1D 2x2 task A/B:
#       plant-only task improvement = +0.05280495736763269 m;
#       actor adaptation contribution = 0.0 m.
#
# IMPORTANT:
#   Only the left-right rear command half-gap is scaled.
#   The mean rear-wheel command is preserved exactly.
#   No steering/friction/force/qpos/qvel override is introduced.
STAGE17_E1E_REAR_DIFF_GAP_SCALE = 2.0

if "_stage17_e1e_rear_differential_base" not in globals():
    _stage17_e1e_rear_differential_base = rear_differential

def rear_differential(center_steer, rear_mean):
    _left, _right = _stage17_e1e_rear_differential_base(
        center_steer,
        rear_mean,
    )
    _left = float(_left)
    _right = float(_right)
    _mid = 0.5 * (_left + _right)
    _half_gap = 0.5 * (_right - _left)
    _half_gap *= STAGE17_E1E_REAR_DIFF_GAP_SCALE
    return _mid - _half_gap, _mid + _half_gap
# <<< STAGE17_E1E_REAR_DIFF_GAP_SCALE_2_END <<<

