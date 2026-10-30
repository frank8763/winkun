from pathlib import Path
from collections import deque
import hashlib
import importlib.util
import json
import sys
import traceback

import numpy as np
import torch

H = Path.home()

JOB = (
    H / "mkmini_stage_td3_t37_policy_forgetting_audit_20261027_v1"
)

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T31 = H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"

T35 = H / "mkmini_stage_td3_t35_random3m_20261027_v1"

CORE = T16 / "td3_core_goal_contrast.py"

CORE_SHA = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

MODELS = {
    "T16": {
        "path": T16 / "checkpoints/episode_0008.pt",
        "sha": (
            "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
        ),
        "frames": 13491,
    },
    "T31A": {
        "path": T31 / "t31a/checkpoints/episode_0011.pt",
        "sha": (
            "081f083010d9794d4a212a1428a04352347facfed995a4dcfb8f9957e249dc1e"
        ),
        "frames": 37491,
    },
    "T35": {
        "path": T35 / "checkpoints/episode_0016.pt",
        "sha": (
            "50fff483557cfcd2aeeb6aee40a74148cefeda581c2e9644d26986d15b095bbe"
        ),
        "frames": 61491,
    },
}


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def to_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return None


def get_observations(obj, depth=0):
    """
    兼容已有TD3回放格式。
    优先提取完整二维观测矩阵。
    对transition列表检查所有可访问记录。
    """

    if depth > 6:
        return None

    arr = to_array(obj)

    if arr is not None:
        if arr.ndim == 2 and arr.shape[1] == 112:
            result = np.asarray(
                arr,
                dtype=np.float32
            )

            if (
                len(result) > 0
                and np.isfinite(result).all()
            ):
                return result

        return None

    if isinstance(obj, dict):

        preferred = [
            "states",
            "state",
            "obs",
            "observations",
            "storage",
            "buffer",
            "data",
            "memory",
        ]

        keys = [
            key for key in preferred
            if key in obj
        ]

        keys.extend(
            key for key in obj
            if key not in keys
        )

        for key in keys:
            result = get_observations(
                obj[key],
                depth + 1
            )

            if result is not None:
                return result

        return None

    if isinstance(obj, (list, tuple, deque)):

        if len(obj) == 0:
            return None

        first = obj[0]

        if isinstance(first, (tuple, list, dict)):

            observations = []

            for row in obj:

                value = None

                if isinstance(row, dict):
                    for key in (
                        "state",
                        "obs",
                        "observation"
                    ):
                        if key in row:
                            value = row[key]
                            break

                elif len(row) > 0:
                    value = row[0]

                if value is None:
                    break

                arr = to_array(value)

                if arr is None:
                    try:
                        arr = np.asarray(value)
                    except Exception:
                        break

                if arr.size != 112:
                    break

                observations.append(
                    np.asarray(
                        arr,
                        dtype=np.float32
                    ).reshape(112)
                )

            if (
                len(observations) == len(obj)
                and len(observations) > 0
            ):
                result = np.stack(observations)

                if np.isfinite(result).all():
                    return result

        for item in list(obj)[:8]:
            result = get_observations(
                item,
                depth + 1
            )

            if result is not None:
                return result

    return None


def print_coverage(name, obs):

    front = obs[:, 8:40].min(axis=1)
    all_lidar = obs[:, :48].min(axis=1)
    goal_m = obs[:, 48] * 10.0

    groups = {
        "FRONT_LT_0P6M": front < 0.10,
        "FRONT_LT_0P3M": front < 0.05,
        "ANY_RAY_LT_0P6M": all_lidar < 0.10,
        "GOAL_0P2_TO_0P6M": (
            (goal_m >= 0.2) & (goal_m <= 0.6)
        ),
        "GOAL_LT_0P4M": goal_m < 0.4,
    }

    print(
        "T37_REPLAY",
        name,
        "ROWS=", len(obs),
        "FRONT_MIN=", float(front.min()),
        "GOAL_MIN_M=", float(goal_m.min())
    )

    for label, mask in groups.items():

        count = int(mask.sum())

        print(
            "T37_COVERAGE",
            name,
            label,
            "COUNT=", count,
            "PERCENT=",
            round(100.0 * count / len(obs), 3)
        )

    return groups


def evaluate_actions(label, actor, obs, groups):

    torch.set_num_threads(2)

    throttle_parts = []
    steer_parts = []

    actor.eval()

    with torch.no_grad():

        for begin in range(0, len(obs), 2048):

            part = torch.from_numpy(
                np.ascontiguousarray(
                    obs[begin:begin + 2048]
                )
            )

            raw = actor(part).cpu().numpy()

            throttle_parts.append(
                np.clip(
                    (raw[:, 0] + 1.0) / 2.0,
                    0.0,
                    1.0
                )
            )

            steer_parts.append(
                np.clip(raw[:, 1], -1.0, 1.0)
            )

    throttle = np.concatenate(throttle_parts)
    steer = np.concatenate(steer_parts)

    signed_goal_steer = (
        np.sign(obs[:, 49]) * steer
    )

    direction_mask = (
        (np.abs(obs[:, 49]) > 0.17)
        & (obs[:, 50] > 0.2)
        & (obs[:, :48].min(axis=1) > 0.12)
    )

    selected = {
        "ALL": np.ones(len(obs), dtype=bool),
        **groups,
        "CLEAR_DIRECTION": direction_mask,
    }

    for group_name, mask in selected.items():

        count = int(mask.sum())

        if count == 0:
            print(
                "T37_ACTOR",
                label,
                group_name,
                "COUNT=0"
            )
            continue

        print(
            "T37_ACTOR",
            label,
            group_name,
            "COUNT=", count,
            "THROTTLE=",
            round(float(throttle[mask].mean()), 5),
            "ABS_STEER=",
            round(float(np.abs(steer[mask]).mean()), 5),
            "SIGNED_STEER=",
            round(float(signed_goal_steer[mask].mean()), 5)
        )

    return throttle, steer


def main():

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("READ_ONLY=TRUE")
    print("EXTERNAL_PLANNER=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTED=FALSE")

    require(
        CORE.is_file() and sha(CORE) == CORE_SHA,
        "CORE_INTEGRITY_FAILED"
    )

    spec = importlib.util.spec_from_file_location(
        "t37_stage_core",
        CORE
    )

    require(
        spec is not None and spec.loader is not None,
        "CORE_IMPORT_FAILED"
    )

    core = importlib.util.module_from_spec(spec)
    sys.modules["t37_stage_core"] = core

    spec.loader.exec_module(core)

    states = {}
    actors = {}

    print("\n========== T37 检查点验证 ==========")

    for name, config in MODELS.items():

        path = config["path"]

        require(
            path.is_file() and sha(path) == config["sha"],
            "CHECKPOINT_CHANGED:" + name
        )

        ck = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        require(
            ck.get("algorithm") == "TD3"
            and ck.get("frames") == config["frames"]
            and ck["manifest"]["source_core"] == CORE_SHA,
            "MODEL_PROVENANCE_INVALID:" + name
        )

        actor = core.Actor(112, 2)

        actor.load_state_dict(
            ck["actor"],
            strict=True
        )

        actor.eval()
        actors[name] = actor

        print(
            "T37_VERIFIED",
            name,
            "FRAMES=", ck.get("frames"),
            "UPDATES=", ck.get("updates"),
            "EPISODES=", ck.get("episode_index"),
            "SUCCESS_COUNT=", ck.get("success"),
            "COLLISION_COUNT=", ck.get("collisions")
        )

        replay = ck.get("replay")

        if replay is None:
            print(
                "T37_REPLAY_UNAVAILABLE=",
                name
            )
        else:
            print(
                "T37_REPLAY_TYPE=",
                name,
                type(replay).__name__
            )

            if isinstance(replay, dict):
                print(
                    "T37_REPLAY_KEYS=",
                    name,
                    list(replay.keys())[:25]
                )

            obs = get_observations(replay)

            if obs is not None:
                states[name] = obs
            else:
                print(
                    "T37_REPLAY_FORMAT_NEEDS_INSPECTION=",
                    name
                )

    require(
        "T16" in states and "T35" in states,
        "REPLAY_FORMAT_NOT_RESOLVED"
    )

    for cohort in ("T16", "T35"):

        print(
            "\n========== T37 状态来源",
            cohort,
            "=========="
        )

        obs = states[cohort]

        groups = print_coverage(
            cohort,
            obs
        )

        # 相同状态下对比不同模型，避免状态分布混淆。
        # 只用于诊断，不用于控制。
        sample_step = max(
            1,
            len(obs) // 8192
        )

        test_obs = np.asarray(
            obs[::sample_step][:8192],
            dtype=np.float32
        )

        test_groups = print_coverage(
            cohort + "_ACTION_SAMPLE",
            test_obs
        )

        responses = {}

        for model in ("T16", "T31A", "T35"):

            responses[model] = evaluate_actions(
                cohort + "_" + model,
                actors[model],
                test_obs,
                test_groups
            )

        t16_throttle, t16_steer = responses["T16"]

        for model in ("T31A", "T35"):

            throttle, steer = responses[model]

            print(
                "T37_POLICY_DRIFT",
                "OBS_SOURCE=", cohort,
                "MODEL=", model,
                "MINUS_T16_THROTTLE=",
                round(
                    float(
                        (throttle - t16_throttle).mean()
                    ),
                    6
                ),
                "MEAN_ABS_STEER_DIFFERENCE=",
                round(
                    float(
                        np.abs(
                            steer - t16_steer
                        ).mean()
                    ),
                    6
                )
            )

    print("\nT37_REPLAY_AUDIT=PASS")
    print("T37_NO_MODEL_PROMOTION=TRUE")


try:
    main()

except Exception as error:
    print(
        "T37_AUDIT=BLOCKED",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
