from pathlib import Path
from collections import deque
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t29_lidar_collision_audit_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

CORE = T16 / "td3_core_goal_contrast.py"
CKPT = T16 / "checkpoints/episode_0008.pt"

HELPER = (
    H / "mkmini_stage_td3_independent_20261027_105202_577031_v1"
    / "td3_smoke.py"
)

STAGE = (
    H / "mkmini_stage_c147_fixedgoal_sac_20261027_v1"
    / "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = (
    H / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
)

EXPECTED = {
    CORE:
        "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    CKPT:
        "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
}

# T28已确认发生碰撞的两个3米场景。
# 使用相同种子、方向、距离、检查点。
CASES = [
    {
        "name": "right_3m_collision",
        "seed": 2029652801,
        "angle": 20,
        "distance": 3.0,
        "expected_collision_frame": 7758,
        "expected_start": [-1.5, 2.75],
    },
    {
        "name": "left_3m_collision",
        "seed": 2029652802,
        "angle": -20,
        "distance": 3.0,
        "expected_collision_frame": 5273,
        "expected_start": [-1.25, 3.0],
    },
]

CAP = 14000
WALL = 1800


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "MODULE_SPEC_INVALID"
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path, data):
    temporary = Path(str(path) + ".partial")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temporary, path)


def prepare():
    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("EXTERNAL_PLANNER=FALSE")
    print("MODEL_MODIFICATION=FALSE")

    for path, expected in EXPECTED.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    torch.set_num_threads(2)

    core = load_module(CORE, "stage_t29_core")

    state = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    check(
        state["algorithm"] == "TD3"
        and state["frames"] == 13491,
        "CHECKPOINT_METADATA_INVALID"
    )

    actor = core.Actor(112, 2)
    actor.load_state_dict(state["actor"])
    actor.eval()

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER, "stage_t29_helper"
    )

    legacy = load_module(
        STAGE, "stage_t29_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_CHANGED"
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_t29_native"
    )

    _, v, native_mod, prep, _ = (
        runner.load_native()
    )

    check(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    print("T29_PREFLIGHT=PASS")

    return actor, helper, make_base, v, native_mod, N


def run_case(
    cfg, actor, helper,
    make_base, v, native_mod, N, deadline
):
    env = None
    renderer = None

    try:
        env, renderer, obs, actual_seed = (
            helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                cfg["seed"],
                cfg["angle"],
                cfg["distance"]
            )
        )

        env.max_steps = CAP

        start = np.asarray(
            env.data.qpos[:2], dtype=float
        ).copy()

        check(
            int(actual_seed) == cfg["seed"],
            "ACTUAL_SEED_MISMATCH"
        )

        check(
            np.allclose(
                start,
                cfg["expected_start"],
                atol=1e-5
            ),
            "START_POSITION_MISMATCH"
        )

        initial_dist = float(
            env._goal_state()[1]
        )

        check(
            abs(initial_dist - 3.0) < 0.01,
            "GOAL_DISTANCE_MISMATCH"
        )

        try:
            source = inspect.getsource(
                env._lidar48
            )

            path = JOB / "native_lidar48_source.txt"

            if path.exists():
                check(
                    path.read_text() == source,
                    "LIDAR_SOURCE_CHANGED"
                )
            else:
                path.write_text(source)

            print("T29_NATIVE_LIDAR_SOURCE=CAPTURED")

        except (OSError, TypeError) as error:
            print(
                "T29_LIDAR_SOURCE_UNAVAILABLE=",
                repr(error)
            )

        obs = np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)

        check(
            obs.shape == (112,),
            "OBSERVATION_DIM_INVALID"
        )

        recent = deque(maxlen=500)
        timeline = []

        closest = initial_dist
        reason = "CAP"

        success = False
        collision = False
        remaining = initial_dist
        scan_checks = []

        print(
            "T29_CASE_BEGIN=",
            cfg["name"],
            "SEED=", actual_seed,
            "START=", start.tolist(),
            flush=True
        )

        for frame in range(1, CAP + 1):

            if time.monotonic() >= deadline:
                print(
                    "T29_PAUSED_WALL_LIMIT=",
                    cfg["name"]
                )
                return None

            check(
                np.isfinite(obs).all(),
                "OBSERVATION_NONFINITE"
            )

            lidar = obs[:48].copy()

            # 注意：此处记录的是网络输入中的LiDAR特征，
            # 不假设其数值已经是米。
            lidar_min = float(np.min(lidar))

            # 分成6个特征区段，不预设哪一段对应前方。
            lidar_segments = [
                round(
                    float(np.min(lidar[i:i + 8])),
                    5
                )
                for i in range(0, 48, 8)
            ]

            pose_before = np.asarray(
                env.data.qpos[:2],
                dtype=float
            ).copy()

            time_before = float(env.data.time)

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, -1)
                    )
                ).cpu().numpy()[0]

            check(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "ACTOR_OUTPUT_INVALID"
            )

            throttle = float(
                np.clip(
                    (float(raw[0]) + 1.0) / 2.0,
                    0.0, 1.0
                )
            )

            steer = float(
                np.clip(
                    float(raw[1]),
                    -1.0, 1.0
                )
            )

            # 与T28保持完全相同的动作映射。
            action = np.asarray(
                [throttle, steer],
                dtype=np.float32
            )

            next_obs, reward, terminated, truncated, info = (
                N.step_checked(env, action)
            )

            obs = np.asarray(
                next_obs,
                dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,),
                "NEXT_OBSERVATION_INVALID"
            )

            remaining = float(
                env._goal_state()[1]
            )

            closest = min(closest, remaining)

            pose_after = np.asarray(
                env.data.qpos[:2],
                dtype=float
            )

            dt = float(
                env.data.time - time_before
            )

            speed = (
                float(
                    np.linalg.norm(
                        pose_after - pose_before
                    ) / dt
                )
                if dt > 0
                else None
            )

            record = {
                "frame": frame,
                "goal_distance_m": round(remaining, 6),
                "lidar_min_feature": round(lidar_min, 6),
                "lidar_segments": lidar_segments,
                "throttle": round(throttle, 6),
                "steer": round(steer, 6),
                "speed_mps": (
                    round(speed, 6)
                    if speed is not None
                    else None
                ),
                "reward": round(float(reward), 6)
            }

            recent.append(record)

            if frame == 1 or frame % 200 == 0:
                timeline.append(record)

            # 对比新一帧网络观测与重新读取的雷达。
            if frame % 500 == 0:
                fresh = np.asarray(
                    env._lidar48(),
                    dtype=np.float32
                ).reshape(-1)

                check(
                    fresh.shape == (48,),
                    "FRESH_LIDAR_DIM_INVALID"
                )

                scan_checks.append({
                    "frame": frame,
                    "max_abs_difference": round(
                        float(
                            np.max(
                                np.abs(
                                    fresh - obs[:48]
                                )
                            )
                        ),
                        8
                    )
                })

            if frame % 2000 == 0:
                print(
                    "T29_PROGRESS=",
                    cfg["name"],
                    "FRAME=", frame,
                    "GOAL_M=", round(remaining, 3),
                    "LIDAR_MIN_FEATURE=",
                    round(lidar_min, 4),
                    "THROTTLE=",
                    round(throttle, 3),
                    "STEER=", round(steer, 3),
                    flush=True
                )

            success = bool(
                info.get("reach_goal", False)
            )

            collision = bool(
                info.get("collision", False)
            )

            if (
                success or collision
                or terminated or truncated
            ):
                reason = (
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        recent_list = list(recent)

        report = {
            "case": cfg["name"],
            "seed": actual_seed,
            "start_xy": start.tolist(),
            "goal_xy": np.asarray(
                env.goal_xy, dtype=float
            ).tolist(),
            "frames": frame,
            "expected_collision_frame":
                cfg["expected_collision_frame"],
            "collision_reproduced": (
                collision
                and abs(
                    frame - cfg["expected_collision_frame"]
                ) <= 1
            ),
            "success": success,
            "collision": collision,
            "end": reason,
            "closest_goal_m": closest,
            "final_goal_m": remaining,
            "scan_checks": scan_checks,
            "timeline": timeline,
            "last_500_frames": recent_list,
            "training": False,
            "action_override": False
        }

        save_json(
            JOB / (cfg["name"] + ".json"),
            report
        )

        print(
            "T29_CASE_END=",
            cfg["name"],
            "END=", reason,
            "FRAME=", frame,
            "REPRODUCED=",
            report["collision_reproduced"],
            flush=True
        )

        # 显示碰撞前典型动作变化。
        for row in recent_list[::max(
            1, len(recent_list) // 8
        )]:
            print(
                "T29_PRECONTACT_TRACE=",
                json.dumps(
                    {
                        "case": cfg["name"],
                        **row
                    },
                    ensure_ascii=False
                ),
                flush=True
            )

        return report

    finally:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()


def main():
    JOB.mkdir(
        parents=True, exist_ok=True
    )

    actor, helper, make_base, v, native_mod, N = (
        prepare()
    )

    deadline = time.monotonic() + WALL

    for cfg in CASES:

        path = JOB / (cfg["name"] + ".json")

        if path.exists():
            old = json.loads(path.read_text())
            check(
                old["case"] == cfg["name"]
                and old["seed"] == cfg["seed"],
                "EXISTING_CASE_MISMATCH"
            )
            print(
                "T29_EXISTING_CASE=",
                cfg["name"]
            )
            continue

        result = run_case(
            cfg, actor, helper,
            make_base, v, native_mod, N,
            deadline
        )

        if result is None:
            print(
                "T29_STATE=PAUSED_RESUMABLE"
            )
            return

    check(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    print("T29_STATE=COMPLETED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()

except Exception as error:
    print(
        "T29_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
