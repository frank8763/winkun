from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t21_terminal_trace_20261027_v1"

CORE = H / (
    "mkmini_stage_td3_t16_long_finetune_20261027_v1/"
    "td3_core_goal_contrast.py"
)

HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_"
    "20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

MODELS = {
    "T16": (
        H / "mkmini_stage_td3_t16_long_finetune_20261027_v1/"
        "checkpoints/episode_0009.pt",
        "15d691e0b7809015db146c6cd3d651c3d7e790083f7bddb9646c3a39e75ffb87"
    ),
    "T19R": (
        H / "mkmini_stage_td3_t19r_manifest_fix_20261027_v1/"
        "checkpoints/episode_0009.pt",
        "186f9933ee1e58ef1d401be105c1b32c7e52d012e76850fce40790bd71932628"
    )
}

SOURCE_HASHES = {
    CORE: "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    HELPER: "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE: "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN: "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
}

# 复用T20已测试场景，仅作机理诊断。
# 不将结果视为新的独立泛化评估。
TASKS = [
    (2029637102, -20, 1.0),
    (2029637102,  20, 1.0),
]

CAP = 8000
WALL = 1800


def sha(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1048576), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def emit(label, data):
    print(
        label,
        json.dumps(
            data,
            ensure_ascii=False,
            default=str
        ),
        flush=True
    )


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "MODULE_SPEC_INVALID"
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def atomic_json(path, data):
    tmp = Path(str(path) + ".partial")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(tmp, path)


def verify():
    for path, expected in SOURCE_HASHES.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_CHANGED:" + str(path)
        )

    for name, (path, expected) in MODELS.items():
        check(
            path.is_file() and sha(path) == expected,
            "MODEL_CHANGED:" + name
        )

    manifest = {
        "experiment": "STAGE_TD3_T21",
        "models": {
            name: expected
            for name, (_, expected) in MODELS.items()
        },
        "tasks": TASKS,
        "cap": CAP,
        "training": False,
        "action_override": False,
        "automatic_promotion": False
    }

    path = JOB / "manifest.json"

    if path.exists():
        check(
            json.loads(path.read_text()) == manifest,
            "MANIFEST_CHANGED"
        )
    else:
        atomic_json(path, manifest)

    emit("T21_PREFLIGHT=PASS", manifest)


def simulate(
    helper,
    make_base,
    v,
    native_mod,
    N,
    actor,
    name,
    task,
    deadline
):
    seed, angle, distance = task

    env = None
    renderer = None

    try:
        env, renderer, obs, actual_seed = (
            helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                seed,
                angle,
                distance
            )
        )

        env.max_steps = CAP

        obs = np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)

        check(
            obs.shape == (112,),
            "INITIAL_OBSERVATION_INVALID"
        )

        start_xy = np.asarray(
            env.data.qpos[:2],
            dtype=float
        ).copy()

        goal_xy = np.asarray(
            env.goal_xy,
            dtype=float
        ).copy()

        initial_m = float(
            env._goal_state()[1]
        )

        check(
            abs(initial_m - distance) < 0.03,
            "INITIAL_DISTANCE_INVALID"
        )

        closest_m = initial_m
        closest_state = None

        history = []

        near_throttle = []
        near_speed = []
        near_bearing = []
        near_steer = []

        success = False
        collision = False
        reason = "CAP"

        for frame in range(1, CAP + 1):

            if time.monotonic() >= deadline:
                emit("T21_PAUSED=", {
                    "model": name,
                    "angle": angle,
                    "uncommitted_frames": frame - 1
                })
                return None

            old_xy = np.asarray(
                env.data.qpos[:2],
                dtype=float
            ).copy()

            old_time = float(
                env.data.time
            )

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

            action = np.asarray([
                0.5 * (float(raw[0]) + 1),
                float(raw[1])
            ], dtype=np.float32)

            obs_new, reward, terminated, truncated, info = (
                N.step_checked(env, action)
            )

            obs = np.asarray(
                obs_new, dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "STEP_OBSERVATION_INVALID"
            )

            xy = np.asarray(
                env.data.qpos[:2],
                dtype=float
            )

            dt = float(env.data.time) - old_time

            check(
                dt > 0,
                "NONPOSITIVE_PHYSICS_DT"
            )

            speed = float(
                np.linalg.norm(xy - old_xy) / dt
            )

            goal = env._goal_state()

            remaining = float(goal[1])

            bearing = math.degrees(
                float(goal[2])
            )

            if remaining < 0.45:
                near_throttle.append(float(action[0]))
                near_speed.append(speed)
                near_bearing.append(abs(bearing))
                near_steer.append(float(action[1]))

            if remaining < closest_m:
                closest_m = remaining

                closest_state = {
                    "frame": frame,
                    "distance_m": remaining,
                    "bearing_deg": bearing,
                    "speed_mps": speed,
                    "throttle": float(action[0]),
                    "steer": float(action[1]),
                    "xy": xy.tolist()
                }

            if frame % 100 == 0 or remaining < 0.21:
                if frame % 25 == 0 or frame % 100 == 0:
                    history.append({
                        "frame": frame,
                        "distance_m": remaining,
                        "bearing_deg": bearing,
                        "speed_mps": speed,
                        "throttle": float(action[0]),
                        "steer": float(action[1])
                    })

            success = bool(
                info.get("reach_goal", False)
            )

            collision = bool(
                info.get("collision", False)
            )

            if success or collision or terminated or truncated:
                reason = (
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        result = {
            "model": name,
            "requested_seed": seed,
            "actual_seed": actual_seed,
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start_xy.tolist(),
            "goal_xy": goal_xy.tolist(),
            "frames": frame,
            "success": success,
            "collision": collision,
            "end": reason,
            "closest_m": closest_m,
            "final_m": remaining,
            "drift_m": remaining - closest_m,
            "closest_state": closest_state,
            "near_samples": len(near_throttle),
            "near_mean_throttle": (
                float(np.mean(near_throttle))
                if near_throttle else None
            ),
            "near_mean_speed_mps": (
                float(np.mean(near_speed))
                if near_speed else None
            ),
            "near_mean_abs_bearing_deg": (
                float(np.mean(near_bearing))
                if near_bearing else None
            ),
            "near_mean_steer": (
                float(np.mean(near_steer))
                if near_steer else None
            ),
            "trace": history
        }

        emit("T21_CASE=", {
            k: value
            for k, value in result.items()
            if k != "trace"
        })

        return result

    finally:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("EXTERNAL_PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    verify()

    torch.set_num_threads(2)

    core = load(CORE, "stage_t21_td3_core")

    actors = {}

    for name, (path, _) in MODELS.items():
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            checkpoint.get("algorithm") == "TD3",
            "ALGORITHM_INVALID:" + name
        )

        actor = core.Actor(112, 2)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        actors[name] = actor

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(HELPER, "stage_t21_helper")
    legacy = load(STAGE, "stage_t21_legacy")

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED"
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_t21_native"
    )

    _, v, native_mod, prep, hashes = runner.load_native()

    check(len(prep) >= 4, "NATIVE_BOOTSTRAP_INVALID")

    N = prep[3]
    make_base = legacy.make_short_task(runner.make_task)

    deadline = time.monotonic() + WALL

    for index, task in enumerate(TASKS, start=1):

        previous = None

        for name, actor in actors.items():

            path = JOB / f"case_{index}_{name}.json"

            if path.exists():
                result = json.loads(path.read_text())
                print("T21_EXISTING_CASE_VERIFIED=", path.name)
            else:
                result = simulate(
                    helper,
                    make_base,
                    v,
                    native_mod,
                    N,
                    actor,
                    name,
                    task,
                    deadline
                )

                if result is None:
                    print("T21_STATE=PAUSED_RESUMABLE")
                    return

                atomic_json(path, result)

            check(
                result["requested_seed"] == task[0]
                and result["angle_deg"] == task[1]
                and result["distance_m"] == task[2],
                "RESUME_TASK_MISMATCH"
            )

            if previous is not None:
                for key in ("actual_seed", "start_xy", "goal_xy"):
                    check(
                        previous[key] == result[key],
                        "PAIR_INITIAL_STATE_MISMATCH:" + key
                    )

                emit("T21_PAIR=", {
                    "angle_deg": task[1],
                    "T16_closest_m": previous["closest_m"],
                    "T19R_closest_m": result["closest_m"],
                    "T16_near_throttle":
                        previous["near_mean_throttle"],
                    "T19R_near_throttle":
                        result["near_mean_throttle"],
                    "T16_near_speed":
                        previous["near_mean_speed_mps"],
                    "T19R_near_speed":
                        result["near_mean_speed_mps"],
                    "T16_nearest": previous["closest_state"],
                    "T19R_nearest": result["closest_state"]
                })

            previous = result

    check(
        sha(FROZEN) == SOURCE_HASHES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    print("T21_STATE=DIAGNOSIS_COMPLETED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()

except Exception as error:
    print("T21_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
