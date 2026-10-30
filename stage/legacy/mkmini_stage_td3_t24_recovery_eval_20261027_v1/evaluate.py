from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t24_recovery_eval_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T23 = H / "mkmini_stage_td3_t23_balanced_curriculum_20261027_v1"

CORE = T16 / "td3_core_goal_contrast.py"

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

MODELS = {
    "T16_13491": {
        "path": T16 / "checkpoints/episode_0008.pt",
        "sha": "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
        "frames": 13491,
        "updates": 6618,
        "episode": 8,
    },
    "T23_18091": {
        "path": T23 / "checkpoints/episode_0009.pt",
        "sha": "d1dfd19d3cfd791c7f536e05623340c39a61ca81eb8b0d235058c2301026da8b",
        "frames": 18091,
        "updates": 8918,
        "episode": 9,
    }
}

SOURCES = {
    CORE:
        "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

TASKS = [
    (seed, angle, distance)
    for seed in (2029648101, 2029648102)
    for distance in (1.2, 3.0)
    for angle in (-20, 20)
]

CAP = 14000
WALL = 4500


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(name, value):
    print(
        name,
        json.dumps(value, ensure_ascii=False),
        flush=True
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    check(
        spec is not None and spec.loader is not None,
        "MODULE_SPEC_INVALID:" + name
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path, data):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(temp, path)


def prepare():
    for path, digest in SOURCES.items():
        check(
            path.is_file() and sha(path) == digest,
            "SOURCE_CHANGED:" + str(path)
        )

    actors = {}
    core = load_module(CORE, "stage_t24_core")

    for name, cfg in MODELS.items():
        path = cfg["path"]

        check(
            path.is_file() and sha(path) == cfg["sha"],
            "CHECKPOINT_CHANGED:" + name
        )

        state = torch.load(
            path, map_location="cpu", weights_only=False
        )

        check(
            state["algorithm"] == "TD3"
            and state["frames"] == cfg["frames"]
            and state["updates"] == cfg["updates"]
            and state["episode_index"] == cfg["episode"],
            "CHECKPOINT_METADATA_INVALID:" + name
        )

        actor = core.Actor(112, 2)
        actor.load_state_dict(state["actor"])
        actor.eval()
        actors[name] = actor

    manifest = {
        "experiment": "STAGE_TD3_T24",
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "cap": CAP,
        "training": False,
        "noise": 0,
        "planner": False,
        "automatic_promotion": False
    }

    mf = JOB / "manifest.json"

    if mf.exists():
        check(
            json.loads(mf.read_text()) == manifest,
            "EXPERIMENT_MANIFEST_CHANGED"
        )
    else:
        save_json(mf, manifest)

    emit("T24_PREFLIGHT=PASS", manifest)
    return actors


def run_case(
    helper, make_base, v, native_mod, N,
    actor, model_name, seed, angle, distance, deadline
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
                seed,
                angle,
                distance
            )
        )

        env.max_steps = CAP

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        check(
            obs.shape == (112,)
            and np.isfinite(obs).all(),
            "OBSERVATION_INVALID"
        )

        start = np.asarray(env.data.qpos[:2], dtype=float).tolist()
        goal = np.asarray(env.goal_xy, dtype=float).tolist()

        initial = float(env._goal_state()[1])

        check(
            abs(initial - distance) < 0.04,
            "DISTANCE_GATE_FAILED"
        )

        closest = initial
        success = False
        collision = False
        reason = "CAP"
        throttle_sum = 0.0
        steer_sum = 0.0

        for frame in range(1, CAP + 1):
            if time.monotonic() >= deadline:
                return None

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(obs.reshape(1, -1))
                ).cpu().numpy()[0]

            check(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "ACTION_INVALID"
            )

            action = np.asarray(
                [
                    np.clip((float(raw[0]) + 1) / 2, 0, 1),
                    np.clip(float(raw[1]), -1, 1)
                ],
                dtype=np.float32
            )

            throttle_sum += float(action[0])
            steer_sum += float(action[1])

            next_obs, _, terminated, truncated, info = (
                N.step_checked(env, action)
            )

            obs = np.asarray(
                next_obs,
                dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "STEP_OBSERVATION_INVALID"
            )

            remaining = float(env._goal_state()[1])
            closest = min(closest, remaining)

            success = bool(info.get("reach_goal", False))
            collision = bool(info.get("collision", False))

            if frame % 3500 == 0:
                emit("T24_PROGRESS", {
                    "model": model_name,
                    "distance": distance,
                    "angle": angle,
                    "frames": frame,
                    "remaining_m": round(remaining, 4)
                })

            if success or collision or terminated or truncated:
                reason = (
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        return {
            "model": model_name,
            "requested_seed": seed,
            "actual_seed": int(actual_seed),
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start,
            "goal_xy": goal,
            "frames": frame,
            "initial_m": initial,
            "closest_m": closest,
            "remaining_m": remaining,
            "success": success,
            "collision": collision,
            "end": reason,
            "mean_throttle": throttle_sum / frame,
            "mean_steer": steer_sum / frame
        }

    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    torch.set_num_threads(2)

    actors = prepare()

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(HELPER, "stage_t24_helper")
    legacy = load_module(STAGE, "stage_t24_legacy")

    for path, digest in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == digest,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(legacy.SRC, "stage_t24_native")
    _, v, native_mod, prep, _ = runner.load_native()

    check(len(prep) >= 4, "NATIVE_BOOTSTRAP_INVALID")

    N = prep[3]
    make_base = legacy.make_short_task(runner.make_task)

    deadline = time.monotonic() + WALL
    all_pairs = []

    for index, (seed, angle, distance) in enumerate(TASKS, 1):
        pair = []

        for model_name, actor in actors.items():
            result_file = (
                JOB / f"task_{index:02d}_{model_name}.json"
            )

            if result_file.exists():
                result = json.loads(result_file.read_text())
            else:
                result = run_case(
                    helper, make_base, v, native_mod, N,
                    actor, model_name, seed, angle, distance,
                    deadline
                )

                if result is None:
                    print("T24_STATE=PAUSED_RESUMABLE")
                    return

                save_json(result_file, result)
                emit("T24_CASE", result)

            check(
                result["model"] == model_name
                and result["requested_seed"] == seed
                and result["angle_deg"] == angle
                and result["distance_m"] == distance,
                "RESUME_TASK_MISMATCH"
            )

            pair.append(result)

        a, b = pair

        check(
            a["actual_seed"] == b["actual_seed"]
            and a["start_xy"] == b["start_xy"]
            and a["goal_xy"] == b["goal_xy"],
            "PAIR_INITIAL_STATE_MISMATCH"
        )

        all_pairs.append(pair)

        emit("T24_PAIR", {
            "task": index,
            "distance_m": distance,
            "angle_deg": angle,
            "T16": a["end"],
            "T23_EARLY": b["end"],
            "T16_closest_m": round(a["closest_m"], 4),
            "T23_closest_m": round(b["closest_m"], 4)
        })

    for model in MODELS:
        data = [
            r for pair in all_pairs
            for r in pair
            if r["model"] == model
        ]

        emit("T24_MODEL", {
            "model": model,
            "runs": len(data),
            "success": sum(r["success"] for r in data),
            "collisions": sum(r["collision"] for r in data),
            "left_success": sum(
                r["success"] for r in data
                if r["angle_deg"] < 0
            ),
            "right_success": sum(
                r["success"] for r in data
                if r["angle_deg"] > 0
            ),
            "success_3m": sum(
                r["success"] for r in data
                if r["distance_m"] == 3.0
            ),
            "mean_closest_m": round(
                sum(r["closest_m"] for r in data) / len(data),
                5
            )
        })

    check(
        sha(FROZEN) == SOURCES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    print("T24_STATE=EVALUATED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()
except Exception as e:
    print("T24_BLOCKED=", repr(e))
    traceback.print_exc()
finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
