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
JOB = Path('/home/linghu/mkmini_stage_td3_t36_random3m_screen_20261027_v1')

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T31 = H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"

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

MODELS = {
    'T16': {
        'path': Path('/home/linghu/mkmini_stage_td3_t16_long_finetune_20261027_v1/checkpoints/episode_0008.pt'),
        'sha': '21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a',
        'frames': 13491,
        'core_sha': '58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68',
    },
    'T31A': {
        'path': Path('/home/linghu/mkmini_stage_td3_t31_lidar_risk_20261027_v1/t31a/checkpoints/episode_0011.pt'),
        'sha': '081f083010d9794d4a212a1428a04352347facfed995a4dcfb8f9957e249dc1e',
        'frames': 37491,
        'core_sha': '58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68',
    },
    'T35': {
        'path': Path('/home/linghu/mkmini_stage_td3_t35_random3m_20261027_v1/checkpoints/episode_0016.pt'),
        'sha': '50fff483557cfcd2aeeb6aee40a74148cefeda581c2e9644d26986d15b095bbe',
        'frames': 61491,
        'core_sha': '58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68',
    },
}

SEEDS = (2029653601, 2029653602)

TASKS = [(2029653601, -17.3, 3.0), (2029653601, 31.7, 3.0), (2029653602, -39.4, 3.0), (2029653602, 13.6, 3.0)]

CAP = 14000
WALL = 2400


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def emit(tag, data):
    print(
        tag,
        json.dumps(data, ensure_ascii=False),
        flush=True
    )


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "MODULE_IMPORT_INVALID:" + name
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


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


def prepare():
    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    for path, expected in SOURCES.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    core = load(CORE, "t32_stage_core")
    torch.set_num_threads(2)

    actors = {}

    for name, config in MODELS.items():
        path = config["path"]

        check(
            path.is_file() and sha(path) == config["sha"],
            "MODEL_SHA_FAILED:" + name
        )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3"
            and int(state["frames"]) == config["frames"],
            "MODEL_METADATA_FAILED:" + name
        )

        check(
            state["manifest"]["source_core"]
            == config["core_sha"],
            "MODEL_CORE_PROVENANCE_FAILED:" + name
        )

        actor = core.Actor(112, 2)
        actor.load_state_dict(
            state["actor"],
            strict=True
        )
        actor.eval()

        actors[name] = actor

        print(
            "T36_MODEL_VERIFIED=",
            name,
            "FRAMES=",
            config["frames"],
            flush=True
        )

    manifest = {
        "experiment": "STAGE_TD3_T36",
        "models": {
            name: item["sha"]
            for name, item in MODELS.items()
        },
        "tasks": TASKS,
        "physics_cap": CAP,
        "training": False,
        "noise": 0,
        "automatic_promotion": False
    }

    manifest_path = JOB / "manifest.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "T36_MANIFEST_CHANGED"
        )
    else:
        atomic_json(manifest_path, manifest)

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(
        HELPER,
        "t32_stage_helper"
    )

    legacy = load(
        STAGE,
        "t32_stage_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "t32_stage_native"
    )

    _, v, native_mod, prep, _ = runner.load_native()

    check(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    print("T36_PREFLIGHT=PASS", flush=True)

    return actors, helper, make_base, v, native_mod, N


def run_case(
    seed,
    angle,
    distance,
    name,
    actor,
    helper,
    make_base,
    v,
    native_mod,
    N,
    deadline
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

        obs = np.asarray(
            obs,
            dtype=np.float32
        ).reshape(-1)

        check(
            obs.shape == (112,)
            and np.isfinite(obs).all(),
            "INITIAL_OBSERVATION_INVALID"
        )

        start = np.asarray(
            env.data.qpos[:2],
            dtype=float
        ).tolist()

        goal = np.asarray(
            env.goal_xy,
            dtype=float
        ).tolist()

        initial = float(
            env._goal_state()[1]
        )

        check(
            abs(initial - distance) < 0.04,
            "ACTUAL_TASK_DISTANCE_INVALID"
        )

        closest = initial
        remaining = initial

        success = False
        collision = False
        reason = "CAP"

        danger_frames = 0
        danger_throttle_sum = 0.0

        for frame in range(1, CAP + 1):
            if time.monotonic() >= deadline:
                return None

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, 112)
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
                    0.0,
                    1.0
                )
            )

            steer = float(
                np.clip(
                    raw[1],
                    -1.0,
                    1.0
                )
            )

            # 仅统计，不干预控制。
            forward_lidar = float(
                np.min(obs[8:40])
            )

            if forward_lidar < 0.10:
                danger_frames += 1
                danger_throttle_sum += throttle

            action = np.asarray(
                [throttle, steer],
                dtype=np.float32
            )

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

            remaining = float(
                env._goal_state()[1]
            )

            closest = min(
                closest,
                remaining
            )

            success = bool(
                info.get("reach_goal", False)
            )

            collision = bool(
                info.get("collision", False)
            )

            if frame % 3500 == 0:
                emit("T36_PROGRESS", {
                    "model": name,
                    "seed": seed,
                    "angle": angle,
                    "frame": frame,
                    "remaining_m": round(
                        remaining, 4
                    )
                })

            if (
                success
                or collision
                or terminated
                or truncated
            ):
                reason = (
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        return {
            "model": name,
            "checkpoint_sha256": MODELS[name]["sha"],
            "requested_seed": seed,
            "actual_seed": int(actual_seed),
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start,
            "goal_xy": goal,
            "frames": frame,
            "closest_m": closest,
            "remaining_m": remaining,
            "success": success,
            "collision": collision,
            "end": reason,
            "danger_frames": danger_frames,
            "danger_mean_throttle": (
                danger_throttle_sum / danger_frames
                if danger_frames
                else None
            ),
            "training": False,
            "action_override": False
        }

    finally:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()


def main():
    JOB.mkdir(
        parents=True,
        exist_ok=True
    )

    actors, helper, make_base, v, native_mod, N = (
        prepare()
    )

    deadline = time.monotonic() + WALL

    complete_pairs = []

    for index, (seed, angle, distance) in enumerate(
        TASKS, 1
    ):
        group = []

        for name, actor in actors.items():
            result_path = (
                JOB / f"task_{index:02d}_{name}.json"
            )

            if result_path.exists():
                result = json.loads(
                    result_path.read_text()
                )

                check(
                    result["model"] == name
                    and result["checkpoint_sha256"]
                    == MODELS[name]["sha"]
                    and result["requested_seed"] == seed
                    and result["angle_deg"] == angle
                    and result["distance_m"] == distance
                    and result["training"] is False,
                    "RESUME_CASE_IDENTITY_FAILED"
                )

                print(
                    "T36_RESUMED_CASE=",
                    index,
                    name
                )

            else:
                result = run_case(
                    seed,
                    angle,
                    distance,
                    name,
                    actor,
                    helper,
                    make_base,
                    v,
                    native_mod,
                    N,
                    deadline
                )

                if result is None:
                    print(
                        "T36_STATE=PAUSED_RESUMABLE"
                    )
                    return

                atomic_json(
                    result_path,
                    result
                )

                emit("T36_CASE", {
                    "task": index,
                    **result
                })

            group.append(result)

        # 严格配对：必须相同起点、目标及实际场景种子。
        baseline = group[0]

        for result in group[1:]:
            check(
                result["actual_seed"]
                == baseline["actual_seed"],
                "PAIR_SEED_MISMATCH"
            )

            check(
                np.allclose(
                    result["start_xy"],
                    baseline["start_xy"],
                    atol=1e-7,
                    rtol=0
                ),
                "PAIR_START_MISMATCH"
            )

            check(
                np.allclose(
                    result["goal_xy"],
                    baseline["goal_xy"],
                    atol=1e-7,
                    rtol=0
                ),
                "PAIR_GOAL_MISMATCH"
            )

        complete_pairs.append(group)

        emit("T36_PAIR", {
            "task": index,
            "seed": seed,
            "angle": angle,
            "results": {
                result["model"]: {
                    "end": result["end"],
                    "closest_m": round(
                        result["closest_m"], 4
                    ),
                    "frames": result["frames"]
                }
                for result in group
            }
        })

    for name in MODELS:
        rows = [
            result
            for group in complete_pairs
            for result in group
            if result["model"] == name
        ]

        emit("T36_MODEL", {
            "model": name,
            "runs": len(rows),
            "success": sum(
                r["success"] for r in rows
            ),
            "collisions": sum(
                r["collision"] for r in rows
            ),
            "left_success": sum(
                r["success"]
                for r in rows
                if r["angle_deg"] < 0
            ),
            "right_success": sum(
                r["success"]
                for r in rows
                if r["angle_deg"] > 0
            ),
            "mean_closest_m": round(
                sum(
                    r["closest_m"]
                    for r in rows
                ) / len(rows),
                5
            )
        })

    check(
        len(complete_pairs) == len(TASKS),
        "EVALUATION_INCOMPLETE"
    )

    check(
        sha(FROZEN) == SOURCES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    for name, config in MODELS.items():
        check(
            sha(config["path"]) == config["sha"],
            "MODEL_CHANGED_DURING_EVALUATION:" + name
        )

    print("T36_PAIR_GATE=PASS")
    print("T36_STATE=EVALUATED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()

except Exception as error:
    print(
        "T36_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
