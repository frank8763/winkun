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
JOB = H / "mkmini_stage_td3_t10_paired_eval_20261027_v1"

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)

CORE = H / (
    "mkmini_stage_td3_adapter_20261027_v1/"
    "td3_core.py"
)

HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_"
    "20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

CONTROL = H / (
    "mkmini_stage_td3_t9_control_20261027_v1/"
    "checkpoints/episode_0004.pt"
)

REGULARIZED = H / (
    "mkmini_stage_td3_t9_regularized_20261027_v1/"
    "checkpoints/episode_0004.pt"
)

EXPECTED = {
    STAGE:
    "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",

    CORE:
    "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",

    HELPER:
    "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",

    FROZEN:
    "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",

    CONTROL:
    "580b6aa2929fff3cb432d4ce1036bb03564d973d357e23e00cb3a3c9b75e63cf",

    REGULARIZED:
    "fdbcc174597d2980b1541734e2b5d832bd0dbb4c324f8528419a152e777b9f75",
}

CAP = 6000
WALL = 4800

# 全新开发种子，与T6/T9训练种子隔离。
# 每个任务仅有一个最终目标，没有规划器和路点。
TASKS = [
    [2029541101, -20, 0.4],
    [2029541101,  20, 0.4],
    [2029541101, -20, 0.8],
    [2029541101,  20, 0.8],
    [2029541102, -20, 0.4],
    [2029541102,  20, 0.4],
    [2029541102, -20, 0.8],
    [2029541102,  20, 0.8],
]

BRANCHES = (
    ("control", CONTROL),
    ("regularized", REGULARIZED),
)


def sha(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1048576),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(key, data):
    print(
        key,
        json.dumps(
            data,
            ensure_ascii=False,
            default=str
        ),
        flush=True
    )


def save_json(path, data):
    temporary = Path(str(path) + ".partial")

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str
        ) + "\n"
    )

    os.replace(temporary, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    require(
        spec is not None
        and spec.loader is not None,
        "MODULE_IMPORT_SPEC_INVALID"
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def preflight():
    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("EXPLORATION_NOISE=0")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("FROZEN_MODEL_WRITE=FALSE")

    for path, expected in EXPECTED.items():
        require(
            path.is_file(),
            "REQUIRED_FILE_MISSING:" + str(path)
        )

        require(
            sha(path) == expected,
            "SHA256_MISMATCH:" + str(path)
        )

    manifest = {
        "experiment": "TD3_T10_PAIRED_EVAL",
        "control_sha256": EXPECTED[CONTROL],
        "regularized_sha256": EXPECTED[REGULARIZED],
        "stage_sha256": EXPECTED[STAGE],
        "core_sha256": EXPECTED[CORE],
        "helper_sha256": EXPECTED[HELPER],
        "evaluation_script_sha256": sha(Path(__file__)),
        "tasks": TASKS,
        "cap": CAP,
        "exploration_noise": 0,
        "training": False,
        "automatic_promotion": False,
    }

    manifest_file = JOB / "manifest.json"

    if manifest_file.exists():
        require(
            json.loads(manifest_file.read_text())
            == manifest,
            "T10_MANIFEST_CHANGED"
        )
    else:
        save_json(manifest_file, manifest)

    emit("TD3_T10_PREFLIGHT=PASS", {
        "tasks": len(TASKS),
        "models": 2,
        "max_physics_frames":
            len(TASKS) * len(BRANCHES) * CAP,
        "equal_training_frames_required": True,
        "checkpoint_write": False
    })


def load_actors(core):
    actors = {}
    states = {}

    for name, path in BRANCHES:

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        require(
            state.get("algorithm") == "TD3",
            "ALGORITHM_MISMATCH:" + name
        )

        require(
            int(state["frames"]) == 3675,
            "TRAINING_FRAMES_MISMATCH:" + name
        )

        require(
            int(state["updates"]) == 1710,
            "TRAINING_UPDATES_MISMATCH:" + name
        )

        require(
            state["episode_index"] == 4,
            "TRAINING_EPISODE_MISMATCH:" + name
        )

        require(
            state["manifest"]["source_stage"]
            == EXPECTED[STAGE],
            "STAGE_SOURCE_MISMATCH:" + name
        )

        actor = core.Actor(112, 2)

        actor.load_state_dict(
            state["actor"]
        )

        actor.eval()

        actors[name] = actor
        states[name] = state

        emit("TD3_T10_ACTOR_LOADED=PASS", {
            "branch": name,
            "frames": state["frames"],
            "updates": state["updates"],
            "checkpoint_sha256": sha(path)
        })

    # 确认两个分支的回合任务序列一致。
    for a, b in zip(
        states["control"]["episodes"],
        states["regularized"]["episodes"]
    ):
        for key in (
            "seed",
            "side",
            "angle",
            "distance_m",
            "physics_frames"
        ):
            require(
                a[key] == b[key],
                "TRAINING_PAIR_MISMATCH:" + key
            )

    return actors


def evaluate_case(
    helper,
    make_base,
    v,
    native_mod,
    N,
    actor,
    task_index,
    branch,
    task,
    deadline
):
    requested_seed, angle, distance = task

    env = None
    renderer = None

    try:
        env, renderer, obs, actual_seed = (
            helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                requested_seed,
                angle,
                distance
            )
        )

        env.max_steps = CAP

        obs = np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)

        require(
            obs.shape == (112,)
            and np.isfinite(obs).all(),
            "INVALID_INITIAL_OBSERVATION"
        )

        initial = float(
            env._goal_state()[1]
        )

        require(
            abs(initial - distance) < 0.03,
            "GOAL_DISTANCE_INVALID"
        )

        start_xy = np.asarray(
            env.data.qpos[:2],
            dtype=np.float64
        ).tolist()

        goal_xy = np.asarray(
            env.goal_xy,
            dtype=np.float64
        ).tolist()

        closest = initial

        steps = 0
        success = False
        collision = False

        steer_total = 0.0
        throttle_total = 0.0

        positive = 0
        negative = 0

        end_reason = "CAP"

        while steps < CAP:

            if time.monotonic() >= deadline:
                emit("TD3_T10_CASE_PAUSED=", {
                    "task": task_index + 1,
                    "branch": branch,
                    "steps_not_committed": steps
                })

                return None

            # 确定性Actor，无探索噪声，无梯度更新。
            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, -1)
                    )
                ).cpu().numpy()[0]

            require(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "NONFINITE_ACTOR_ACTION"
            )

            raw = np.clip(
                raw,
                -1.0,
                1.0
            )

            action = np.asarray(
                [
                    (float(raw[0]) + 1.0) * 0.5,
                    float(raw[1])
                ],
                dtype=np.float32
            )

            before = float(env.data.time)

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info
            ) = N.step_checked(
                env,
                action
            )

            dt = float(
                env.model.opt.timestep
            )

            nframe = int(round(
                (
                    float(env.data.time)
                    - before
                ) / dt
            ))

            require(
                nframe == 1,
                "PHYSICS_FRAME_COUNT_INVALID"
            )

            obs = np.asarray(
                next_obs,
                dtype=np.float32
            ).reshape(-1)

            require(
                obs.shape == (112,)
                and np.isfinite(obs).all()
                and math.isfinite(float(reward)),
                "INVALID_MUJOCO_TRANSITION"
            )

            steps += 1

            current = float(
                env._goal_state()[1]
            )

            closest = min(
                closest,
                current
            )

            steer_total += float(action[1])
            throttle_total += float(action[0])

            positive += int(
                action[1] > 0.25
            )

            negative += int(
                action[1] < -0.25
            )

            collision = bool(
                info.get("collision", False)
            )

            success = bool(
                info.get("reach_goal", False)
            )

            if (
                success
                or collision
                or terminated
                or truncated
            ):

                end_reason = (
                    "GOAL"
                    if success
                    else "CONTACT"
                    if collision
                    else "TERMINATED"
                    if terminated
                    else "TRUNCATED"
                )

                break

            if steps % 2000 == 0:
                emit("TD3_T10_PROGRESS=", {
                    "task": task_index + 1,
                    "branch": branch,
                    "frames": steps,
                    "remaining_m": round(
                        current,
                        5
                    ),
                    "closest_m": round(
                        closest,
                        5
                    )
                })

        row = {
            "task": task_index + 1,
            "branch": branch,
            "seed_requested": requested_seed,
            "seed_actual": actual_seed,
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start_xy,
            "goal_xy": goal_xy,
            "initial_m": initial,
            "closest_m": closest,
            "remaining_m": float(
                env._goal_state()[1]
            ),
            "frames": steps,
            "end": end_reason,
            "success": bool(success),
            "collision": bool(collision),
            "mean_throttle":
                throttle_total / steps,
            "mean_steer":
                steer_total / steps,
            "positive_steer_fraction":
                positive / steps,
            "negative_steer_fraction":
                negative / steps
        }

        emit("TD3_T10_CASE=", row)

        return row

    finally:

        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()


def paired_check(control, regularized):
    require(
        control["task"] == regularized["task"],
        "PAIR_TASK_MISMATCH"
    )

    require(
        control["seed_actual"]
        == regularized["seed_actual"],
        "PAIR_ACTUAL_SEED_MISMATCH"
    )

    require(
        control["angle_deg"]
        == regularized["angle_deg"],
        "PAIR_ANGLE_MISMATCH"
    )

    require(
        control["distance_m"]
        == regularized["distance_m"],
        "PAIR_DISTANCE_MISMATCH"
    )

    for key in ("start_xy", "goal_xy"):
        a = np.asarray(
            control[key],
            dtype=np.float64
        )

        b = np.asarray(
            regularized[key],
            dtype=np.float64
        )

        require(
            np.allclose(
                a,
                b,
                atol=1e-7,
                rtol=0
            ),
            "PAIR_INITIAL_STATE_MISMATCH:"
            + key
        )


def summarize(results):
    report = {}

    for name, _ in BRANCHES:

        rows = [
            x for x in results
            if x["branch"] == name
        ]

        report[name] = {
            "completed": len(rows),
            "success": sum(
                int(x["success"])
                for x in rows
            ),
            "collisions": sum(
                int(x["collision"])
                for x in rows
            ),
            "left_success": sum(
                int(x["success"])
                for x in rows
                if x["angle_deg"] < 0
            ),
            "right_success": sum(
                int(x["success"])
                for x in rows
                if x["angle_deg"] > 0
            ),
            "mean_closest_m": (
                float(np.mean([
                    x["closest_m"]
                    for x in rows
                ]))
                if rows else None
            )
        }

    pairs = []

    for index in range(len(results) // 2):
        a = results[index * 2]
        b = results[index * 2 + 1]

        paired_check(a, b)

        pairs.append({
            "task": index + 1,
            "angle_deg": a["angle_deg"],
            "distance_m": a["distance_m"],
            "control_success": a["success"],
            "regularized_success": b["success"],
            "control_collision": a["collision"],
            "regularized_collision": b["collision"],
            "closest_improvement_m":
                a["closest_m"] - b["closest_m"]
        })

    return report, pairs


def main():
    JOB.mkdir(
        parents=True,
        exist_ok=True
    )

    preflight()

    torch.set_num_threads(2)

    core = load_module(
        CORE,
        "td3_t10_core"
    )

    actors = load_actors(core)

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER,
        "td3_t10_helper"
    )

    legacy = helper.load_module(
        STAGE,
        "td3_t10_stage"
    )

    for path, expected in legacy.EXPECTED.items():

        require(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_INVALID:"
            + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "td3_t10_native_runner"
    )

    _, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    require(
        isinstance(hashes, dict)
        and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_FAILED"
    )

    N = prep[3]

    require(
        callable(N.step_checked),
        "NATIVE_STEP_UNAVAILABLE"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    results_file = JOB / "results.json"

    if results_file.is_file():
        results = json.loads(
            results_file.read_text()
        )
    else:
        results = []

    require(
        isinstance(results, list)
        and len(results) <= len(TASKS) * 2,
        "RESULTS_RESUME_INVALID"
    )

    # 断点恢复时校验已完成任务的顺序。
    for index, row in enumerate(results):

        task_index = index // 2
        branch_index = index % 2

        seed, angle, dist = TASKS[task_index]

        require(
            row["task"] == task_index + 1
            and row["branch"]
                == BRANCHES[branch_index][0]
            and row["seed_requested"] == seed
            and row["angle_deg"] == angle
            and row["distance_m"] == dist,
            "RESUME_TASK_SEQUENCE_MISMATCH"
        )

    # 已完成的每一对必须具有相同初始条件。
    summarize(results)

    deadline = time.monotonic() + WALL

    total = len(TASKS) * len(BRANCHES)

    for index in range(
        len(results),
        total
    ):

        task_index = index // 2
        branch_index = index % 2

        name = BRANCHES[branch_index][0]

        row = evaluate_case(
            helper,
            make_base,
            v,
            native_mod,
            N,
            actors[name],
            task_index,
            name,
            TASKS[task_index],
            deadline
        )

        if row is None:
            break

        if branch_index == 1:

            paired_check(
                results[-1],
                row
            )

        results.append(row)

        save_json(
            results_file,
            results
        )

        summary, pairs = summarize(
            results
        )

        completed = (
            len(results) == total
        )

        report = {
            "state": (
                "EVALUATED"
                if completed
                else "PAUSED_RESUMABLE"
            ),
            "completed_runs": len(results),
            "total_runs": total,
            "completed_pairs": len(pairs),
            "control": summary["control"],
            "regularized": summary["regularized"],
            "training": False,
            "noise": 0.0,
            "equal_training_frames": 3675,
            "model_promoted": False
        }

        save_json(
            JOB / "status.json",
            report
        )

        emit(
            "TD3_T10_STATUS=",
            report
        )

    summary, pairs = summarize(results)

    final = {
        "state": (
            "EVALUATED"
            if len(results) == total
            else "PAUSED_RESUMABLE"
        ),
        "completed_runs": len(results),
        "completed_pairs": len(pairs),
        "total_pairs": len(TASKS),
        "control": summary["control"],
        "regularized": summary["regularized"],
        "training": False,
        "exploration_noise": 0.0,
        "model_promoted": False,
        "scientific_final_acceptance": False
    }

    save_json(
        JOB / "status.json",
        final
    )

    save_json(
        JOB / "pairs.json",
        pairs
    )

    emit("TD3_T10_FINAL=", final)

    require(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        emit(
            "TD3_T10_BLOCKED=",
            {"error": repr(error)}
        )

        traceback.print_exc()

        raise

    finally:

        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
