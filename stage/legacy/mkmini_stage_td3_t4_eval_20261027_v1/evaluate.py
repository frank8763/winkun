from pathlib import Path
import fcntl
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
JOB = H / "mkmini_stage_td3_t4_eval_20261027_v1"

T3 = H / "mkmini_stage_td3_t3_20261027_v1"
CK = T3 / "latest.pt"
STATUS = T3 / "status.json"

T1 = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

CORE = H / (
    "mkmini_stage_td3_adapter_20261027_v1/td3_core.py"
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

EXPECTED = {
    CK:
        "43e228b52715bd20cbbfdea339b30502f36fbaf985340c90a778e374f62ee041",
    CORE:
        "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
}

CAP = 12000
WALL = 6300

# 全新开发诊断任务，与TD3训练种子隔离。
# 每组测试最终目标，绝不提供人工路点。
TASKS = [
    (2029513101, -20, 0.4),
    (2029513101,  20, 0.4),
    (2029513101, -20, 0.8),
    (2029513101,  20, 0.8),
    (2029513102, -20, 0.4),
    (2029513102,  20, 0.4),
    (2029513102, -20, 0.8),
    (2029513102,  20, 0.8)
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for buf in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(buf)

    return h.hexdigest()


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(name, data):
    print(
        name,
        json.dumps(data, ensure_ascii=False, default=str),
        flush=True
    )


def save(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str
        ) + "\n"
    )

    os.replace(tmp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )

    need(
        spec is not None
        and spec.loader is not None,
        "IMPORT_SPEC_INVALID"
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def preflight():
    for path, expected in EXPECTED.items():
        need(
            path.is_file() and sha(path) == expected,
            "SHA_MISMATCH:" + str(path)
        )

    need(
        STATUS.is_file() and T1.is_file(),
        "EVALUATION_DEPENDENCY_MISSING"
    )

    state = json.loads(STATUS.read_text())

    need(
        state.get("state") == "TRAINED_BUDGET"
        and state.get("additional_frames") == 24000
        and state.get("checkpoint_sha256") == EXPECTED[CK],
        "T3_TRAINING_NOT_VERIFIED"
    )

    import ast

    ast.parse(T1.read_text())
    ast.parse(STAGE.read_text())
    ast.parse(CORE.read_text())

    manifest = JOB / "manifest.json"

    details = {
        "algorithm": "TD3",
        "checkpoint_sha256": EXPECTED[CK],
        "core_sha256": EXPECTED[CORE],
        "stage_sha256": EXPECTED[STAGE],
        "helper_sha256": sha(T1),
        "tasks": TASKS,
        "max_steps_per_task": CAP,
        "exploration_noise": 0.0,
        "automatic_promotion": False
    }

    if manifest.exists():
        need(
            json.loads(manifest.read_text()) == details,
            "EVALUATION_MANIFEST_CHANGED"
        )
    else:
        save(manifest, details)

    emit("TD3_T4_PREFLIGHT=PASS", {
        "tasks": len(TASKS),
        "max_total_physics": len(TASKS) * CAP,
        "wall_seconds": WALL,
        "checkpoint_sha256": EXPECTED[CK],
        "noise": 0.0,
        "training": False
    })


def run_case(
    helper, make_base, v, native_mod, N,
    agent, index, task, deadline
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

        # 训练时单回合上限1500；
        # 这里统一使用更长的独立评价上限。
        env.max_steps = CAP

        initial = float(env._goal_state()[1])

        need(
            abs(initial - distance) < 0.03,
            "EVALUATION_GOAL_MISMATCH"
        )

        obs = np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)

        need(
            obs.shape == (112,),
            "EVALUATION_OBSERVATION_DIM_INVALID"
        )

        start_xy = np.asarray(
            env.data.qpos[:2],
            dtype=float
        ).tolist()

        minimum = initial
        steps = 0

        success = False
        collision = False
        end_reason = "CAP"

        throttle_sum = 0.0
        steer_sum = 0.0
        positive_steer = 0
        negative_steer = 0

        while steps < CAP:

            if time.monotonic() >= deadline:
                emit("TD3_T4_CASE_PAUSED=", {
                    "case": index + 1,
                    "steps_not_committed": steps
                })
                return None

            # 严格确定性评估：
            # 只使用Actor输出，不增加探索噪声。
            raw = np.asarray(
                agent.act(obs),
                dtype=np.float32
            ).reshape(-1)

            need(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "EVALUATION_ACTION_INVALID"
            )

            raw = np.clip(raw, -1.0, 1.0)

            action = np.array(
                [
                    (float(raw[0]) + 1.0) * 0.5,
                    float(raw[1])
                ],
                dtype=np.float32
            )

            throttle_sum += float(action[0])
            steer_sum += float(action[1])

            positive_steer += int(action[1] > 0.05)
            negative_steer += int(action[1] < -0.05)

            before_time = float(env.data.time)

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info
            ) = N.step_checked(env, action)

            nframe = int(round(
                (
                    float(env.data.time)
                    - before_time
                ) / float(env.model.opt.timestep)
            ))

            need(
                nframe == 1,
                "EVALUATION_FRAME_DELTA_INVALID"
            )

            obs = np.asarray(
                next_obs,
                dtype=np.float32
            ).reshape(-1)

            need(
                obs.shape == (112,)
                and np.isfinite(obs).all()
                and math.isfinite(float(reward)),
                "EVALUATION_TRANSITION_INVALID"
            )

            steps += nframe

            current = float(
                env._goal_state()[1]
            )

            minimum = min(
                minimum,
                current
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
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TRUNCATED"
                )
                break

            if steps % 2000 == 0:
                emit("TD3_T4_PROGRESS=", {
                    "case": index + 1,
                    "steps": steps,
                    "remaining_m": round(current, 5),
                    "closest_m": round(minimum, 5)
                })

        result = {
            "case": index + 1,
            "seed_requested": seed,
            "seed_actual": actual_seed,
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start_xy,
            "initial_m": initial,
            "remaining_m": float(
                env._goal_state()[1]
            ),
            "closest_m": minimum,
            "steps": steps,
            "end": end_reason,
            "success": success,
            "collision": collision,
            "mean_throttle": throttle_sum / steps,
            "mean_steer": steer_sum / steps,
            "positive_steer_steps": positive_steer,
            "negative_steer_steps": negative_steer
        }

        emit("TD3_T4_CASE=", result)

        return result

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

    preflight()

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    torch.set_num_threads(2)
    torch.manual_seed(20261027)

    helper = load_module(
        T1,
        "stage_td3_t4_helper"
    )

    core = load_module(
        CORE,
        "stage_td3_t4_core"
    )

    legacy = helper.load_module(
        STAGE,
        "stage_td3_t4_bootstrap"
    )

    for path, expected in legacy.EXPECTED.items():
        need(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_MISMATCH"
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_td3_t4_native"
    )

    p2, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    need(
        isinstance(hashes, dict)
        and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_FAILED"
    )

    N = prep[3]

    need(
        callable(N.step_checked),
        "NATIVE_STEP_UNAVAILABLE"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    agent = core.TD3(
        state_dim=112,
        action_dim=2,
        device="cpu"
    )

    state = torch.load(
        CK,
        map_location="cpu",
        weights_only=False
    )

    need(
        state.get("algorithm") == "TD3",
        "CHECKPOINT_ALGORITHM_INVALID"
    )

    need(
        int(state.get("additional_frames", -1)) == 24000,
        "CHECKPOINT_FRAMES_INVALID"
    )

    agent.actor.load_state_dict(
        state["actor"]
    )

    agent.actor.eval()

    print(
        "TD3_T4_ACTOR_LOADED=PASS",
        flush=True
    )

    results_path = JOB / "results.json"
    results = []

    if results_path.is_file():
        results = json.loads(
            results_path.read_text()
        )

    need(
        isinstance(results, list)
        and len(results) <= len(TASKS),
        "EXISTING_RESULTS_INVALID"
    )

    for i, row in enumerate(results):
        seed, angle, dist = TASKS[i]

        need(
            row["case"] == i + 1
            and row["seed_requested"] == seed
            and row["angle_deg"] == angle
            and row["distance_m"] == dist,
            "EVALUATION_RESUME_TASK_MISMATCH"
        )

    deadline = time.monotonic() + WALL

    for i in range(
        len(results),
        len(TASKS)
    ):
        row = run_case(
            helper,
            make_base,
            v,
            native_mod,
            N,
            agent,
            i,
            TASKS[i],
            deadline
        )

        if row is None:
            break

        results.append(row)

        save(
            results_path,
            results
        )

        save(
            JOB / "status.json",
            {
                "state": "RUNNING",
                "completed": len(results),
                "total": len(TASKS),
                "success": sum(
                    int(x["success"])
                    for x in results
                ),
                "collisions": sum(
                    int(x["collision"])
                    for x in results
                ),
                "training": False
            }
        )

    finished = (
        len(results) == len(TASKS)
    )

    summary = {
        "state": (
            "EVALUATED"
            if finished
            else "PAUSED_RESUMABLE"
        ),
        "completed": len(results),
        "total": len(TASKS),
        "success": sum(
            int(x["success"])
            for x in results
        ),
        "collisions": sum(
            int(x["collision"])
            for x in results
        ),
        "left_success": sum(
            int(x["success"])
            for x in results
            if x["angle_deg"] < 0
        ),
        "right_success": sum(
            int(x["success"])
            for x in results
            if x["angle_deg"] > 0
        ),
        "checkpoint_sha256": sha(CK),
        "checkpoint_promoted": False,
        "scientific_paired_comparison": False,
        "exploration_noise": 0.0
    }

    save(
        JOB / "status.json",
        summary
    )

    emit(
        "TD3_T4_FINAL=",
        summary
    )

    need(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_PARENT_CHANGED"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        emit("TD3_T4_BLOCKED=", {
            "error": repr(error)
        })
        traceback.print_exc()
        raise

    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
