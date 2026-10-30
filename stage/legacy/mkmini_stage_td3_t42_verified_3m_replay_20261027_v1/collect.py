from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t42_verified_3m_replay_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T36 = H / "mkmini_stage_td3_t36_random3m_screen_20261027_v1"

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

TRAIN = (
    H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"
    / "t31a/train.py"
)

HASHES = {
    CORE:
        "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    CKPT:
        "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
}

CASES = [
    {
        "task": 1,
        "seed": 2029653601,
        "angle": -17.3,
        "side": "LEFT",
        "frames": 10612,
    },
    {
        "task": 4,
        "seed": 2029653602,
        "angle": 13.6,
        "side": "RIGHT",
        "frames": 9707,
    },
]

ARCHIVE = JOB / "verified_3m_success.npz"
MANIFEST = JOB / "manifest.json"


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    require(
        spec is not None and spec.loader is not None,
        "MODULE_LOAD_INVALID:" + name
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def reference_for(case):
    path = (
        T36
        / f"task_{case['task']:02d}_T16.json"
    )

    require(
        path.is_file(),
        "T36_REFERENCE_MISSING:" + str(path)
    )

    data = json.loads(path.read_text())

    require(
        data.get("checkpoint_sha256") == HASHES[CKPT]
        and data.get("requested_seed") == case["seed"]
        and data.get("actual_seed") == case["seed"]
        and data.get("angle_deg") == case["angle"]
        and data.get("distance_m") == 3.0
        and data.get("frames") == case["frames"]
        and data.get("success") is True
        and data.get("collision") is False
        and data.get("end") == "GOAL",
        "T36_REFERENCE_NOT_SUCCESS:"
        + str(case["task"])
    )

    return data


def collect_case(
    case,
    reference,
    actor,
    helper,
    make_base,
    v,
    native_mod,
    N,
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
                case["seed"],
                case["angle"],
                3.0
            )
        )

        env.max_steps = 14000

        start = np.asarray(
            env.data.qpos[:2], dtype=float
        ).copy()

        goal = np.asarray(
            env.goal_xy, dtype=float
        ).copy()

        require(
            int(actual_seed) == case["seed"],
            "ACTUAL_SEED_MISMATCH"
        )

        require(
            np.allclose(
                start,
                reference["start_xy"],
                atol=1e-7,
                rtol=0
            ),
            "START_MISMATCH"
        )

        require(
            np.allclose(
                goal,
                reference["goal_xy"],
                atol=1e-7,
                rtol=0
            ),
            "GOAL_MISMATCH"
        )

        require(
            abs(float(env._goal_state()[1]) - 3.0) < 0.01,
            "INITIAL_DISTANCE_MISMATCH"
        )

        obs = np.asarray(
            obs, dtype=np.float32
        ).reshape(-1)

        require(
            obs.shape == (112,),
            "INITIAL_OBSERVATION_INVALID"
        )

        rows = []

        final_success = False
        final_collision = False
        final_distance = 3.0
        end = "CAP"

        for frame in range(1, 14001):
            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, 112)
                    )
                ).cpu().numpy()[0]

            require(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "ACTOR_ACTION_INVALID"
            )

            action = np.array(
                [
                    np.clip(
                        (float(raw[0]) + 1) / 2,
                        0, 1
                    ),
                    np.clip(
                        float(raw[1]), -1, 1
                    ),
                ],
                dtype=np.float32
            )

            next_obs, reward, terminated, truncated, info = (
                N.step_checked(env, action)
            )

            next_obs = np.asarray(
                next_obs, dtype=np.float32
            ).reshape(-1)

            require(
                next_obs.shape == (112,)
                and np.isfinite(next_obs).all()
                and np.isfinite(float(reward)),
                "STEP_DATA_INVALID"
            )

            final_success = bool(
                info.get("reach_goal", False)
            )

            final_collision = bool(
                info.get("collision", False)
            )

            done = bool(
                final_success
                or final_collision
                or terminated
                or truncated
            )

            rows.append(
                (
                    obs.copy(),
                    raw.copy(),
                    float(reward),
                    next_obs.copy(),
                    float(done),
                    case["side"],
                )
            )

            obs = next_obs

            final_distance = float(
                env._goal_state()[1]
            )

            if frame % 2000 == 0:
                print(
                    "T42_PROGRESS",
                    "TASK=", case["task"],
                    "FRAME=", frame,
                    "DISTANCE_M=",
                    round(final_distance, 4),
                    flush=True
                )

            if done:
                end = (
                    "GOAL" if final_success
                    else "CONTACT" if final_collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        require(
            final_success
            and not final_collision
            and end == "GOAL"
            and frame == reference["frames"]
            and abs(final_distance - reference["remaining_m"])
                < 1e-4,
            "T42_SUCCESS_REPRODUCTION_FAILED:"
            + str(case["task"])
        )

        result = {
            "task": case["task"],
            "seed": case["seed"],
            "angle_deg": case["angle"],
            "side": case["side"],
            "distance_m": 3.0,
            "frames": frame,
            "success": final_success,
            "collision": final_collision,
            "end": end,
            "remaining_m": final_distance,
            "start_xy": start.tolist(),
            "goal_xy": goal.tolist(),
        }

        print(
            "T42_SUCCESS_REPRODUCED=",
            json.dumps(result, ensure_ascii=False),
            flush=True
        )

        return rows, result

    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main():
    print("T42_STAGE_ONLY=TRUE")
    print("T42_NATIVE_MUJOCO=TRUE")
    print("T42_TRAINING=FALSE")
    print("T42_PLANNER=FALSE")
    print("T42_ACTION_OVERRIDE=FALSE")
    print("T42_MODEL_PROMOTED=FALSE")

    for path, expected in HASHES.items():
        require(
            path.is_file() and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    provenance = T36 / "provenance.json"
    evaluator = T36 / "evaluate.py"

    require(
        provenance.is_file()
        and evaluator.is_file(),
        "T36_SOURCE_MISSING"
    )

    previous = json.loads(provenance.read_text())

    require(
        sha(evaluator) == previous["generated_sha256"],
        "T36_EVALUATOR_CHANGED"
    )

    references = {
        item["task"]: reference_for(item)
        for item in CASES
    }

    if ARCHIVE.exists() or MANIFEST.exists():
        require(
            ARCHIVE.exists() and MANIFEST.exists(),
            "PARTIAL_ARCHIVE"
        )

        saved = json.loads(MANIFEST.read_text())

        require(
            saved.get("experiment")
            == "STAGE_TD3_T42"
            and saved.get("parent_sha256") == HASHES[CKPT]
            and saved.get("archive_sha256") == sha(ARCHIVE)
            and saved.get("verified_successes") == 2
            and saved.get("total_transitions") == 20319,
            "EXISTING_ARCHIVE_INVALID"
        )

        with np.load(ARCHIVE) as data:
            require(
                data["states"].shape == (20319, 112)
                and data["raw_actions"].shape == (20319, 2)
                and data["next_states"].shape == (20319, 112)
                and np.array_equal(
                    data["episode_offsets"],
                    np.array([0, 10612, 20319])
                ),
                "EXISTING_ARCHIVE_SHAPE_INVALID"
            )

        print("T42_ARCHIVE=EXISTING_VERIFIED")
        print("T42_CAPTURE=PASS")
        return

    torch.set_num_threads(2)

    core = load(CORE, "t42_stage_core")

    state = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    require(
        state.get("algorithm") == "TD3"
        and state.get("frames") == 13491,
        "PARENT_CHECKPOINT_INVALID"
    )

    actor = core.Actor(112, 2)

    actor.load_state_dict(
        state["actor"], strict=True
    )

    actor.eval()

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(
        HELPER, "t42_stage_helper"
    )

    legacy = load(
        STAGE, "t42_stage_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        require(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC, "t42_stage_native"
    )

    _, v, native_mod, prep, _ = (
        runner.load_native()
    )

    require(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    all_rows = []
    episodes = []
    offsets = [0]

    for case in CASES:
        rows, result = collect_case(
            case,
            references[case["task"]],
            actor,
            helper,
            make_base,
            v,
            native_mod,
            N,
        )

        all_rows.extend(rows)
        episodes.append(result)
        offsets.append(len(all_rows))

    require(
        len(all_rows) == 20319
        and offsets == [0, 10612, 20319],
        "TRANSITION_TOTAL_MISMATCH"
    )

    states = np.stack(
        [r[0] for r in all_rows]
    ).astype(np.float32)

    actions = np.stack(
        [r[1] for r in all_rows]
    ).astype(np.float32)

    rewards = np.asarray(
        [r[2] for r in all_rows],
        dtype=np.float32
    )

    next_states = np.stack(
        [r[3] for r in all_rows]
    ).astype(np.float32)

    dones = np.asarray(
        [r[4] for r in all_rows],
        dtype=np.float32
    )

    sides = np.asarray(
        [r[5] for r in all_rows]
    )

    require(
        np.isfinite(states).all()
        and np.isfinite(actions).all()
        and np.isfinite(rewards).all()
        and np.isfinite(next_states).all()
        and int(dones.sum()) == 2,
        "RECORDED_TRANSITIONS_INVALID"
    )

    temporary = JOB / "verified_success.partial"

    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            states=states,
            raw_actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            sides=sides,
            episode_offsets=np.asarray(
                offsets, dtype=np.int64
            ),
        )

    os.replace(temporary, ARCHIVE)

    manifest = {
        "experiment": "STAGE_TD3_T42",
        "parent_sha256": HASHES[CKPT],
        "evaluator_sha256": sha(evaluator),
        "archive_sha256": sha(ARCHIVE),
        "verified_successes": 2,
        "total_transitions": len(all_rows),
        "episode_offsets": offsets,
        "episodes": episodes,
        "training": False,
        "raw_actor_actions": True,
        "reward_semantics_review_required": True,
        "old_validation_scenes_now_training_eligible": True,
        "model_promoted": False,
    }

    temp_manifest = JOB / "manifest.partial"

    temp_manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(temp_manifest, MANIFEST)

    print("T42_ARCHIVE_SHA256=", sha(ARCHIVE))
    print("T42_SUCCESS_EPISODES=2")
    print("T42_TRANSITIONS=", len(all_rows))
    print("T42_CAPTURE=PASS")

    # 检查现有训练器的奖励及终止标记处理代码。
    # 当前文件先作为轨迹数据归档，不自动送入TD3更新。
    if TRAIN.is_file():
        lines = TRAIN.read_text().splitlines()

        print("\nT42_TRAINER_TRANSITION_CONTEXT=")

        for i, line in enumerate(lines):
            if "replay.append(" in line:
                for j in range(
                    max(0, i - 36),
                    min(len(lines), i + 18)
                ):
                    print(
                        f"T42_TRAIN_SRC_{j+1:04d}: "
                        + lines[j]
                    )
                break

    require(
        sha(CKPT) == HASHES[CKPT],
        "FROZEN_PARENT_CHANGED"
    )

    print("T42_PARENT_FROZEN=TRUE")


try:
    main()

except Exception as error:
    print("T42_CAPTURE=BLOCKED", repr(error))
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
