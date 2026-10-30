from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback

import numpy as np
import torch


T119_JOB = Path(os.environ["T119_JOB"])
EVAL_JOB = Path(os.environ["EVAL_JOB"])

CHECKPOINT = (
    T119_JOB / "checkpoints" / "episode_0341.pt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "4539dbfb627c2177a16249a3c2a300d4b52e5caf5fc2a5802dc35b939e25897f"
)

EXPECTED_TRAIN_SHA256 = (
    "24a3f5c4bcba89684aae9a0ea6ec748846bd547b4973139932ee955570639cf2"
)

MAX_PHYSICS_FRAMES = 14000
MAX_CONTROL_STEPS = 20000


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_module(path, name):
    path = Path(path)

    if not path.is_file():
        raise RuntimeError("SOURCE_MISSING: " + str(path))

    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("MODULE_SPEC_INVALID: " + str(path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def safe_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def get_distance(env):
    try:
        return float(env._goal_state()[1])
    except Exception:
        return float("nan")


def get_pose(env):
    try:
        q = np.asarray(env.data.qpos, dtype=np.float64).reshape(-1)

        x = float(q[0]) if q.size >= 2 else float("nan")
        y = float(q[1]) if q.size >= 2 else float("nan")

        yaw = float("nan")

        # MuJoCo free-joint quaternion order: w, x, y, z.
        if q.size >= 7:
            qw, qx, qy, qz = map(float, q[3:7])

            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )

        return x, y, yaw

    except Exception:
        return float("nan"), float("nan"), float("nan")


def main():
    print("=" * 82)
    print("STAGE T140 — FROZEN POLICY DIRECTIONAL DIAGNOSIS")
    print("STAGE_ONLY=TRUE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("TRAINING=FALSE")
    print("ACTOR_UPDATE=FALSE")
    print("CHECKPOINT_OVERWRITE=FALSE")
    print("AUTOMATIC_PROMOTION=FALSE")
    print("PHASE_TOUCHED=FALSE")
    print("=" * 82)

    # ---------------------------------------------------------
    # 1. Checkpoint and training-state gates.
    # ---------------------------------------------------------
    status_path = T119_JOB / "status.json"
    manifest_path = T119_JOB / "manifest.json"
    train_path = T119_JOB / "train.py"
    core_path = T119_JOB / "td3_core_goal_contrast.py"

    for p in (
        status_path,
        manifest_path,
        train_path,
        core_path,
        CHECKPOINT,
    ):
        if not p.is_file():
            raise RuntimeError("REQUIRED_FILE_MISSING: " + str(p))

    train_hash = sha256(train_path)
    checkpoint_hash = sha256(CHECKPOINT)

    print("TRAIN_SHA256 =", train_hash)
    print("CHECKPOINT_SHA256 =", checkpoint_hash)

    if train_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("T119_TRAIN_SOURCE_HASH_MISMATCH")

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("T119_CHECKPOINT_HASH_MISMATCH")

    status = json.loads(status_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    print("TRAINING_STATE =", status.get("state"))
    print("FRAMES =", status.get("frames"))
    print("CHECKPOINT =", status.get("checkpoint"))

    if status.get("state") != "PAUSED_WALL_BUDGET":
        raise RuntimeError("UNEXPECTED_T119_TRAINING_STATE")

    if status.get("checkpoint") != "episode_0341.pt":
        raise RuntimeError("STATUS_CHECKPOINT_MISMATCH")

    if status.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("STATUS_CHECKPOINT_HASH_MISMATCH")

    # Confirm T119 is not still training.
    ps = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )

    active = [
        line.strip()
        for line in ps.stdout.splitlines()
        if str(train_path) in line
        and "python" in line.lower()
        and "grep" not in line.lower()
    ]

    print("ACTIVE_T119_PROCESSES =", active)

    if active:
        raise RuntimeError("T119_TRAINING_PROCESS_STILL_ACTIVE")

    # ---------------------------------------------------------
    # 2. Prepare a separate output directory.
    #    Never write into the T119 checkpoint directory.
    # ---------------------------------------------------------
    EVAL_JOB.mkdir(parents=True, exist_ok=True)

    os.environ["TD3_JOB"] = str(EVAL_JOB)
    os.environ["C98_JOB"] = str(EVAL_JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    # ---------------------------------------------------------
    # 3. Verify the exact T119 source dependencies.
    # ---------------------------------------------------------
    train_mod = load_module(
        train_path,
        "stage_t140_t119_definitions",
    )

    core_path = Path(train_mod.CORE)
    helper_path = Path(train_mod.HELPER)
    stage_path = Path(train_mod.STAGE)
    frozen_path = Path(train_mod.FROZEN)

    expected_manifest = {
        "source_core": core_path,
        "source_stage": stage_path,
        "source_helper": helper_path,
        "frozen_reference": frozen_path,
    }

    for key, path in expected_manifest.items():
        expected = manifest.get(key)

        if not path.is_file():
            raise RuntimeError(
                "DEPENDENCY_MISSING: " + key + ": " + str(path)
            )

        actual = sha256(path)

        print(
            "DEPENDENCY",
            key,
            "SHA256=" + actual,
            "MATCH=" + str(actual == expected),
        )

        if not expected or actual != expected:
            raise RuntimeError("MANIFEST_HASH_MISMATCH: " + key)

    helper = train_mod.load_module(
        helper_path,
        "stage_t140_goal_helper",
    )

    core = train_mod.load_module(
        core_path,
        "stage_t140_t119_core",
    )

    legacy = train_mod.load_module(
        stage_path,
        "stage_t140_legacy_stage_environment",
    )

    for path, expected in legacy.EXPECTED.items():
        if not Path(path).is_file():
            raise RuntimeError(
                "NATIVE_DEPENDENCY_MISSING: " + str(path)
            )

        actual = sha256(path)

        if actual != expected:
            raise RuntimeError(
                "NATIVE_DEPENDENCY_HASH_MISMATCH: " + str(path)
            )

    print("T119_DEPENDENCY_GATES=PASS")

    # ---------------------------------------------------------
    # 4. Load the real native Stage environment interface.
    #
    # This is the corrected T139-discovered call chain:
    # legacy.module(...) -> runner.load_native()
    # -> legacy.make_short_task(...)
    # -> helper.make_goal_task(...)
    # ---------------------------------------------------------
    runner = legacy.module(
        legacy.SRC,
        "stage_t140_native_runner",
    )

    p2, v, native_mod, prep, hashes = runner.load_native()

    if not isinstance(hashes, dict):
        raise RuntimeError("NATIVE_SOURCE_MANIFEST_INVALID")

    if len(prep) < 4:
        raise RuntimeError("NATIVE_PREP_INCOMPLETE")

    N = prep[3]

    if not callable(getattr(N, "step_checked", None)):
        raise RuntimeError("NATIVE_STEP_CHECKED_UNAVAILABLE")

    make_base = legacy.make_short_task(
        runner.make_task
    )

    print("NATIVE_STAGE_BOOTSTRAP=PASS")
    print("NATIVE_STEP_CHECKED=PASS")

    # ---------------------------------------------------------
    # 5. Load only the frozen actor.
    #    No optimizer state is restored or stepped.
    # ---------------------------------------------------------
    agent = train_mod.create_agent(core)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "actor" not in checkpoint:
        raise RuntimeError("CHECKPOINT_ACTOR_MISSING")

    if not hasattr(agent, "actor"):
        raise RuntimeError("T119_ACTOR_INTERFACE_MISSING")

    incompatible = agent.actor.load_state_dict(
        checkpoint["actor"],
        strict=True,
    )

    agent.actor.eval()

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("ACTOR_STATE_DICT_INCOMPATIBLE")

    torch.set_num_threads(2)

    print("FROZEN_ACTOR_LOAD=PASS")
    print("ACTOR_UPDATE=FALSE")
    print("CHECKPOINT_OVERWRITE=FALSE")

    # Same random scene seed, opposite goal-angle signs.
    # The actual initialization seed is recorded for each case.
    cases = [
        ("PAIR_3601_NEG", 2029653601, -17.3),
        ("PAIR_3601_POS", 2029653601,  17.3),

        ("PAIR_3602_NEG", 2029653602, -13.6),
        ("PAIR_3602_POS", 2029653602,  13.6),

        ("PAIR_4801_NEG", 2029654801, -12.5),
        ("PAIR_4801_POS", 2029654801,  12.5),

        ("PAIR_4802_NEG", 2029654802, -18.8),
        ("PAIR_4802_POS", 2029654802,  18.8),
    ]

    results = []

    print("=" * 82)
    print("ALL_PRE_SIMULATION_GATES=PASS")
    print("SIMULATION_STARTED=TRUE")
    print("SIMULATION_MODE=FROZEN_POLICY_ONLY")
    print("=" * 82)

    for label, seed, angle in cases:
        print(
            "\nCASE_START",
            label,
            "REQUESTED_SEED=", seed,
            "GOAL_ANGLE_DEG=", angle,
            flush=True,
        )

        env = None
        renderer = None
        result = {
            "label": label,
            "pair": str(seed),
            "requested_seed": seed,
            "angle_deg": angle,
        }

        try:
            env, renderer, obs, actual_seed = (
                helper.make_goal_task(
                    make_base,
                    v,
                    native_mod,
                    N,
                    seed,
                    angle,
                    3.0,
                )
            )

            obs = np.asarray(obs, dtype=np.float32)

            if obs.shape != (112,):
                raise RuntimeError(
                    "OBSERVATION_SHAPE_MISMATCH: "
                    + str(obs.shape)
                )

            result["actual_seed"] = int(actual_seed)
            result["initial_distance_m"] = safe_float(
                get_distance(env)
            )

            trace_path = EVAL_JOB / (label + ".csv")

            columns = [
                "control_step",
                "physics_frames",
                "sim_time_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "raw_throttle",
                "raw_steer",
                "mapped_throttle",
                "mapped_steer",
                "distance_before_m",
                "distance_after_m",
                "obs_goal_distance_norm",
                "goal_angle_sin",
                "goal_angle_cos",
                "reward",
                "collision",
                "reach_goal",
                "terminated",
                "truncated",
            ]

            physics_frames = 0
            positive_steer = 0
            negative_steer = 0
            wrong_sign_steps = 0
            nonzero_steer_steps = 0
            steer_values = []

            collision = False
            reached = False
            end = "STEP_GUARD"

            with trace_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=columns,
                )
                writer.writeheader()

                with torch.no_grad():
                    for control_step in range(
                        1, MAX_CONTROL_STEPS + 1
                    ):
                        distance_before = get_distance(env)

                        raw = np.asarray(
                            agent.act(obs),
                            dtype=np.float32,
                        ).reshape(-1)

                        if raw.size != 2:
                            raise RuntimeError(
                                "ACTOR_ACTION_SHAPE_INVALID"
                            )

                        raw = np.clip(raw, -1.0, 1.0)

                        # Exact T119 action mapping:
                        # raw throttle [-1,1] -> [0,1]
                        # steer remains [-1,1].
                        action = np.asarray(
                            [
                                (float(raw[0]) + 1.0) / 2.0,
                                float(raw[1]),
                            ],
                            dtype=np.float32,
                        )

                        mapped_throttle = float(action[0])
                        mapped_steer = float(action[1])

                        positive_steer += int(
                            mapped_steer > 0.0
                        )
                        negative_steer += int(
                            mapped_steer < 0.0
                        )

                        if abs(mapped_steer) > 1e-5:
                            nonzero_steer_steps += 1

                            if mapped_steer * angle < 0:
                                wrong_sign_steps += 1

                        steer_values.append(mapped_steer)

                        before_time = float(env.data.time)

                        (
                            next_obs,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = N.step_checked(env, action)

                        after_time = float(env.data.time)

                        dt = float(env.model.opt.timestep)

                        nframe = max(
                            0,
                            int(round(
                                (after_time - before_time)
                                / max(dt, 1e-12)
                            )),
                        )

                        physics_frames += nframe

                        distance_after = get_distance(env)

                        x, y, yaw = get_pose(env)

                        collision = bool(
                            info.get("collision", False)
                        )

                        reached = bool(
                            info.get("reach_goal", False)
                        )

                        row = {
                            "control_step": control_step,
                            "physics_frames": physics_frames,
                            "sim_time_s": after_time,
                            "x_m": x,
                            "y_m": y,
                            "yaw_rad": yaw,
                            "raw_throttle": float(raw[0]),
                            "raw_steer": float(raw[1]),
                            "mapped_throttle": mapped_throttle,
                            "mapped_steer": mapped_steer,
                            "distance_before_m": distance_before,
                            "distance_after_m": distance_after,
                            "obs_goal_distance_norm": (
                                float(obs[48])
                                if obs.size > 48 else float("nan")
                            ),
                            "goal_angle_sin": (
                                float(obs[49])
                                if obs.size > 49 else float("nan")
                            ),
                            "goal_angle_cos": (
                                float(obs[50])
                                if obs.size > 50 else float("nan")
                            ),
                            "reward": float(reward),
                            "collision": collision,
                            "reach_goal": reached,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }

                        writer.writerow(row)

                        if control_step % 100 == 0:
                            f.flush()

                        obs = np.asarray(
                            next_obs,
                            dtype=np.float32,
                        )

                        if collision:
                            end = "CONTACT"
                            break

                        if reached:
                            end = "GOAL"
                            break

                        if terminated:
                            end = "TERMINATED"
                            break

                        if truncated:
                            end = "TRUNCATED"
                            break

                        if physics_frames >= MAX_PHYSICS_FRAMES:
                            end = "PHYSICS_FRAME_CAP"
                            break

                f.flush()

            result.update({
                "status": "EVALUATED",
                "end": end,
                "physics_frames": physics_frames,
                "success": bool(reached and not collision),
                "collision": collision,
                "final_distance_m": safe_float(
                    get_distance(env)
                ),
                "positive_steer_steps": positive_steer,
                "negative_steer_steps": negative_steer,
                "wrong_sign_fraction": (
                    wrong_sign_steps / nonzero_steer_steps
                    if nonzero_steer_steps else None
                ),
                "mean_steer": (
                    float(np.mean(steer_values))
                    if steer_values else None
                ),
                "trace_csv": str(trace_path),
            })

            print(
                "CASE_RESULT",
                json.dumps(result, ensure_ascii=False),
                flush=True,
            )

        except Exception as exc:
            result.update({
                "status": "ERROR",
                "error": repr(exc),
            })

            print(
                "CASE_ERROR",
                json.dumps(result, ensure_ascii=False),
                flush=True,
            )

            traceback.print_exc()

        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass

            if env is not None and callable(
                getattr(env, "close", None)
            ):
                try:
                    env.close()
                except Exception:
                    pass

        results.append(result)

    # ---------------------------------------------------------
    # 6. Paired-seed audit.
    # ---------------------------------------------------------
    pair_results = {}

    for seed in sorted({
        case[1] for case in cases
    }):
        pair = [
            r for r in results
            if r.get("pair") == str(seed)
        ]

        actual_seeds = [
            r.get("actual_seed")
            for r in pair
            if r.get("actual_seed") is not None
        ]

        strict_match = (
            len(pair) == 2
            and len(actual_seeds) == 2
            and actual_seeds[0] == actual_seeds[1]
            and all(
                r.get("status") == "EVALUATED"
                for r in pair
            )
        )

        pair_results[str(seed)] = {
            "strict_actual_seed_match": strict_match,
            "cases": pair,
        }

    summary = {
        "experiment": "STAGE_T140_FROZEN_DIRECTION_DIAGNOSIS",
        "stage_only": True,
        "training": False,
        "actor_update": False,
        "checkpoint_overwrite": False,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "results": results,
        "paired_seed_audit": pair_results,
    }

    summary_path = EVAL_JOB / "t140_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 82)
    print("T140_EVALUATION_FINISHED")
    print("SUMMARY =", summary_path)
    print("RESULTS_DIR =", EVAL_JOB)
    print("TRAINING_STARTED=FALSE")
    print("ACTOR_UPDATE=FALSE")
    print("CHECKPOINT_OVERWRITE=FALSE")
    print("PHASE_TOUCHED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
    print("=" * 82)


try:
    main()

except Exception as exc:
    print("T140_ABORTED =", repr(exc))
    print("TRAINING_STARTED=FALSE")
    print("CHECKPOINT_OVERWRITE=FALSE")
    print("PHASE_TOUCHED=FALSE")
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
