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
JOB = H / "mkmini_stage_td3_t7_direction_audit_20261027_v1"

T6 = H / "mkmini_stage_td3_t6_balanced_20261027_v1"
CORE = H / "mkmini_stage_td3_adapter_20261027_v1/td3_core.py"
HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)
STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)
FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

HASHES = {
    CORE:
        "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

CHECKPOINT_HASHES = {
    1: "036ab1c72950a7cab75e06fe2f99c724610c98a17fbb0975c4d32b66008581c9",
    2: "ee66ed967bf8bc2477d2d6c49f008d84b405b4650b8745c95d2e97e79d590150",
    3: "bcc65bacaac38a5f4fb7a817eedddd200d66ab2c46529e8575ae34cfc502c8c0",
    4: "65dc34ffd7d82ba5e68436ae02125cba9bb36530002af4408104449be5df039c",
    5: "048e82702a923a4f04b10fc4e4e2b9f0973a096cefc7c23c02068a85c5c2aa6f",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(key, obj):
    print(
        key,
        json.dumps(obj, ensure_ascii=False, default=str),
        flush=True
    )


def save_report(data):
    target = JOB / "report.json"
    temp = JOB / "report.json.partial"
    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str
        ) + "\n"
    )
    os.replace(temp, target)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    check(
        spec is not None and spec.loader is not None,
        "IMPORT_SPEC_INVALID"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("FROZEN_MODEL_WRITE=FALSE")
    print("TD3_TRAINING=FALSE")
    print("EXTERNAL_PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTOR_PARAMETER_UPDATES=FALSE")

    for path, expected in HASHES.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_HASH_MISMATCH:" + str(path)
        )

    checkpoints = {}

    for index, expected in CHECKPOINT_HASHES.items():
        path = (
            T6 / "checkpoints"
            / f"episode_{index:04d}.pt"
        )
        check(
            path.is_file() and sha(path) == expected,
            "CHECKPOINT_HASH_MISMATCH:" + str(path)
        )
        checkpoints[index] = path

    t6_status = json.loads(
        (T6 / "status.json").read_text()
    )

    check(
        t6_status.get("frames") == 4000
        and t6_status.get("checkpoint_sha256")
            == CHECKPOINT_HASHES[5],
        "T6_FINAL_STATUS_MISMATCH"
    )

    torch.set_num_threads(2)
    torch.manual_seed(20261027)

    core = load(CORE, "stage_td3_t7_core")

    states = {
        i: torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        for i, path in checkpoints.items()
    }

    probes = states[5]["probes"]

    check(
        set(("LEFT", "RIGHT")).issubset(probes),
        "LEFT_RIGHT_PROBES_MISSING"
    )

    print()
    print("========== T7-1: IDENTICAL OBSERVATION ACTOR TEST ==========")

    actor_rows = []

    for i, state in states.items():
        actor = core.Actor(112, 2)
        actor.load_state_dict(state["actor"])
        actor.eval()

        row = {
            "episode": i,
            "frames": state["frames"],
            "updates": state["updates"],
        }

        for side in ("LEFT", "RIGHT"):
            x = np.asarray(
                probes[side],
                dtype=np.float32
            ).reshape(1, -1)

            check(x.shape == (1, 112), "BAD_PROBE")

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(x)
                ).numpy()[0]

            row[side] = {
                "throttle": round(
                    float((raw[0] + 1) / 2), 6
                ),
                "steer": round(float(raw[1]), 6)
            }

        row["steer_gap"] = round(
            abs(
                row["LEFT"]["steer"]
                - row["RIGHT"]["steer"]
            ),
            6
        )

        row["same_direction_high_output"] = (
            row["LEFT"]["throttle"] > 0.95
            and row["RIGHT"]["throttle"] > 0.95
            and row["LEFT"]["steer"] < -0.85
            and row["RIGHT"]["steer"] < -0.85
            and row["steer_gap"] < 0.20
        )

        actor_rows.append(row)

        emit("TD3_T7_ACTOR_STAGE=", row)

    print()
    print("========== T7-2: REAL REPLAY ACTION COVERAGE ==========")

    replay = states[5]["replay"]

    coverage = {}

    for side in ("LEFT", "RIGHT"):
        rows = [
            r for r in replay
            if len(r) >= 6 and r[5] == side
        ]

        a = np.asarray(
            [r[1] for r in rows],
            dtype=np.float32
        )

        check(
            a.ndim == 2 and a.shape[1] == 2,
            "REPLAY_ACTION_INVALID"
        )

        steer = a[:, 1]

        coverage[side] = {
            "count": int(len(steer)),
            "positive_fraction":
                float(np.mean(steer > 0.25)),
            "negative_fraction":
                float(np.mean(steer < -0.25)),
            "steer_mean": float(np.mean(steer))
        }

        emit(
            "TD3_T7_REPLAY_COVERAGE=",
            {"side": side, **coverage[side]}
        )

    print()
    print("========== T7-3: RIGHT-GOAL CRITIC Q TEST ==========")

    critic = core.Critic(112, 2)
    critic.load_state_dict(states[5]["critic"])
    critic.eval()

    x = torch.from_numpy(
        np.asarray(
            probes["RIGHT"],
            dtype=np.float32
        ).reshape(1, 112)
    )

    q_rows = []

    with torch.no_grad():

        for steering in (
            -0.8, -0.4, 0.0, 0.4, 0.8
        ):
            # raw throttle=0.2 -> physical throttle=0.6
            action = torch.tensor(
                [[0.2, steering]],
                dtype=torch.float32
            )

            q1, q2 = critic(x, action)

            q_rows.append({
                "raw_steer": steering,
                "q1": float(q1.item()),
                "q2": float(q2.item()),
                "min_q": float(
                    torch.minimum(q1, q2).item()
                )
            })

    for row in q_rows:
        emit("TD3_T7_Q_STEER=", row)

    print()
    print("========== T7-4: ACTUAL MUJOCO STEERING RESPONSE ==========")

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(HELPER, "stage_td3_t7_helper")
    legacy = load(STAGE, "stage_td3_t7_stage")

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_INVALID:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_td3_t7_native_runner"
    )

    _, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    check(
        isinstance(hashes, dict) and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_FAILED"
    )

    N = prep[3]

    check(
        callable(N.step_checked),
        "NATIVE_STEP_MISSING"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    physics_rows = []
    total_physics = 0

    deadline = time.monotonic() + 900

    for seed, angle in (
        (2029532100, -20),
        (2029532101, 20)
    ):
        pair = []

        for steer in (-0.65, 0.65):
            check(
                time.monotonic() < deadline,
                "WALL_BUDGET_EXCEEDED"
            )

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
                        0.8
                    )
                )

                initial_xy = np.asarray(
                    env.data.qpos[:2],
                    dtype=np.float64
                ).copy()

                initial_yaw = float(env._pose()[2])

                initial_bearing = float(
                    env._goal_state()[2]
                )

                initial_distance = float(
                    env._goal_state()[1]
                )

                steps = 0
                collision = False
                reached = False

                # 固定动作仅用于物理控制响应测试；
                # 不生成导航策略，不写入训练回放。
                action = np.asarray(
                    [0.6, steer],
                    dtype=np.float32
                )

                for _ in range(240):
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

                    check(
                        nframe == 1,
                        "PHYSICS_FRAME_DELTA_INVALID"
                    )

                    check(
                        np.isfinite(next_obs).all(),
                        "OBSERVATION_NONFINITE"
                    )

                    total_physics += nframe
                    steps += nframe

                    check(
                        total_physics <= 960,
                        "T7_PHYSICS_BUDGET_EXCEEDED"
                    )

                    collision = bool(
                        info.get("collision", False)
                    )

                    reached = bool(
                        info.get("reach_goal", False)
                    )

                    if (
                        collision
                        or reached
                        or terminated
                        or truncated
                    ):
                        break

                final_yaw = float(env._pose()[2])

                final_bearing = float(
                    env._goal_state()[2]
                )

                final_distance = float(
                    env._goal_state()[1]
                )

                row = {
                    "seed_requested": seed,
                    "seed_actual": actual_seed,
                    "goal_angle_deg": angle,
                    "steer_command": steer,
                    "throttle_command": 0.6,
                    "start_xy": initial_xy.tolist(),
                    "initial_yaw_rad": initial_yaw,
                    "yaw_change_deg":
                        math.degrees(wrap(
                            final_yaw - initial_yaw
                        )),
                    "bearing_change_deg":
                        math.degrees(wrap(
                            final_bearing - initial_bearing
                        )),
                    "initial_goal_m":
                        initial_distance,
                    "final_goal_m":
                        final_distance,
                    "progress_m":
                        initial_distance - final_distance,
                    "physics_frames": steps,
                    "collision": collision,
                    "reached": reached
                }

                pair.append(row)
                physics_rows.append(row)

                emit("TD3_T7_PHYSICS_CASE=", row)

            finally:
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()

        same_start = (
            pair[0]["seed_actual"]
            == pair[1]["seed_actual"]
            and np.linalg.norm(
                np.asarray(pair[0]["start_xy"])
                - np.asarray(pair[1]["start_xy"])
            ) < 1e-4
        )

        yaw_difference = abs(
            pair[1]["yaw_change_deg"]
            - pair[0]["yaw_change_deg"]
        )

        emit("TD3_T7_STEERING_PAIR=", {
            "angle": angle,
            "same_initial_start": same_start,
            "yaw_response_separation_deg":
                round(yaw_difference, 6),
            "negative_steer_yaw_deg":
                pair[0]["yaw_change_deg"],
            "positive_steer_yaw_deg":
                pair[1]["yaw_change_deg"],
            "interpretation":
                "PHYSICS_DIAGNOSTIC_ONLY"
        })

    check(
        sha(FROZEN) == HASHES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    final = {
        "state": "AUDIT_COMPLETED",
        "actor_stages": actor_rows,
        "replay_coverage": coverage,
        "critic_q_probe": q_rows,
        "physics": physics_rows,
        "total_physics_frames": total_physics,
        "training": False,
        "model_promoted": False,
        "frozen_model_verified": True
    }

    save_report(final)

    print()
    print("TD3_T7_AUDIT=PASS")
    print("TD3_T7_TOTAL_PHYSICS=", total_physics)
    print("TD3_T7_MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        emit(
            "TD3_T7_AUDIT=BLOCKED",
            {"error": repr(error)}
        )
        traceback.print_exc()

    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
