from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback

import numpy as np
import torch
import mujoco

H = Path.home()
JOB = H / "mkmini_stage_td3_t11_goal_sensitivity_20261027_v1"

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

CONTROL = H / (
    "mkmini_stage_td3_t9_control_20261027_v1/"
    "checkpoints/episode_0004.pt"
)

REG3675 = H / (
    "mkmini_stage_td3_t9_regularized_20261027_v1/"
    "checkpoints/episode_0004.pt"
)

REG4000 = H / (
    "mkmini_stage_td3_t9_regularized_20261027_v1/"
    "checkpoints/episode_0005.pt"
)

EXPECTED = {
    CORE: "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",
    HELPER: "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE: "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN: "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
    CONTROL: "580b6aa2929fff3cb432d4ce1036bb03564d973d357e23e00cb3a3c9b75e63cf",
    REG3675: "fdbcc174597d2980b1541734e2b5d832bd0dbb4c324f8528419a152e777b9f75",
    REG4000: "b673a49c3dbae4dacff3b29fa25f4bdb2c3a6cd8ce2c5cd95be5525593b68040",
}

MODELS = {
    "control_3675": CONTROL,
    "regularized_3675": REG3675,
    "regularized_4000": REG4000,
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


def emit(name, value):
    print(
        name,
        json.dumps(value, ensure_ascii=False, default=str),
        flush=True
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    check(spec is not None and spec.loader is not None,
          "MODULE_SPEC_INVALID")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("SIMULATION_STEPS=ZERO")
    print("TD3_TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("MODEL_PROMOTION=FALSE")

    for path, expected in EXPECTED.items():
        check(path.is_file() and sha(path) == expected,
              "SOURCE_INTEGRITY_FAIL:" + str(path))

    torch.set_num_threads(2)

    core = load_module(CORE, "stage_td3_t11_core")

    actors = {}

    for name, path in MODELS.items():
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )

        check(checkpoint.get("algorithm") == "TD3",
              "ALGORITHM_INVALID:" + name)

        actor = core.Actor(112, 2)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        actors[name] = actor

        emit("TD3_T11_MODEL_VERIFIED=", {
            "name": name,
            "frames": checkpoint["frames"],
            "sha256": sha(path)
        })

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(HELPER, "stage_td3_t11_helper")
    legacy = load_module(STAGE, "stage_td3_t11_stage")

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file() and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC, "stage_td3_t11_native"
    )

    _, v, native_mod, prep, hashes = runner.load_native()
    check(len(prep) >= 4, "NATIVE_BOOTSTRAP_INVALID")

    N = prep[3]
    make_base = legacy.make_short_task(runner.make_task)

    rows = []

    # 使用独立诊断种子。
    # 只允许目标终点变化，机器人本体状态完全一致。
    for k in range(18):
        if len(rows) >= 6:
            break

        seed = 2029552000 + k
        env = None
        renderer = None

        try:
            env, renderer, _, actual_seed = (
                helper.make_goal_task(
                    make_base, v, native_mod, N,
                    seed, -20.0, 0.8
                )
            )

            start = np.asarray(
                env.data.qpos[:2], dtype=np.float64
            ).copy()

            yaw = float(env._pose()[2])
            distance = 0.8

            left_goal = start + distance * np.array([
                math.cos(yaw + math.radians(-20)),
                math.sin(yaw + math.radians(-20))
            ])

            right_goal = start + distance * np.array([
                math.cos(yaw + math.radians(20)),
                math.sin(yaw + math.radians(20))
            ])

            # 检查右侧目标位置是否安全。
            # 这里只检查碰撞，不执行物理步。
            qpos0 = env.data.qpos.copy()
            qvel0 = env.data.qvel.copy()
            ctrl0 = env.data.ctrl.copy()

            try:
                env.data.qpos[:2] = right_goal
                mujoco.mj_forward(env.model, env.data)
                right_collides = bool(env._collision())

            finally:
                env.data.qpos[:] = qpos0
                env.data.qvel[:] = qvel0

                if env.data.ctrl.size:
                    env.data.ctrl[:] = ctrl0

                mujoco.mj_forward(env.model, env.data)

            if right_collides:
                emit("TD3_T11_SKIP_UNSAFE_RIGHT_GOAL=", seed)
                continue

            observations = {}
            bearings = {}

            for side, goal in (
                ("LEFT", left_goal),
                ("RIGHT", right_goal),
                ("LEFT_RECHECK", left_goal)
            ):
                env.goal_xy = goal.copy()
                env.prev_goal_dist = float(
                    env._goal_state()[1]
                )

                bearings[side] = math.degrees(
                    wrap(float(env._goal_state()[2]))
                )

                observations[side] = np.asarray(
                    env._observation(update_history=False),
                    dtype=np.float32
                ).reshape(-1).copy()

                check(
                    observations[side].shape == (112,),
                    "OBSERVATION_DIM_INVALID"
                )

                check(
                    np.isfinite(observations[side]).all(),
                    "OBSERVATION_NONFINITE"
                )

            left = observations["LEFT"]
            right = observations["RIGHT"]
            left_recheck = observations["LEFT_RECHECK"]

            repeat_error = float(
                np.max(np.abs(left - left_recheck))
            )

            # 确保仅改变目标时，雷达没有发生变化。
            lidar_error = float(
                np.max(np.abs(left[:48] - right[:48]))
            )

            changed = np.where(
                np.abs(left - right) > 1e-5
            )[0].tolist()

            check(
                abs(bearings["LEFT"] + 20) < 2,
                "LEFT_BEARING_WRONG"
            )
            check(
                abs(bearings["RIGHT"] - 20) < 2,
                "RIGHT_BEARING_WRONG"
            )

            check(
                repeat_error < 1e-4,
                "OBSERVATION_NOT_REPEATABLE"
            )

            check(
                lidar_error < 1e-4,
                "LIDAR_CHANGED_WITH_GOAL"
            )

            check(
                len(changed) > 0,
                "GOAL_DIRECTION_NOT_ENCODED_IN_OBSERVATION"
            )

            outputs = {}

            for name, actor in actors.items():

                with torch.no_grad():
                    raw = actor(
                        torch.from_numpy(
                            np.stack([left, right])
                        )
                    ).cpu().numpy()

                outputs[name] = {
                    "left_throttle": float(
                        (raw[0, 0] + 1) * 0.5
                    ),
                    "left_steer": float(raw[0, 1]),
                    "right_throttle": float(
                        (raw[1, 0] + 1) * 0.5
                    ),
                    "right_steer": float(raw[1, 1]),
                    "steer_difference": float(
                        raw[1, 1] - raw[0, 1]
                    ),
                    "right_turn_positive": bool(
                        raw[1, 1] > 0.05
                    )
                }

            row = {
                "seed": actual_seed,
                "start_xy": start.tolist(),
                "bearing_left_deg": bearings["LEFT"],
                "bearing_right_deg": bearings["RIGHT"],
                "obs_changed_indices": changed,
                "obs_changed_count": len(changed),
                "obs_repeat_error": repeat_error,
                "lidar_error": lidar_error,
                "actors": outputs
            }

            rows.append(row)

            emit("TD3_T11_MATCHED_GOAL_CASE=", row)

        finally:
            if renderer is not None:
                renderer.close()

            if env is not None:
                env.close()

    check(
        len(rows) >= 4,
        "INSUFFICIENT_SAFE_MATCHED_GOAL_PAIRS"
    )

    summary = {}

    for name in MODELS:
        outputs = [r["actors"][name] for r in rows]

        summary[name] = {
            "cases": len(outputs),
            "mean_left_steer": float(np.mean([
                x["left_steer"] for x in outputs
            ])),
            "mean_right_steer": float(np.mean([
                x["right_steer"] for x in outputs
            ])),
            "mean_right_minus_left_steer": float(np.mean([
                x["steer_difference"] for x in outputs
            ])),
            "right_positive_fraction": float(np.mean([
                x["right_turn_positive"] for x in outputs
            ]))
        }

    final = {
        "state": "AUDIT_COMPLETED",
        "valid_matched_pairs": len(rows),
        "models": summary,
        "simulation_steps": 0,
        "training": False,
        "automatic_promotion": False
    }

    report = {
        "summary": final,
        "cases": rows
    }

    output = JOB / "report.json"
    temp = JOB / "report.json.partial"

    temp.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ) + "\n"
    )
    os.replace(temp, output)

    check(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    emit("TD3_T11_FINAL=", final)
    print("FROZEN_MODEL_UNCHANGED=TRUE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "TD3_T11_BLOCKED=",
            repr(error),
            flush=True
        )
        traceback.print_exc()
        raise
    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
