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
import mujoco
import mujoco.viewer

H = Path.home()

JOB = H / "mkmini_stage_td3_t34_native_gui_20261027_v1"

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

REFERENCE = (
    H / "mkmini_stage_td3_t28_direction_validation_20261027_v1"
    / "task_08_T16_13491.json"
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
}

SEED = 2029652802
ANGLE = 20
DISTANCE = 3.0
CAP = 14000


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)

    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    require(
        spec is not None and spec.loader is not None,
        "MODULE_LOAD_FAILED:" + name
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def main():

    print("=" * 70)
    print("STAGE TD3 — NATIVE MUJOCO GUI")
    print("=" * 70)

    print("STAGE_ONLY=TRUE")
    print("NATIVE_MUJOCO_WINDOW=TRUE")
    print("VIDEO_RECORDING=FALSE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    require(
        bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY")),
        "NO_DESKTOP_DISPLAY: 请在Ubuntu图形桌面终端运行"
    )

    for path, expected in EXPECTED.items():
        require(
            path.is_file() and sha(path) == expected,
            "SOURCE_CHECK_FAILED:" + str(path)
        )

    require(
        REFERENCE.is_file(),
        "HISTORICAL_SUCCESS_RECORD_MISSING"
    )

    reference = json.loads(REFERENCE.read_text())

    # 修复原T34错误：
    # T28历史记录可能没有checkpoint_sha256字段。
    # 检查点真实性由上面的文件SHA256直接验证。
    require(
        reference.get("model") == "T16_13491"
        and reference.get("requested_seed") == SEED
        and reference.get("actual_seed") == SEED
        and reference.get("angle_deg") == ANGLE
        and reference.get("distance_m") == DISTANCE
        and reference.get("success") is True
        and reference.get("collision") is False
        and reference.get("frames") == 10339,
        "HISTORICAL_SUCCESS_RECORD_INVALID"
    )

    if "checkpoint_sha256" in reference:
        require(
            reference["checkpoint_sha256"] == EXPECTED[CKPT],
            "REFERENCE_CHECKPOINT_HASH_MISMATCH"
        )

    print("HISTORICAL_SUCCESS=VERIFIED")

    torch.set_num_threads(2)

    core = load_module(CORE, "stage_gui_core")

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    require(
        checkpoint.get("algorithm") == "TD3"
        and checkpoint.get("frames") == 13491,
        "CHECKPOINT_METADATA_INVALID"
    )

    actor = core.Actor(112, 2)

    actor.load_state_dict(
        checkpoint["actor"],
        strict=True
    )

    actor.eval()

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER,
        "stage_gui_helper"
    )

    legacy = load_module(
        STAGE,
        "stage_gui_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        require(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_gui_native"
    )

    _, v, native_mod, prep, _ = runner.load_native()

    require(
        len(prep) >= 4,
        "NATIVE_RUNTIME_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    env = None
    native_renderer = None

    try:

        env, native_renderer, obs, actual_seed = (
            helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                SEED,
                ANGLE,
                DISTANCE
            )
        )

        env.max_steps = CAP

        start = np.asarray(
            env.data.qpos[:2],
            dtype=float
        ).copy()

        goal = np.asarray(
            env.goal_xy,
            dtype=float
        ).copy()

        require(
            int(actual_seed) == SEED,
            "SCENE_SEED_MISMATCH"
        )

        require(
            np.allclose(
                start,
                reference["start_xy"],
                atol=1e-7,
                rtol=0
            ),
            "SCENE_START_MISMATCH"
        )

        require(
            np.allclose(
                goal,
                reference["goal_xy"],
                atol=1e-7,
                rtol=0
            ),
            "SCENE_GOAL_MISMATCH"
        )

        require(
            abs(float(env._goal_state()[1]) - 3.0) < 0.01,
            "SCENE_DISTANCE_MISMATCH"
        )

        print("SCENE_VERIFICATION=PASS")
        print("START_XY=", start.tolist())
        print("GOAL_XY=", goal.tolist())

        # 关闭初始化过程中的离屏Renderer。
        # 后面只使用MuJoCo原生GUI。
        if native_renderer is not None:
            native_renderer.close()
            native_renderer = None

        obs = np.asarray(
            obs,
            dtype=np.float32
        ).reshape(-1)

        require(
            obs.shape == (112,),
            "OBSERVATION_DIM_INVALID"
        )

        print()
        print("正在打开真正的MuJoCo窗口……")
        print("模型：T16-13491")
        print("目标：右侧20度、3米")
        print("关闭MuJoCo窗口即可结束本次运行")
        print()

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:

            # 原生MuJoCo相机
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            viewer.cam.lookat[:] = [
                (start[0] + goal[0]) / 2,
                (start[1] + goal[1]) / 2,
                0.0
            ]

            viewer.cam.distance = 5.5
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -70.0

            # MuJoCo GUI中的红色目标球：
            # 仅作为viewer标记，不加入物理模型。
            if viewer.user_scn.maxgeom > 0:

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.14, 0.14, 0.14]),
                    np.array([
                        goal[0],
                        goal[1],
                        0.18
                    ]),
                    np.eye(3).reshape(-1),
                    np.array([
                        1.0, 0.1, 0.1, 1.0
                    ])
                )

                viewer.user_scn.ngeom = 1

            viewer.sync()

            sim_start = float(env.data.time)
            wall_start = time.monotonic()

            closest = 3.0
            success = False
            collision = False
            reason = "CAP"
            frame = 0

            print("MUJOCO_GUI_OPENED=TRUE", flush=True)

            for frame in range(1, CAP + 1):

                if not viewer.is_running():
                    reason = "USER_CLOSED"
                    break

                with torch.no_grad():

                    raw = actor(
                        torch.from_numpy(
                            obs.reshape(1, 112)
                        )
                    ).cpu().numpy()[0]

                require(
                    raw.shape == (2,)
                    and np.isfinite(raw).all(),
                    "ACTOR_OUTPUT_INVALID"
                )

                # 原始TD3动作映射
                action = np.asarray(
                    [
                        np.clip(
                            (float(raw[0]) + 1.0) / 2.0,
                            0.0,
                            1.0
                        ),
                        np.clip(
                            float(raw[1]),
                            -1.0,
                            1.0
                        )
                    ],
                    dtype=np.float32
                )

                obs, _, terminated, truncated, info = (
                    N.step_checked(env, action)
                )

                obs = np.asarray(
                    obs,
                    dtype=np.float32
                ).reshape(-1)

                remaining = float(
                    env._goal_state()[1]
                )

                closest = min(closest, remaining)

                success = bool(
                    info.get("reach_goal", False)
                )

                collision = bool(
                    info.get("collision", False)
                )

                done = (
                    success
                    or collision
                    or terminated
                    or truncated
                )

                # MuJoCo实时窗口刷新
                if frame % 10 == 0 or done:

                    viewer.sync()

                    # 尽量按照仿真物理时间播放，
                    # 不加速篡改物理，也不覆盖控制动作。
                    simulated_elapsed = (
                        float(env.data.time) - sim_start
                    )

                    actual_elapsed = (
                        time.monotonic() - wall_start
                    )

                    ahead = simulated_elapsed - actual_elapsed

                    if ahead > 0:
                        time.sleep(min(ahead, 0.1))

                if frame % 1000 == 0:

                    print(
                        "MUJOCO_PROGRESS",
                        "FRAME=", frame,
                        "REMAINING_M=", round(remaining, 4),
                        flush=True
                    )

                if done:

                    reason = (
                        "GOAL" if success
                        else "CONTACT" if collision
                        else "TERMINATED" if terminated
                        else "TRUNCATED"
                    )

                    break

            result = {
                "model": "T16_13491",
                "seed": SEED,
                "angle_deg": ANGLE,
                "distance_m": DISTANCE,
                "frames": frame,
                "success": success,
                "collision": collision,
                "end": reason,
                "closest_m": closest,
                "historical_success_frames": 10339,
                "exact_reproduction": (
                    success
                    and not collision
                    and frame == 10339
                ),
                "training": False,
                "native_mujoco_gui": True,
                "action_override": False
            }

            print()
            print(
                "MUJOCO_FINAL_RESULT=",
                json.dumps(
                    result,
                    ensure_ascii=False
                ),
                flush=True
            )

            (JOB / "result.json").write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                ) + "\n"
            )

            # 仿真完成后保留窗口，
            # 方便你查看最终车身与目标位置。
            if viewer.is_running():

                print()
                print("仿真结束，MuJoCo窗口保持打开。")
                print("查看完成后直接关闭MuJoCo窗口。")

                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)

    finally:

        if native_renderer is not None:
            try:
                native_renderer.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    require(
        sha(CKPT) == EXPECTED[CKPT],
        "FROZEN_CHECKPOINT_CHANGED"
    )

    print("FROZEN_CHECKPOINT_UNCHANGED=TRUE")


try:
    main()

except Exception as error:
    print("MUJOCO_GUI_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
