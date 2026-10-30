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

H = Path.home()

JOB = (
    H / "mkmini_stage_td3_t22_visual_20261027_v1"
)

CORE = (
    H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
    / "td3_core_goal_contrast.py"
)

CKPT = (
    H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
    / "checkpoints/episode_0008.pt"
)

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

EXPECTED = {
    CKPT:
        "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
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
    ("right_1m_verified", 2029606102, 20, 1.0, 6000),
    ("left_1p2m_verified", 2029606102, -20, 1.2, 6500),
    ("right_3m_baseline", 2029606102, 20, 3.0, 12000),
    ("left_3m_baseline", 2029606102, -20, 3.0, 12000),
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(part)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

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
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ) + "\n"
    )

    os.replace(temp, path)


def create_writer(path):
    try:
        import imageio.v2 as imageio

        return imageio.get_writer(
            str(path),
            fps=12,
            quality=6,
            macro_block_size=16
        )

    except Exception as error:
        print(
            "T22_VIDEO_BACKEND_UNAVAILABLE=",
            repr(error),
            flush=True
        )
        return None


def make_frame(
    renderer,
    camera,
    env,
    goal,
    task_name,
    frame,
    remaining
):
    xy = np.asarray(
        env.data.qpos[:2], dtype=float
    )

    camera.lookat[:] = [
        float((xy[0] + goal[0]) / 2),
        float((xy[1] + goal[1]) / 2),
        0.05
    ]

    renderer.update_scene(
        env.data,
        camera=camera
    )

    scene = renderer.scene

    # 红球只用于视频显示，不参与物理或控制。
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.17, 0.17, 0.17]),
            np.array([
                goal[0], goal[1], 0.12
            ]),
            np.eye(3).ravel(),
            np.array([1.0, 0.08, 0.08, 1.0])
        )

        scene.ngeom += 1

    rgb = renderer.render()

    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)

        draw.rectangle(
            (5, 5, 635, 42),
            fill=(0, 0, 0)
        )

        label = (
            f"STAGE TD3 | {task_name} | "
            f"frame={frame} | "
            f"goal={remaining:.3f}m"
        )

        draw.text(
            (10, 10),
            label,
            fill=(255, 255, 255)
        )

        return np.asarray(image)

    except ImportError:
        return rgb


def main():
    JOB.mkdir(
        parents=True,
        exist_ok=True
    )

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    for path, expected in EXPECTED.items():
        check(
            path.is_file()
            and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    torch.set_num_threads(2)

    core = load(
        CORE, "stage_t22_td3_core"
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    check(
        checkpoint["algorithm"] == "TD3"
        and checkpoint["frames"] == 13491,
        "CHECKPOINT_METADATA_INVALID"
    )

    actor = core.Actor(112, 2)

    actor.load_state_dict(
        checkpoint["actor"]
    )

    actor.eval()

    print("T22_CHECKPOINT=VERIFIED_13491")

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(
        HELPER, "stage_t22_helper"
    )

    legacy = load(
        STAGE, "stage_t22_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_t22_native"
    )

    _, v, native_mod, prep, hashes = (
        runner.load_native()
    )

    check(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    deadline = time.monotonic() + 2100

    for tag, seed, angle, distance, cap in TASKS:

        output = JOB / (tag + ".json")
        movie = JOB / (tag + ".mp4")
        temporary_movie = JOB / (tag + ".partial.mp4")

        if output.exists():
            existing = json.loads(
                output.read_text()
            )

            check(
                existing.get("checkpoint_sha256")
                == EXPECTED[CKPT]
                and existing["task"]
                == [seed, angle, distance, cap],
                "RESUME_TASK_CHANGED:" + tag
            )

            print(
                "T22_ALREADY_COMPLETED=",
                tag,
                existing.get("success")
            )
            continue

        if time.monotonic() >= deadline:
            print(
                "T22_WALL_BUDGET_REACHED=TRUE"
            )
            break

        env = None
        native_renderer = None
        renderer = None
        writer = None

        try:
            (
                env,
                native_renderer,
                obs,
                actual_seed
            ) = helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                seed,
                angle,
                distance
            )

            if native_renderer is not None:
                native_renderer.close()
                native_renderer = None

            env.max_steps = cap

            obs = np.asarray(
                obs, dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "OBSERVATION_INVALID"
            )

            goal_xy = np.asarray(
                env.goal_xy,
                dtype=float
            ).copy()

            initial = float(
                env._goal_state()[1]
            )

            check(
                abs(initial - distance) < 0.04,
                "INITIAL_GOAL_DISTANCE_INVALID"
            )

            print(
                "T22_TASK_BEGIN=",
                tag,
                "DISTANCE_M=", distance,
                "SEED=", actual_seed,
                flush=True
            )

            renderer = mujoco.Renderer(
                env.model,
                width=640,
                height=360
            )

            camera = mujoco.MjvCamera()
            camera.type = (
                mujoco.mjtCamera.mjCAMERA_FREE
            )

            camera.distance = max(
                3.2,
                1.5 * distance + 1.8
            )

            camera.azimuth = 135.0
            camera.elevation = -58.0

            writer = create_writer(
                temporary_movie
            )

            stride = max(
                10,
                int(round(
                    1.0 / (
                        12.0
                        * float(env.model.opt.timestep)
                    )
                ))
            )

            closest = initial
            frames = 0
            success = False
            collision = False
            reason = "CAP"
            paused = False

            snapshot = JOB / (
                tag + "_snapshot.png"
            )

            for frame in range(1, cap + 1):

                if time.monotonic() >= deadline:
                    print(
                        "T22_TIMELIMIT_CASE_UNSAVED=",
                        tag
                    )
                    paused = True
                    break

                with torch.no_grad():
                    raw = actor(
                        torch.from_numpy(
                            obs.reshape(1, -1)
                        )
                    ).cpu().numpy()[0]

                check(
                    raw.shape == (2,)
                    and np.isfinite(raw).all(),
                    "ACTOR_ACTION_INVALID"
                )

                action = np.asarray(
                    [
                        np.clip(
                            (float(raw[0]) + 1) / 2,
                            0,
                            1
                        ),
                        np.clip(
                            float(raw[1]),
                            -1,
                            1
                        )
                    ],
                    dtype=np.float32
                )

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

                frames = frame

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

                if (
                    frame == 1
                    or frame % stride == 0
                    or done
                ):
                    rgb = make_frame(
                        renderer,
                        camera,
                        env,
                        goal_xy,
                        tag,
                        frame,
                        remaining
                    )

                    if writer is not None:
                        writer.append_data(rgb)

                    if (
                        frame == 1
                        or done
                        or frame % (stride * 20) == 0
                    ):
                        try:
                            from PIL import Image

                            Image.fromarray(rgb).save(
                                snapshot
                            )
                        except Exception:
                            pass

                if frame % 2000 == 0:
                    print(
                        "T22_PROGRESS=",
                        tag,
                        "FRAMES=", frame,
                        "REMAINING_M=",
                        round(remaining, 3),
                        flush=True
                    )

                if done:
                    if success:
                        reason = "GOAL"
                    elif collision:
                        reason = "CONTACT"
                    elif terminated:
                        reason = "TERMINATED"
                    else:
                        reason = "TRUNCATED"

                    break

            if writer is not None:
                writer.close()
                writer = None

                if not paused:
                    os.replace(
                        temporary_movie,
                        movie
                    )

            if paused:
                print(
                    "T22_CASE_PAUSED_RESUMABLE=",
                    tag
                )
                break

            result = {
                "task": [
                    seed,
                    angle,
                    distance,
                    cap
                ],
                "actual_seed": int(actual_seed),
                "checkpoint_sha256": EXPECTED[CKPT],
                "frames": frames,
                "success": bool(success),
                "collision": bool(collision),
                "end": reason,
                "closest_m": closest,
                "final_m": remaining,
                "mp4": (
                    str(movie)
                    if movie.exists()
                    else None
                ),
                "snapshot": (
                    str(snapshot)
                    if snapshot.exists()
                    else None
                ),
                "training": False,
                "action_override": False
            }

            save_json(
                output,
                result
            )

            print(
                "T22_RESULT=",
                json.dumps(
                    {
                        "tag": tag,
                        **result
                    },
                    ensure_ascii=False
                ),
                flush=True
            )

        except Exception as error:
            print(
                "T22_CASE_BLOCKED=",
                tag,
                repr(error),
                flush=True
            )
            traceback.print_exc()

        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass

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

    check(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    print(
        "T22_FINISHED_NO_MODEL_PROMOTION=TRUE"
    )


try:
    main()

except Exception as error:
    print(
        "T22_BLOCKED=",
        repr(error),
        flush=True
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
        flush=True
    )
