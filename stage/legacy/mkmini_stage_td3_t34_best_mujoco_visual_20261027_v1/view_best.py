from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch
import mujoco

H = Path.home()

JOB = H / "mkmini_stage_td3_t34_best_mujoco_visual_20261027_v1"

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

FROZEN = (
    H / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
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
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

SEED = 2029652802
ANGLE = 20
DISTANCE = 3.0
CAP = 14000

VIDEO = JOB / "T16_best_verified_3m.mp4"
PARTIAL = JOB / "T16_best_verified_3m.partial.mp4"
REPORT = JOB / "T16_best_verified_3m.json"


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


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


class VideoWriter:
    def __init__(self, path, width, height):
        self.mode = None
        self.proc = None
        self.writer = None

        exe = shutil.which("ffmpeg")

        if exe:
            command = [
                exe,
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-y",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}",
                "-framerate", "20",
                "-i", "pipe:0",
                "-an",
                "-vcodec", "mpeg4",
                "-q:v", "4",
                "-pix_fmt", "yuv420p",
                str(path)
            ]

            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            self.mode = "ffmpeg"
            return

        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()

            self.writer = imageio.get_writer(
                str(path),
                fps=20,
                codec="mpeg4",
                macro_block_size=16
            )

            self.mode = "imageio"
            return

        except (ImportError, OSError, RuntimeError):
            pass

        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                20.0,
                (width, height)
            )

            if writer.isOpened():
                self.writer = writer
                self.mode = "opencv"
                return

            writer.release()

        except ImportError:
            pass

        raise RuntimeError(
            "VIDEO_ENCODER_MISSING: "
            "请先安装ffmpeg或imageio-ffmpeg"
        )

    def add(self, frame):
        rgb = np.ascontiguousarray(
            frame,
            dtype=np.uint8
        )

        if self.mode == "ffmpeg":
            self.proc.stdin.write(rgb.tobytes())

        elif self.mode == "imageio":
            self.writer.append_data(rgb)

        elif self.mode == "opencv":
            import cv2

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            self.writer.write(bgr)

    def close(self):
        if self.mode == "ffmpeg":
            self.proc.stdin.close()
            rc = self.proc.wait()

            if rc != 0:
                message = self.proc.stderr.read(
                4096
                ).decode(errors="replace")

                raise RuntimeError(
                    "FFMPEG_FAILED:" + message
                )

            self.proc.stderr.close()

        elif self.mode == "imageio":
            self.writer.close()

        elif self.mode == "opencv":
            self.writer.release()


def make_frame(renderer, camera, env, goal, frame, remaining):
    xy = np.asarray(
        env.data.qpos[:2],
        dtype=float
    )

    camera.lookat[:] = [
        0.45 * xy[0] + 0.55 * goal[0],
        0.45 * xy[1] + 0.55 * goal[1],
        0.0
    ]

    renderer.update_scene(
        env.data,
        camera=camera
    )

    scene = renderer.scene

    # 红色目标球仅作为可视化标记。
    # 不参与MuJoCo物理或Actor输入。
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.14, 0.14, 0.14]),
            np.array([
                goal[0],
                goal[1],
                0.17
            ]),
            np.eye(3).ravel(),
            np.array([1.0, 0.1, 0.1, 1.0])
        )

        scene.ngeom += 1

    rgb = renderer.render()

    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)

        draw.rectangle(
            (5, 5, 630, 40),
            fill=(0, 0, 0)
        )

        label = (
            f"STAGE TD3 | 3.0m | "
            f"frame={frame} | "
            f"goal={remaining:.3f}m"
        )

        draw.text(
            (12, 12),
            label,
            fill=(255, 255, 255)
        )

        return np.asarray(image)

    except ImportError:
        return rgb


def main():
    print("T34_STAGE_ONLY=TRUE")
    print("T34_TRAINING=FALSE")
    print("T34_PLANNER=FALSE")
    print("T34_ACTION_OVERRIDE=FALSE")

    for path, expected in EXPECTED.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_HASH_FAILED:" + str(path)
        )

    check(
        REFERENCE.is_file(),
        "T28_SUCCESS_REFERENCE_MISSING"
    )

    reference = json.loads(
        REFERENCE.read_text()
    )

    check(
        reference["model"] == "T16_13491"
        and reference["checkpoint_sha256"] == EXPECTED[CKPT]
        and reference["requested_seed"] == SEED
        and reference["actual_seed"] == SEED
        and reference["angle_deg"] == ANGLE
        and reference["distance_m"] == DISTANCE
        and reference["success"] is True
        and reference["collision"] is False
        and reference["frames"] == 10339,
        "T28_SUCCESS_REFERENCE_INVALID"
    )

    print("T34_HISTORICAL_3M_SUCCESS=VERIFIED")

    if VIDEO.exists():
        check(
            REPORT.exists(),
            "VIDEO_REPORT_MISSING"
        )

        old = json.loads(REPORT.read_text())

        check(
            old["success"] is True
            and old["collision"] is False
            and old["checkpoint_sha256"] == EXPECTED[CKPT],
            "EXISTING_VIDEO_NOT_VERIFIED"
        )

        print("T34_EXISTING_VIDEO=", VIDEO)
        print("T34_REPLAY=ALREADY_VERIFIED")
        return

    check(
        not PARTIAL.exists(),
        "PARTIAL_VIDEO_ALREADY_EXISTS"
    )

    torch.set_num_threads(2)

    core = load(CORE, "t34_stage_core")

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    check(
        checkpoint.get("algorithm") == "TD3"
        and checkpoint["frames"] == 13491,
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

    helper = load(
        HELPER,
        "t34_stage_helper"
    )

    legacy = load(
        STAGE,
        "t34_stage_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "t34_stage_native"
    )

    _, v, native_mod, prep, _ = runner.load_native()

    check(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_FAILED"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    env = None
    native_renderer = None
    renderer = None
    writer = None

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
        )

        goal = np.asarray(
            env.goal_xy,
            dtype=float
        )

        check(
            actual_seed == SEED,
            "SEED_MISMATCH"
        )

        check(
            np.allclose(
                start,
                reference["start_xy"],
                atol=1e-7,
                rtol=0
            ),
            "START_POSITION_CHANGED"
        )

        check(
            np.allclose(
                goal,
                reference["goal_xy"],
                atol=1e-7,
                rtol=0
            ),
            "GOAL_POSITION_CHANGED"
        )

        initial = float(
            env._goal_state()[1]
        )

        check(
            abs(initial - 3.0) < 0.01,
            "GOAL_DISTANCE_CHANGED"
        )

        print("T34_SCENE_IDENTITY=PASS")
        print("T34_START_XY=", start.tolist())
        print("T34_GOAL_XY=", goal.tolist())

        if native_renderer is not None:
            native_renderer.close()
            native_renderer = None

        renderer = mujoco.Renderer(
            env.model,
            width=640,
            height=360
        )

        camera = mujoco.MjvCamera()

        camera.type = (
            mujoco.mjtCamera.mjCAMERA_FREE
        )

        camera.distance = 5.2
        camera.azimuth = 135.0
        camera.elevation = -73.0

        writer = VideoWriter(
            PARTIAL,
            640,
            360
        )

        print("T34_VIDEO_ENCODER=", writer.mode)

        obs = np.asarray(
            obs,
            dtype=np.float32
        ).reshape(-1)

        closest = initial
        remaining = initial

        success = False
        collision = False
        reason = "CAP"

        rendered_frames = 0

        for frame in range(1, CAP + 1):
            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "OBSERVATION_INVALID"
            )

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, 112)
                    )
                ).cpu().numpy()[0]

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

            obs, reward, terminated, truncated, info = (
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

            if (
                frame == 1
                or frame % 25 == 0
                or done
            ):
                image = make_frame(
                    renderer,
                    camera,
                    env,
                    goal,
                    frame,
                    remaining
                )

                writer.add(image)
                rendered_frames += 1

            if frame % 2000 == 0:
                print(
                    "T34_PROGRESS",
                    "FRAME=", frame,
                    "REMAINING_M=",
                    round(remaining, 4),
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

        writer.close()
        writer = None

        os.replace(
            PARTIAL,
            VIDEO
        )

        reproduced = (
            success
            and not collision
            and frame == reference["frames"]
        )

        result = {
            "model": "T16_13491",
            "checkpoint_sha256": EXPECTED[CKPT],
            "seed": SEED,
            "angle_deg": ANGLE,
            "distance_m": DISTANCE,
            "frames": frame,
            "historical_frames": reference["frames"],
            "success": success,
            "collision": collision,
            "end": reason,
            "closest_m": closest,
            "final_m": remaining,
            "exact_reproduction": reproduced,
            "video_frames": rendered_frames,
            "video": str(VIDEO),
            "training": False,
            "action_override": False
        }

        REPORT.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ) + "\n"
        )

        print("T34_RESULT=", json.dumps(result))
        print("T34_VIDEO_READY=", VIDEO)

        if reproduced:
            print("T34_BEST_3M_REPLAY=PASS")
        else:
            print("T34_BEST_3M_REPLAY=NOT_REPRODUCED")

        check(
            sha(CKPT) == EXPECTED[CKPT]
            and sha(FROZEN) == EXPECTED[FROZEN],
            "FROZEN_MODEL_CHANGED"
        )

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


try:
    main()

except Exception as error:
    print(
        "T34_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
