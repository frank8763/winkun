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
JOB = Path('/home/linghu/mkmini_stage_td3_t83_failure_video_20261028_TQzYAP')

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
  'T53': {
    'path': Path('/home/linghu/mkmini_stage_td3_t53_actor_anchor_20261027_v1/checkpoints/episode_0010.pt'),
    'sha': '8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0',
    'frames': 22491,
    'core_sha': '35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07',
  },
  'T82': {
    'path': Path('/home/linghu/mkmini_stage_td3_t82_balanced_replay_20261028_Wc0lpr/checkpoints/episode_0028.pt'),
    'sha': 'bb0ad57b18a48d72abe6bc4d740e82c407935c587ce6e2e16d43e9a50609b7f2',
    'frames': 190491,
    'core_sha': 'c7e06605579a8dec7d0bd95033ae4379975833968fe69968e7294396e9b21cdc',
  },
}

SEEDS = (2029653601, 2029653602)

TASKS = [(2029653601, -17.3, 3.0), (2029653602, 13.6, 3.0), (2029654801, -12.5, 3.0), (2029654802, 18.8, 3.0)]

CAP = 14000
WALL = 4800


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
            "T71E_MODEL_VERIFIED=",
            name,
            "FRAMES=",
            config["frames"],
            flush=True
        )

    manifest = {
        "experiment": "STAGE_TD3_T71E",
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
            "T71E_MANIFEST_CHANGED"
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

    print("T71E_PREFLIGHT=PASS", flush=True)

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
                _t83_capture(N.step_checked, env, action, locals())
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
                emit("T71E_PROGRESS", {
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



# =========================================================
# T83: observation-only native MuJoCo video capture.
# Original Actor action and native step are unchanged.
# =========================================================
import atexit as _t83_atexit
import csv as _t83_csv
import math as _t83_math
import numpy as _t83_np
import mujoco as _t83_mj
import cv2 as _t83_cv2
import os as _t83_os

_T83_DIR = Path(_t83_os.environ["T83_JOB"]) / "videos"
_T83_DIR.mkdir(parents=True, exist_ok=True)

_T83_TASKS = {
    2029654801: {
        "tag": "LEFT_-12p5",
        "start": (1.75, -4.0),
        "goal": (3.3623608142541768, -6.529879958546968),
    },
    2029654802: {
        "tag": "RIGHT_+18p8",
        "start": (1.0, 2.25),
        "goal": (3.6920169243256953, 0.9259739884941749),
    },
}

_T83_RECORDS = {}
_T83_CONTEXT_SHOWN = False

def _t83_label(context):
    name = context.get("name")
    seed = context.get("seed")

    if name != "T82":
        for value in context.values():
            if isinstance(value, str) and value == "T82":
                name = value
                break

    if seed not in _T83_TASKS:
        for value in context.values():
            try:
                if (
                    not isinstance(value, (str, bytes, bool))
                    and int(value) in _T83_TASKS
                ):
                    seed = int(value)
                    break
            except (TypeError, ValueError, OverflowError):
                pass

    return name, seed

def _t83_add_sphere(scene, xyz, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]

    _t83_mj.mjv_initGeom(
        geom,
        _t83_mj.mjtGeom.mjGEOM_SPHERE,
        _t83_np.array([radius, radius, radius]),
        _t83_np.asarray(xyz, dtype=float),
        _t83_np.eye(3).reshape(-1),
        _t83_np.asarray(rgba, dtype=float),
    )

    scene.ngeom += 1

class _T83Recorder:
    def __init__(self, env, seed):
        self.seed = seed
        self.task = _T83_TASKS[seed]
        self.calls = 0
        self.saved = 0
        self.trail = []

        name = "T82_" + self.task["tag"] + "_seed" + str(seed)

        self.video_path = _T83_DIR / (name + ".mp4")
        self.csv_path = _T83_DIR / (name + ".csv")

        self.width = 640
        self.height = 480

        self.renderer = _t83_mj.Renderer(
            env.model,
            width=self.width,
            height=self.height,
        )

        self.camera = _t83_mj.MjvCamera()

        self.camera.type = (
            _t83_mj.mjtCamera.mjCAMERA_FREE
        )

        start = self.task["start"]
        goal = self.task["goal"]

        self.camera.lookat[:] = [
            (start[0] + goal[0]) / 2.0,
            (start[1] + goal[1]) / 2.0,
            0.0,
        ]

        self.camera.distance = 9.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -72.0

        self.writer = _t83_cv2.VideoWriter(
            str(self.video_path),
            _t83_cv2.VideoWriter_fourcc(*"mp4v"),
            12.0,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                "T83_MP4_WRITER_UNAVAILABLE:" +
                str(self.video_path)
            )

        self.csv_file = self.csv_path.open(
            "w", newline=""
        )

        self.csv_writer = _t83_csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "sample",
            "physics_steps_seen",
            "x_m",
            "y_m",
            "goal_distance_m",
            "goal_bearing_rad",
            "throttle",
            "steer",
            "lidar_min_m",
        ])

        print(
            "T83_RECORDING_STARTED",
            "SEED=", seed,
            "VIDEO=", self.video_path,
            flush=True
        )

    def snapshot(self, env, action):
        self.calls += 1

        # Recording only; native simulation still runs
        # at its original physics stepping frequency.
        if self.calls % 60 != 1:
            return

        model = env.model
        data = env.data

        bid = _t83_mj.mj_name2id(
            model,
            _t83_mj.mjtObj.mjOBJ_BODY,
            "mkmini_base",
        )

        if bid < 0:
            raise RuntimeError(
                "T83_NATIVE_CHASSIS_BODY_MISSING"
            )

        xyz = _t83_np.asarray(
            data.xpos[bid],
            dtype=float
        ).copy()

        x, y = float(xyz[0]), float(xyz[1])

        goal = self.task["goal"]

        distance = _t83_math.hypot(
            goal[0] - x,
            goal[1] - y,
        )

        goal_state = env._goal_state()

        bearing = float(goal_state[2])

        lidar = _t83_np.asarray(
            env._lidar48(),
            dtype=float
        ).reshape(-1)

        lidar_min = (
            float(lidar.min()) * 6.0
            if lidar.size == 48
            else float("nan")
        )

        self.csv_writer.writerow([
            self.saved,
            self.calls,
            x,
            y,
            distance,
            bearing,
            float(action[0]),
            float(action[1]),
            lidar_min,
        ])

        self.trail.append(
            (x, y)
        )

        self.renderer.update_scene(
            data,
            camera=self.camera,
        )

        scene = self.renderer.scene

        # Red marker: goal.
        _t83_add_sphere(
            scene,
            (goal[0], goal[1], 0.22),
            0.18,
            (1.0, 0.15, 0.10, 1.0),
        )

        # Yellow markers: previous robot positions.
        for tx, ty in self.trail[-55::3]:
            _t83_add_sphere(
                scene,
                (tx, ty, 0.07),
                0.045,
                (1.0, 0.85, 0.10, 1.0),
            )

        rgb = self.renderer.render()

        frame = _t83_cv2.cvtColor(
            rgb,
            _t83_cv2.COLOR_RGB2BGR,
        )

        label = (
            f"T82 | seed {self.seed} | "
            f"step {self.calls} | "
            f"goal {distance:.3f}m"
        )

        controls = (
            f"throttle {float(action[0]):.3f}  "
            f"steer {float(action[1]):+.3f}  "
            f"LiDAR {lidar_min:.2f}m"
        )

        _t83_cv2.rectangle(
            frame,
            (0, 0),
            (self.width, 63),
            (20, 20, 20),
            -1,
        )

        _t83_cv2.putText(
            frame,
            label,
            (12, 24),
            _t83_cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            _t83_cv2.LINE_AA,
        )

        _t83_cv2.putText(
            frame,
            controls,
            (12, 48),
            _t83_cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            (230, 230, 230),
            1,
            _t83_cv2.LINE_AA,
        )

        self.writer.write(frame)

        self.saved += 1

    def close(self):
        self.writer.release()
        self.csv_file.close()
        self.renderer.close()

        print(
            "T83_VIDEO_FINALIZED",
            "SEED=", self.seed,
            "FRAMES=", self.saved,
            "VIDEO=", self.video_path,
            flush=True,
        )

def _t83_close_all():
    for rec in _T83_RECORDS.values():
        try:
            rec.close()
        except Exception as error:
            print(
                "T83_CLOSE_WARNING=",
                repr(error),
                flush=True,
            )

_t83_atexit.register(_t83_close_all)

def _t83_capture(native_step, env, action, context):
    global _T83_CONTEXT_SHOWN

    if not _T83_CONTEXT_SHOWN:
        print(
            "T83_STEP_CONTEXT_KEYS=",
            sorted(str(k) for k in context.keys()),
            flush=True,
        )
        _T83_CONTEXT_SHOWN = True

    name, seed = _t83_label(context)

    # Preserve the exact native step call.
    result = native_step(env, action)

    if name == "T82" and seed in _T83_TASKS:
        if seed not in _T83_RECORDS:
            _T83_RECORDS[seed] = _T83Recorder(
                env, seed
            )

        _T83_RECORDS[seed].snapshot(
            env, action
        )

    return result


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
                    "T71E_RESUMED_CASE=",
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
                        "T71E_STATE=PAUSED_RESUMABLE"
                    )
                    return

                atomic_json(
                    result_path,
                    result
                )

                emit("T71E_CASE", {
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

        emit("T71E_PAIR", {
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

        emit("T71E_MODEL", {
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

    print("T71E_PAIR_GATE=PASS")
    print("T71E_STATE=EVALUATED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()

except Exception as error:
    print(
        "T71E_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
