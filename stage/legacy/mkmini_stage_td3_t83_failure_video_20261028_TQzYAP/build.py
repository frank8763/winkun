from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

SRC = Path(os.environ["T82_JOB"]) / "pilot4/evaluate.py"
JOB = Path(os.environ["T83_JOB"])

EXPECTED_EVAL_SHA = (
    "aea31b20c9f92ece56010360dce8f222a8ba5693"
    "b356ce04bb5715ffd3d1fa18"
)

EXPECTED_MODEL_SHA = (
    "bb0ad57b18a48d72abe6bc4d740e82c407935"
    "c587ce6e2e16d43e9a50609b7f2"
)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def replace_top(source, name, expression):
    tree = ast.parse(source)

    found = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]

    check(len(found) == 1, "TOP_BINDING_INVALID:" + name)

    n = found[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

HELPER = r'''
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
'''

try:
    print("=" * 76)
    print("T83 — T82 NATIVE FAILURE VIDEO BUILDER")
    print("=" * 76)

    check(
        SRC.is_file()
        and sha(SRC) == EXPECTED_EVAL_SHA,
        "T82_EVALUATOR_HASH_MISMATCH"
    )

    status = json.loads(
        (
            Path(os.environ["T82_JOB"])
            / "status.json"
        ).read_text()
    )

    ckpt = (
        Path(os.environ["T82_JOB"])
        / "checkpoints"
        / status["checkpoint"]
    )

    check(
        ckpt.is_file()
        and sha(ckpt) == EXPECTED_MODEL_SHA
        and status["checkpoint_sha256"]
            == EXPECTED_MODEL_SHA
        and int(status["frames"]) == 190491,
        "T82_FINAL_MODEL_CHANGED"
    )

    # Confirm EGL rendering and MP4 writing are available.
    import mujoco
    import cv2

    check(
        hasattr(mujoco, "Renderer"),
        "MUJOCO_RENDERER_UNAVAILABLE"
    )

    check(
        hasattr(cv2, "VideoWriter"),
        "OPENCV_VIDEO_WRITER_UNAVAILABLE"
    )

    source = SRC.read_text()
    tree = ast.parse(source)

    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "N"
        and n.func.attr == "step_checked"
    ]

    check(
        len(matches) == 1,
        "NATIVE_STEP_CALL_NOT_UNIQUE"
    )

    call = matches[0]

    check(
        len(call.args) == 2
        and all(
            isinstance(a, ast.Name)
            for a in call.args
        )
        and [
            a.id for a in call.args
        ] == ["env", "action"]
        and not call.keywords,
        "NATIVE_STEP_ARGUMENTS_UNEXPECTED"
    )

    old = ast.get_source_segment(
        source, call
    )

    check(
        source.count(old) == 1,
        "STEP_SOURCE_ANCHOR_NOT_UNIQUE"
    )

    source = source.replace(
        old,
        "_t83_capture("
        "N.step_checked, env, action, locals())",
        1,
    )

    check(
        source.count("def main():") == 1,
        "MAIN_ENTRY_NOT_UNIQUE"
    )

    source = source.replace(
        "def main():",
        HELPER + "\n\ndef main():",
        1,
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    ast.parse(source)

    # Only observer instrumentation + output path.
    output = JOB / "evaluate_video.py"

    check(
        not output.exists(),
        "T83_EVALUATOR_ALREADY_EXISTS"
    )

    output.write_text(source)

    report = {
        "experiment": "STAGE_TD3_T83_VIDEO",
        "source_eval_sha256": EXPECTED_EVAL_SHA,
        "source_model_sha256": EXPECTED_MODEL_SHA,
        "video_eval_sha256": sha(output),
        "tasks": [2029654801, 2029654802],
        "training": False,
        "action_override": False,
        "original_tasks_unchanged": True,
        "original_models_unchanged": True,
    }

    (JOB / "capture_setup.json").write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print("T83_ORIGINAL_EVALUATOR=PASS")
    print("T83_T82_CHECKPOINT=PASS")
    print("T83_NATIVE_STEP_OBSERVER=PASS")
    print("T83_EVALUATOR_SHA256=", sha(output))
    print("T83_PREFLIGHT=PASS")

except Exception as error:
    print(
        "T83_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
