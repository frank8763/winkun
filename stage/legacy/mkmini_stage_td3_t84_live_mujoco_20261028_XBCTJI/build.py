from pathlib import Path
import os
import ast
import json
import hashlib
import traceback

ROOT = Path(os.environ["T82_JOB"])
JOB = Path(os.environ["T84_JOB"])
SRC = ROOT / "pilot4/evaluate.py"

EXPECTED_SOURCE_SHA = (
    "aea31b20c9f92ece56010360dce8f222a8ba5693"
    "b356ce04bb5715ffd3d1fa18"
)

EXPECTED_MODEL_SHA = (
    "bb0ad57b18a48d72abe6bc4d740e82c407935"
    "c587ce6e2e16d43e9a50609b7f2"
)

TASKS = [
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()

def check(value, message):
    if not value:
        raise RuntimeError(message)

def replace_top(source, name, expression):
    tree = ast.parse(source)

    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in node.targets
        )
    ]

    check(
        len(nodes) == 1,
        "TOP_BINDING_INVALID:" + name
    )

    node = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

GUI_CODE = r'''
# T84: live native MuJoCo viewer.
# This code observes the native simulation.
# It never replaces Actor actions or physics stepping.

import mujoco as _t84_mj
import mujoco.viewer as _t84_viewer
import time as _t84_time
import numpy as _t84_np
import atexit as _t84_atexit

_T84_GOALS = {
    2029654801: (3.3623608142541768, -6.529879958546968),
    2029654802: (3.6920169243256953, 0.9259739884941749),
}

_T84_STARTS = {
    2029654801: (1.75, -4.0),
    2029654802: (1.0, 2.25),
}

_T84_HANDLE = None
_T84_CURRENT = None
_T84_COUNT = 0
_T84_PAUSED = {"value": False}
_T84_DISABLED = set()

def _t84_keyboard(key):
    if key == 32:
        _T84_PAUSED["value"] = not _T84_PAUSED["value"]
        print(
            "T84_PAUSED=",
            _T84_PAUSED["value"],
            flush=True
        )

def _t84_close():
    global _T84_HANDLE

    handle = _T84_HANDLE
    _T84_HANDLE = None

    if handle is not None:
        try:
            handle.close()
        except Exception as error:
            print(
                "T84_VIEW_CLOSE_WARNING=",
                repr(error),
                flush=True
            )

_t84_atexit.register(_t84_close)

def _t84_open(env, seed):
    global _T84_HANDLE
    global _T84_CURRENT
    global _T84_COUNT

    _t84_close()

    _T84_CURRENT = seed
    _T84_COUNT = 0
    _T84_PAUSED["value"] = False

    handle = _t84_viewer.launch_passive(
        env.model,
        env.data,
        key_callback=_t84_keyboard,
        show_left_ui=False,
        show_right_ui=False
    )

    start = _T84_STARTS[seed]
    goal = _T84_GOALS[seed]

    with handle.lock():
        handle.cam.lookat[:] = [
            (start[0] + goal[0]) / 2.0,
            (start[1] + goal[1]) / 2.0,
            0.0
        ]

        handle.cam.distance = 7.0
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -75.0

        # Viewer-only red target marker.
        # No changes to the MuJoCo physical model.
        scene = handle.user_scn

        if scene.maxgeom > 0:
            _t84_mj.mjv_initGeom(
                scene.geoms[0],
                _t84_mj.mjtGeom.mjGEOM_SPHERE,
                _t84_np.array([0.16, 0.16, 0.16]),
                _t84_np.array([
                    goal[0], goal[1], 0.2
                ]),
                _t84_np.eye(3).reshape(-1),
                _t84_np.array([1.0, 0.1, 0.1, 1.0])
            )

            scene.ngeom = 1

    _T84_HANDLE = handle

    print(
        "T84_LIVE_MUJOCO_OPEN=PASS",
        "SEED=", seed,
        "ANGLE=",
        -12.5 if seed == 2029654801 else 18.8,
        flush=True
    )

def _t84_live_step(native_step, env, action, context):
    global _T84_COUNT

    name = context.get("name")
    seed = context.get("seed")

    # Observing T82 only, never T53.
    selected = (
        name == "T82"
        and seed in _T84_GOALS
        and seed not in _T84_DISABLED
    )

    if selected and _T84_CURRENT != seed:
        try:
            _t84_open(env, seed)
        except Exception as error:
            print(
                "T84_GUI_OPEN_FAILED=",
                repr(error),
                flush=True
            )
            _T84_DISABLED.add(seed)
            _t84_close()

    if selected and _T84_HANDLE is not None:
        while (
            _T84_PAUSED["value"]
            and _T84_HANDLE.is_running()
        ):
            _T84_HANDLE.sync()
            _t84_time.sleep(0.05)

    # Original TD3 action and original physics execution.
    result = native_step(env, action)

    if selected and _T84_HANDLE is not None:
        _T84_COUNT += 1

        if _T84_HANDLE.is_running():
            if _T84_COUNT % 20 == 0:
                _T84_HANDLE.sync()
                _t84_time.sleep(0.07)
        else:
            print(
                "T84_WINDOW_CLOSED_BY_USER",
                "SEED=", seed,
                flush=True
            )
            _T84_DISABLED.add(seed)
            _t84_close()

    return result
'''

try:
    print("=" * 76)
    print("T84 — LIVE MUJOCO TWO FAILURE SCENES")
    print("=" * 76)

    check(
        os.environ.get("MUJOCO_GL") == "glfw",
        "GUI_GL_BACKEND_NOT_GLFW"
    )

    check(
        bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY")),
        "DESKTOP_DISPLAY_NOT_AVAILABLE"
    )

    import mujoco
    import mujoco.viewer

    check(
        SRC.is_file()
        and sha(SRC) == EXPECTED_SOURCE_SHA,
        "ORIGINAL_EVALUATOR_CHANGED"
    )

    status = json.loads(
        (ROOT / "status.json").read_text()
    )

    ck = ROOT / "checkpoints" / status["checkpoint"]

    check(
        ck.is_file()
        and sha(ck) == EXPECTED_MODEL_SHA
        and status["checkpoint_sha256"]
            == EXPECTED_MODEL_SHA
        and int(status["frames"]) == 190491,
        "T82_CHECKPOINT_INVALID"
    )

    source = SRC.read_text()

    # Only use the two original failed seeds.
    original_tree = ast.parse(source)

    tasks_nodes = [
        n for n in original_tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "TASKS"
            for t in n.targets
        )
    ]

    check(
        len(tasks_nodes) == 1,
        "ORIGINAL_TASKS_UNAVAILABLE"
    )

    original_tasks = ast.literal_eval(
        tasks_nodes[0].value
    )

    check(
        original_tasks == [
            (2029653601, -17.3, 3.0),
            (2029653602, 13.6, 3.0),
            (2029654801, -12.5, 3.0),
            (2029654802, 18.8, 3.0)
        ],
        "ORIGINAL_TASKS_CHANGED"
    )

    tree = ast.parse(source)

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "N"
        and n.func.attr == "step_checked"
    ]

    check(
        len(calls) == 1,
        "NATIVE_STEP_CALL_NOT_UNIQUE"
    )

    old = ast.get_source_segment(
        source, calls[0]
    )

    check(
        old == "N.step_checked(\n                    env,\n                    action\n                )"
        or (
            "N.step_checked" in old
            and len(calls[0].args) == 2
            and all(
                isinstance(a, ast.Name)
                for a in calls[0].args
            )
            and [
                a.id for a in calls[0].args
            ] == ["env", "action"]
        ),
        "NATIVE_STEP_ARGUMENTS_CHANGED"
    )

    check(
        source.count(old) == 1,
        "NATIVE_STEP_ANCHOR_INVALID"
    )

    source = source.replace(
        old,
        "_t84_live_step("
        "N.step_checked, env, action, locals())",
        1
    )

    check(
        source.count("def main():") == 1,
        "MAIN_ENTRY_INVALID"
    )

    source = source.replace(
        "def main():",
        GUI_CODE + "\n\ndef main():",
        1
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    source = replace_top(
        source,
        "TASKS",
        repr(TASKS)
    )

    check(
        "T82" in source
        and str(ck) in source
        and EXPECTED_MODEL_SHA in source,
        "T82_MODEL_BINDING_CHANGED"
    )

    ast.parse(source)

    output = JOB / "live_mujoco.py"

    check(
        not output.exists(),
        "T84_ALREADY_PREPARED"
    )

    output.write_text(source)

    print("T84_T82_MODEL=PASS")
    print("T84_NATIVE_STEP_UNCHANGED=PASS")
    print("T84_TWO_FAILURE_SEEDS=PASS")
    print("T84_GUI_BACKEND=GLFW")
    print("T84_PREFLIGHT=PASS")

except Exception as e:
    print("T84_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
