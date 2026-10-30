from pathlib import Path
import ast
import hashlib
import os
import re
import textwrap

ROOT = Path(os.environ["T72_JOB"])
T71 = Path(os.environ["T71_JOB"])

SOURCE = T71 / "T71B_pilot4/evaluate.py"

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

require(SOURCE.is_file(), "ORIGINAL_EVALUATOR_NOT_FOUND")

status = ast.literal_eval(
    repr(__import__("json").loads(
        (T71 / "T71B/status.json").read_text()
    ))
)

checkpoint = (
    T71 / "T71B/checkpoints" / status["checkpoint"]
)

require(
    checkpoint.is_file()
    and sha(checkpoint) == status["checkpoint_sha256"],
    "FROZEN_CHECKPOINT_INVALID"
)

original = SOURCE.read_text()
ast.parse(original)

require(
    "T71B" in original,
    "T71B_EVALUATOR_IDENTITY_MISSING"
)

def set_top_assignment(source, name, expression):
    tree = ast.parse(source)

    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]

    require(
        len(nodes) == 1,
        "ASSIGNMENT_NOT_UNIQUE:" + name
    )

    node = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[node.lineno - 1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

def inject_viewer(source):
    loop = re.search(
        r"(?m)^(?P<indent>[ ]*)"
        r"for frame in range\(1,\s*CAP\s*\+\s*1\):[ ]*$",
        source
    )

    require(loop is not None, "FRAME_LOOP_NOT_FOUND")

    indent = loop.group("indent")

    code = """
_t72_viewer = None
_t72_watch = (name == "T71B")

if _t72_watch:
    import mujoco
    import mujoco.viewer

    print(
        "T72_VIEWER_OPEN MODEL=T71B ANGLE=",
        angle,
        flush=True
    )

    _t72_viewer = mujoco.viewer.launch_passive(
        env.model,
        env.data,
        show_left_ui=False,
        show_right_ui=False
    )

    try:
        with _t72_viewer.lock():
            _t72_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            _t72_viewer.cam.distance = 5.8
            _t72_viewer.cam.azimuth = 135.0
            _t72_viewer.cam.elevation = -45.0

            if "start" in locals() and "goal" in locals():
                _t72_viewer.cam.lookat[:] = [
                    (float(start[0]) + float(goal[0])) / 2.0,
                    (float(start[1]) + float(goal[1])) / 2.0,
                    0.15
                ]

        _t72_viewer.sync()

    except Exception:
        _t72_viewer.close()
        raise
"""

    insertion = textwrap.indent(
        textwrap.dedent(code).strip() + "\n",
        indent
    )

    source = (
        source[:loop.start()]
        + insertion
        + source[loop.start():]
    )

    step = re.search(
        r"(?m)^(?P<indent>[ ]*)"
        r"next_obs, _, terminated, truncated, info = \(\n"
        r"[ ]*N\.step_checked\(env, action\)\n"
        r"[ ]*\)\n",
        source
    )

    require(step is not None, "NATIVE_STEP_NOT_FOUND")

    indent = step.group("indent")

    code = """
if _t72_viewer is not None:
    if not _t72_viewer.is_running():
        raise RuntimeError("T72_VIEWER_CLOSED_BY_USER")

    if frame % 20 == 0:
        _t72_viewer.sync()

        _t72_dt = float(env.model.opt.timestep)
        _t72_speed = max(
            0.1,
            float(os.environ.get("T72_SPEED", "1.0"))
        )

        time.sleep(min(
            0.20,
            20.0 * _t72_dt / _t72_speed
        ))

    if frame % 500 == 0:
        print(
            "T72_VISUAL_PROGRESS",
            "FRAME=", frame,
            "REMAINING_M=",
            round(float(env._goal_state()[1]), 4),
            "THROTTLE=", round(float(throttle), 4),
            "STEER=", round(float(steer), 4),
            flush=True
        )
"""

    insertion = textwrap.indent(
        textwrap.dedent(code).strip() + "\n",
        indent
    )

    source = (
        source[:step.end()]
        + insertion
        + source[step.end():]
    )

    cleanup = re.search(
        r"(?m)^(?P<indent>[ ]*)finally:\n"
        r"(?P<body>[ ]*)if renderer is not None:",
        source
    )

    require(
        cleanup is not None,
        "RENDERER_CLEANUP_NOT_FOUND"
    )

    body_indent = cleanup.group("body")

    code = """
if "_t72_viewer" in locals() and _t72_viewer is not None:
    try:
        if _t72_viewer.is_running():
            _t72_viewer.sync()
            print(
                "T72_SCENE_FINISHED",
                "ANGLE=", angle,
                "REMAINING_M=",
                round(float(env._goal_state()[1]), 4),
                flush=True
            )
            time.sleep(8)
    finally:
        _t72_viewer.close()
"""

    insertion = textwrap.indent(
        textwrap.dedent(code).strip() + "\n",
        body_indent
    )

    point = cleanup.start("body")

    source = (
        source[:point]
        + insertion
        + source[point:]
    )

    ast.parse(source)
    return source

SCENES = {
    "right_collision": (2029654802, 18.8, 3.0),
    "left_timeout": (2029654801, -12.5, 3.0),
}

for name, task in SCENES.items():
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)

    result = set_top_assignment(
        original,
        "JOB",
        f"Path({str(folder)!r})"
    )

    result = set_top_assignment(
        result,
        "TASKS",
        repr([task])
    )

    result = inject_viewer(result)

    output = folder / "evaluate_visual.py"
    output.write_text(result)

    print(
        "T72_VISUAL_READY=",
        name,
        "SCRIPT=", output,
        "SHA256=", sha(output),
        flush=True
    )

print("T72_BUILD=PASS")
print("ORIGINAL_EVALUATOR_MODIFIED=FALSE")
print("FROZEN_MODEL_MODIFIED=FALSE")
print("ACTION_OVERRIDE=FALSE")
print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
