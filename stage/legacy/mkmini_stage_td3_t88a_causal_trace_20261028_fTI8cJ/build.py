from pathlib import Path
import ast
import hashlib
import os
import traceback

H = Path.home()
JOB = Path(os.environ["T88A_JOB"])

SOURCE = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j/"
    "pilot4_verified_UhtIxl/evaluate.py"
)

EXPECTED_EVAL_SHA = (
    "3d23dad58d7c84cc8ecb47a4df09fc8005ca53"
    "a0a9718ef26ae7222621fae276"
)

EXPECTED_MODEL_SHA = (
    "06fad7346c51c4055aff8e867d0e956f38cb9f"
    "eef68009015212fe7d96f6342c"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def replace_top(source, name, expression):
    tree = ast.parse(source)

    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == name
            for t in n.targets
        )
    ]

    check(
        len(nodes) == 1,
        "BINDING_INVALID:" + name
    )

    n = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

HELPER = r'''
# T88A: Read-only observer of native Stage TD3 execution.
# No changes to actor action or MuJoCo physics.

import csv as _t88_csv
import math as _t88_math
import atexit as _t88_atexit
import os as _t88_os
from pathlib import Path as _t88_Path

_T88_ROOT = _t88_Path(
    _t88_os.environ["T88A_JOB"]
) / "traces"

_T88_ROOT.mkdir(parents=True, exist_ok=True)

_T88_FILES = {}
_T88_WRITERS = {}
_T88_COUNT = {}

_T88_SEEDS = (
    2029653601,
    2029653602,
    2029654801,
    2029654802,
)

def _t88_close():
    for f in _T88_FILES.values():
        try:
            f.flush()
            f.close()
        except Exception as error:
            print(
                "T88A_CLOSE_WARNING=",
                repr(error),
                flush=True
            )

_t88_atexit.register(_t88_close)

def _t88_observe(native_step, env, action, context):

    # Exactly one original native physical step.
    result = native_step(env, action)

    if context.get("name") != "T87":
        return result

    try:
        seed = int(context.get("seed", -1))
    except (ValueError, TypeError):
        return result

    if seed not in _T88_SEEDS:
        return result

    n = _T88_COUNT.get(seed, 0) + 1
    _T88_COUNT[seed] = n

    if n != 1 and n % 60 != 0:
        return result

    if seed not in _T88_WRITERS:
        path = (
            _T88_ROOT
            / f"T87_seed{seed}.csv"
        )

        f = path.open("w", newline="")

        writer = _t88_csv.writer(f)

        writer.writerow([
            "physics_step",
            "x_m",
            "y_m",
            "yaw_rad",
            "goal_distance_m",
            "goal_angle_rad",
            "throttle",
            "steer",
            "lidar_min_m",
        ])

        _T88_FILES[seed] = f
        _T88_WRITERS[seed] = writer

        print(
            "T88A_TRACE_STARTED",
            "SEED=", seed,
            "PATH=", path,
            flush=True
        )

    x, y, yaw, vx, vy, yaw_rate = env._pose()

    features, distance, angle = env._goal_state()

    a0 = float(action[0])
    a1 = float(action[1])

    lidar_min_m = float("nan")

    try:
        obs = result[0]

        if len(obs) == 112:
            lidar_min_m = (
                6.0 * min(
                    float(v) for v in obs[:48]
                )
            )

    except (TypeError, IndexError, ValueError):
        pass

    _T88_WRITERS[seed].writerow([
        n,
        float(x),
        float(y),
        float(yaw),
        float(distance),
        float(angle),
        a0,
        a1,
        lidar_min_m,
    ])

    if n % 3000 == 0:
        _T88_FILES[seed].flush()

        print(
            "T88A_TRACE_PROGRESS",
            "SEED=", seed,
            "STEP=", n,
            "DIST=", round(float(distance), 4),
            "ANGLE_DEG=",
            round(_t88_math.degrees(angle), 2),
            "STEER=", round(a1, 4),
            flush=True
        )

    return result
'''

try:
    print("=" * 76)
    print("T88A NATIVE STAGE TD3 CAUSAL TRACE")
    print("=" * 76)

    check(
        SOURCE.is_file(),
        "VERIFIED_EVALUATOR_MISSING"
    )

    check(
        sha(SOURCE) == EXPECTED_EVAL_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    source = SOURCE.read_text()

    check(
        EXPECTED_MODEL_SHA in source,
        "T87_MODEL_BINDING_MISSING"
    )

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
        and not call.keywords
        and all(
            isinstance(a, ast.Name)
            for a in call.args
        )
        and [a.id for a in call.args]
        == ["env", "action"],
        "NATIVE_STEP_ARGUMENTS_CHANGED"
    )

    old = ast.get_source_segment(
        source, call
    )

    check(
        old is not None
        and source.count(old) == 1,
        "NATIVE_STEP_ANCHOR_INVALID"
    )

    source = source.replace(
        old,
        "_t88_observe("
        "N.step_checked, env, action, locals())",
        1
    )

    check(
        source.count("def main():") == 1,
        "MAIN_ENTRY_NOT_UNIQUE"
    )

    source = source.replace(
        "def main():",
        HELPER + "\n\ndef main():",
        1
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    ast.parse(source)

    output = JOB / "evaluate_trace.py"

    check(
        not output.exists(),
        "OBSERVER_ALREADY_EXISTS"
    )

    output.write_text(source)

    print("T88A_ORIGINAL_EVALUATOR=PASS")
    print("T88A_T87_MODEL_BINDING=PASS")
    print("T88A_NATIVE_ACTION_UNCHANGED=PASS")
    print("T88A_OBSERVER_SHA=", sha(output))
    print("T88A_PREFLIGHT=PASS")

except Exception as error:
    print(
        "T88A_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
