from pathlib import Path
import ast
import hashlib
import os
import traceback

H = Path.home()
JOB = Path(os.environ["T91A_JOB"])

SRC = H / (
    "mkmini_stage_td3_t90s_steer_regularization_"
    "20261028_kPT54Q/native_eval_EaguFO/evaluate.py"
)

EXPECTED_SHA = (
    "085f159ed672a843a49960f1f03f918c6e17d49"
    "c951f83d112efbeda3d40a4d5"
)

MODEL_SHA = (
    "0f2490c418ee4905093924031987dd79d2e066"
    "a05f89553563562fe2bd44261f"
)

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

HELPER = r'''
import csv as _t91_csv
import math as _t91_math
import os as _t91_os
import atexit as _t91_atexit
from pathlib import Path as _t91_Path

_T91_ROOT = (
    _t91_Path(_t91_os.environ["T91A_JOB"])
    / "traces"
)

_T91_ROOT.mkdir(parents=True, exist_ok=True)

_T91_FILES = {}
_T91_WRITERS = {}
_T91_COUNT = {}

_T91_SEEDS = {2029654801, 2029654802}

def _t91_close():
    for f in _T91_FILES.values():
        try:
            f.flush()
            f.close()
        except Exception as e:
            print("T91_CLOSE_WARNING=", repr(e))

_t91_atexit.register(_t91_close)

def _t91_step(native_step, env, action, context):

    # The original action goes directly to the original
    # native MuJoCo stepping function, unchanged.
    result = native_step(env, action)

    if context.get("name") != "T90S":
        return result

    try:
        seed = int(context.get("seed", -1))
    except (TypeError, ValueError):
        return result

    if seed not in _T91_SEEDS:
        return result

    n = _T91_COUNT.get(seed, 0) + 1
    _T91_COUNT[seed] = n

    if n != 1 and n % 60 != 0:
        return result

    if seed not in _T91_WRITERS:

        path = (
            _T91_ROOT
            / f"T90S_seed{seed}.csv"
        )

        f = path.open("w", newline="")
        writer = _t91_csv.writer(f)

        writer.writerow([
            "step",
            "x_m",
            "y_m",
            "yaw_rad",
            "speed_mps",
            "yaw_rate",
            "goal_dist_m",
            "goal_angle_rad",
            "throttle",
            "steer",
            "lidar_min_m",
        ] + [
            f"lidar_{i:02d}" for i in range(48)
        ])

        _T91_FILES[seed] = f
        _T91_WRITERS[seed] = writer

        print(
            "T91_TRACE_STARTED",
            "SEED=", seed,
            "FILE=", path,
            flush=True
        )

    x, y, yaw, vx, vy, yaw_rate = env._pose()
    features, distance, angle = env._goal_state()

    lidar = [float("nan")] * 48

    try:
        obs = result[0]

        if len(obs) == 112:
            lidar = [
                float(obs[i]) for i in range(48)
            ]
    except (TypeError, ValueError, IndexError):
        pass

    finite_lidar = [
        v for v in lidar
        if _t91_math.isfinite(v)
    ]

    lidar_min = (
        6.0 * min(finite_lidar)
        if finite_lidar
        else float("nan")
    )

    _T91_WRITERS[seed].writerow([
        n,
        float(x),
        float(y),
        float(yaw),
        _t91_math.hypot(float(vx), float(vy)),
        float(yaw_rate),
        float(distance),
        float(angle),
        float(action[0]),
        float(action[1]),
        lidar_min,
    ] + lidar)

    if n % 1200 == 0:

        _T91_FILES[seed].flush()

        print(
            "T91_PROGRESS",
            "SEED=", seed,
            "STEP=", n,
            "DIST=", round(float(distance), 4),
            "GOAL_ANGLE_DEG=",
            round(_t91_math.degrees(angle), 2),
            "STEER=", round(float(action[1]), 3),
            "THROTTLE=", round(float(action[0]), 3),
            "SPEED_MPS=",
            round(
                _t91_math.hypot(
                    float(vx), float(vy)
                ), 4
            ),
            "LIDAR_MIN_M=",
            round(lidar_min, 3),
            flush=True
        )

    return result
'''

try:
    print("=" * 76)
    print("T91A STAGE TD3 NATIVE FAILURE TELEMETRY")
    print("=" * 76)

    check(
        SRC.is_file(),
        "VERIFIED_EVALUATOR_MISSING"
    )

    source = SRC.read_text()

    source_sha = hashlib.sha256(
        SRC.read_bytes()
    ).hexdigest()

    check(
        source_sha == EXPECTED_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    check(
        MODEL_SHA in source,
        "T90S_MODEL_IDENTITY_INVALID"
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
        "NATIVE_STEP_NOT_UNIQUE"
    )

    call = calls[0]

    old = ast.get_source_segment(
        source, call
    )

    check(
        old == "N.step_checked(env, action)"
        and source.count(old) == 1,
        "NATIVE_STEP_SIGNATURE_CHANGED"
    )

    source = source.replace(
        old,
        "_t91_step("
        "N.step_checked, env, action, locals())",
        1
    )

    check(
        source.count("def main():") == 1,
        "MAIN_ENTRY_INVALID"
    )

    source = source.replace(
        "def main():",
        HELPER + "\n\ndef main():",
        1
    )

    ast.parse(source)

    dst = JOB / "evaluate_trace.py"

    check(
        not dst.exists(),
        "TRACE_EVALUATOR_ALREADY_EXISTS"
    )

    dst.write_text(source)

    print("T91_SOURCE_SHA=PASS")
    print("T91_NATIVE_STEP=UNCHANGED")
    print("T91_ACTION_OVERRIDE=FALSE")
    print("T91_VIDEO_RECORDING=FALSE")
    print("T91_PREFLIGHT=PASS")

except Exception as e:
    print("T91_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
