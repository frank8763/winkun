from pathlib import Path
import ast
import hashlib
import json
import os

JOB = Path(os.environ["T75_JOB"])
T71 = Path(os.environ["T71_JOB"])
T73 = Path(os.environ["T73_JOB"])

SRC = T73 / "evaluate_trace.py"

EXPECTED_SRC = (
    "39e572da144e2e88a38c348f8fbbe6ce77fa6e33d5"
    "c270c8d1e3df793d1ef6be"
)

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

check(SRC.is_file(), "T73_SOURCE_MISSING")
check(
    sha(SRC) == EXPECTED_SRC,
    "T73_SOURCE_HASH_MISMATCH"
)

status = json.loads(
    (T71 / "T71B/status.json").read_text()
)

ck = (
    T71 / "T71B/checkpoints"
    / status["checkpoint"]
)

check(
    ck.is_file()
    and sha(ck) == status["checkpoint_sha256"]
    and int(status["frames"]) == 190491,
    "T71B_FROZEN_CHECKPOINT_CHANGED"
)

source = SRC.read_text()
ast.parse(source)

def set_assignment(code, name, expression):
    tree = ast.parse(code)
    nodes = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if any(
            isinstance(t, ast.Name) and t.id == name
            for t in node.targets
        ):
            nodes.append(node)

    check(
        len(nodes) == 1,
        "ASSIGNMENT_NOT_UNIQUE:" + name
    )

    n = nodes[0]
    lines = code.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    return "".join(lines)

source = set_assignment(
    source,
    "_T73_TRACE",
    f"_T73Path({str(JOB / 'trajectory.jsonl')!r})"
)

source = set_assignment(
    source,
    "JOB",
    f"Path({str(JOB)!r})"
)

marker = "    row = {\n"

check(
    source.count(marker) == 1,
    "TRACE_ROW_ANCHOR_MISMATCH"
)

extra = '''    _t75_live_lidar = np.asarray(
        env._lidar48(), dtype=float
    ).reshape(-1)

    if (
        _t75_live_lidar.size != 48
        or raw_obs.size != 112
    ):
        raise RuntimeError(
            "T75_OBSERVATION_DIMENSION_INVALID"
        )

    _t75_sync_error = float(
        np.max(
            np.abs(
                raw_obs[:48] - _t75_live_lidar
            )
        )
    )

    row = {
        "t75_obs_112": [
            float(v) for v in raw_obs
        ],
        "t75_native_lidar48": [
            float(v) for v in _t75_live_lidar
        ],
        "t75_lidar_sync_error": _t75_sync_error,
        "t75_native_goal_state": repr(
            env._goal_state()
        ),
'''

source = source.replace(marker, extra, 1)

ast.parse(source)

out = JOB / "evaluate.py"
out.write_text(source)

print("T75_ORIGINAL_SHA256=", sha(SRC))
print("T75_NEW_SHA256=", sha(out))
print("T75_MODEL_SHA256=", sha(ck))
print("T75_BUILD=PASS")
print("TRAINING=FALSE")
print("ACTION_OVERRIDE=FALSE")
print("FROZEN_MODEL_MODIFIED=FALSE")
print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
