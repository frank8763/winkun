from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import textwrap
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T82_JOB"])

BASE = H / (
    "mkmini_stage_td3_t71_actor_ablation_"
    "20261028_dE1arY/T71B"
)

INITIAL = H / (
    "mkmini_stage_td3_t77_goal_recovery_"
    "20261028_AMyCJl/checkpoints/episode_0022.pt"
)

PARENT = H / (
    "mkmini_stage_td3_t67_3m_nightrelay_"
    "20261027_kEPAW0/T67C022"
)

TRAIN_SHA = (
    "19a03f58bba9b7122b1eb17c1f9806f9e00b75"
    "a0acc058a0a7af3eaaac410763"
)

CORE_SHA = (
    "c7e06605579a8dec7d0bd95033ae437997583"
    "3968fe69968e7294396e9b21cdc"
)

def check(value, reason):
    if not value:
        raise RuntimeError(reason)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

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
    node = found[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]
    return "".join(lines)

def formatted_lines(fragment, indent):
    fragment = textwrap.dedent(fragment).strip("\n")
    return [
        " " * indent + line + "\n"
        for line in fragment.splitlines()
    ]

def patch_replay(source):
    tree = ast.parse(source)

    appends = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "replay"
        and n.value.func.attr == "append"
    ]

    draws = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "ids"
            for t in n.targets
        )
        and isinstance(n.value, ast.Call)
        and "rng.choice" in ast.unparse(n.value)
        and "len(replay)" in ast.unparse(n.value)
    ]

    check(len(appends) == 1, "REPLAY_APPEND_NOT_UNIQUE")
    check(len(draws) == 1, "REPLAY_DRAW_NOT_UNIQUE")

    append = appends[0]
    draw = draws[0]

    check(
        append.end_lineno < draw.lineno,
        "REPLAY_CONTROL_FLOW_UNEXPECTED"
    )

    track = """
if "_t82_abs" not in locals():
    _t82_abs = len(replay) - 1
    _t82_left = deque(
        i for i, row in enumerate(replay)
        if len(row) >= 6 and row[5] == "LEFT"
    )
    _t82_right = deque(
        i for i, row in enumerate(replay)
        if len(row) >= 6 and row[5] == "RIGHT"
    )
    print(
        "T82_TRACKER_INIT",
        "LEFT=", len(_t82_left),
        "RIGHT=", len(_t82_right),
        flush=True
    )
else:
    _t82_abs += 1
    if side == "LEFT":
        _t82_left.append(_t82_abs)
    elif side == "RIGHT":
        _t82_right.append(_t82_abs)
    else:
        raise RuntimeError("T82_INVALID_SIDE")

    _t82_first = _t82_abs - len(replay) + 1

    while _t82_left and _t82_left[0] < _t82_first:
        _t82_left.popleft()

    while _t82_right and _t82_right[0] < _t82_first:
        _t82_right.popleft()
"""

    sampling = """
if len(_t82_left) >= 32 and len(_t82_right) >= 32:
    _t82_first = _t82_abs - len(replay) + 1

    _t82_l = rng.choice(
        np.asarray(_t82_left, dtype=np.int64),
        size=32,
        replace=False
    )
    _t82_r = rng.choice(
        np.asarray(_t82_right, dtype=np.int64),
        size=32,
        replace=False
    )

    ids = np.concatenate(
        (_t82_l, _t82_r)
    ) - _t82_first

    rng.shuffle(ids)
    _t82_balanced = True
else:
    ids = rng.choice(
        len(replay),
        size=64,
        replace=False
    )
    _t82_balanced = False

check(
    len(ids) == 64
    and int(np.min(ids)) >= 0
    and int(np.max(ids)) < len(replay)
    and len(np.unique(ids)) == 64,
    "T82_REPLAY_INDEX_INVALID"
)

if _t82_balanced:
    _t82_sample_left = sum(
        replay[int(i)][5] == "LEFT"
        for i in ids
    )
    check(
        _t82_sample_left == 32,
        "T82_BALANCE_GATE_FAILED"
    )

if agent.total_updates % 2000 == 0:
    print(
        "T82_REPLAY_AUDIT",
        "UPDATES=", agent.total_updates,
        "LEFT_AVAILABLE=", len(_t82_left),
        "RIGHT_AVAILABLE=", len(_t82_right),
        "BALANCED=", _t82_balanced,
        flush=True
    )
"""

    lines = source.splitlines(keepends=True)

    # 先替换靠后的采样节点，避免改变前面的行号。
    lines[draw.lineno-1:draw.end_lineno] = formatted_lines(
        sampling, draw.col_offset
    )

    # 在已有replay.append之后维护左右绝对索引。
    lines[append.end_lineno:append.end_lineno] = formatted_lines(
        track, append.col_offset
    )

    updated = "".join(lines)
    ast.parse(updated)

    check(
        "T82_BALANCE_GATE_FAILED" in updated,
        "T82_PATCH_MISSING"
    )

    return updated

def main():
    print("=" * 76)
    print("T82 STAGE TD3 BALANCED REPLAY PREFLIGHT")
    print("=" * 76)

    train_path = BASE / "train.py"
    core_path = BASE / "td3_core_goal_contrast.py"

    check(
        train_path.is_file()
        and sha(train_path) == TRAIN_SHA,
        "BASE_TRAIN_CHANGED"
    )

    check(
        core_path.is_file()
        and sha(core_path) == CORE_SHA,
        "BASE_CORE_CHANGED"
    )

    check(INITIAL.is_file(), "INITIAL_MISSING")

    initial = torch.load(
        INITIAL,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(initial["frames"]) == 134491
        and initial["manifest"]["source_core"] == CORE_SHA,
        "INITIAL_STATE_INVALID"
    )

    parent_status = json.loads(
        (PARENT / "status.json").read_text()
    )

    parent_ck = (
        PARENT / "checkpoints"
        / parent_status["checkpoint"]
    )

    check(
        parent_ck.is_file()
        and sha(parent_ck) == parent_status["checkpoint_sha256"],
        "FROZEN_PARENT_CHANGED"
    )

    parent = torch.load(
        parent_ck,
        map_location="cpu",
        weights_only=False
    )

    for name in (
        "actor",
        "critic",
        "actor_target",
        "critic_target"
    ):
        a = initial[name]
        b = parent[name]
        check(
            set(a) == set(b)
            and all(torch.equal(a[k], b[k]) for k in a),
            "INITIAL_PARAMETER_MISMATCH:" + name
        )

    source = train_path.read_text()

    check(
        source.count("STAGE_TD3_T71B") == 1,
        "EXPERIMENT_ID_UNEXPECTED"
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    source = source.replace(
        "STAGE_TD3_T71B",
        "STAGE_TD3_T82",
        1
    )

    source = patch_replay(source)

    check(
        source.count("maxlen=20000") == 2,
        "REPLAY_CAPACITY_CHANGED"
    )

    check(
        not (JOB / "train.py").exists(),
        "T82_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(source)
    (JOB / "td3_core_goal_contrast.py").write_bytes(
        core_path.read_bytes()
    )
    (JOB / "goal_pairs.npz").write_bytes(
        (BASE / "goal_pairs.npz").read_bytes()
    )

    (JOB / "checkpoints").mkdir(exist_ok=True)

    child = copy.deepcopy(initial)
    child["manifest"]["experiment"] = "STAGE_TD3_T82"
    child["manifest"]["target_frames"] = 190491

    destination = JOB / "checkpoints/episode_0022.pt"

    torch.save(child, destination)

    status = copy.deepcopy(parent_status)
    status.update({
        "state": "RUNNING",
        "frames": 134491,
        "target_frames": 190491,
        "checkpoint": destination.name,
        "checkpoint_sha256": sha(destination),
        "automatic_promotion": False
    })

    (JOB / "status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )

    manifest = {
        "experiment": "STAGE_TD3_T82",
        "baseline": "T71B",
        "start_frames": 134491,
        "target_frames": 190491,
        "single_change": "current_replay_side_balanced_sampling",
        "minibatch_size": 64,
        "left_samples": 32,
        "right_samples": 32,
        "fallback": "original_uniform_sampling",
        "replay_capacity": 20000,
        "core_sha256": sha(core_path),
        "train_sha256": sha(JOB / "train.py"),
        "initial_sha256": sha(destination),
        "automatic_promotion": False
    }

    (JOB / "experiment.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("T82_INITIAL_NETWORK=PASS")
    print("T82_ORIGINAL_CORE=PASS")
    print("T82_REPLAY_PATCH=PASS")
    print("T82_SOURCE_SHA256=", sha(JOB / "train.py"))
    print("T82_PREFLIGHT=PASS")

try:
    main()
except Exception as e:
    print("T82_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()
finally:
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
