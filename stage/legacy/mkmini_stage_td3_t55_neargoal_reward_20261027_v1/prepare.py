from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import torch

H = Path.home()

JOB = (
    H / "mkmini_stage_td3_t55_neargoal_reward_20261027_v1"
)

BASE = (
    H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
)

PARENT = BASE / "checkpoints/episode_0010.pt"
TRAIN = BASE / "train.py"
CORE = BASE / "td3_core_goal_contrast.py"
PAIRS = BASE / "goal_pairs.npz"

PARENT_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0"
)

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07"
)

START = 22491
TARGET = 34491

# 父模型episode_index=10，首先执行TASKS[2]。
# 顺序：左1.2、右1.2、左3、右3。
TASKS = (
    (-20.0, 3.0),
    (20.0, 3.0),
    (-20.0, 1.2),
    (20.0, 1.2),
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(part)

    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def save_json(path, value):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


def replace_assignments(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in changes
                ):
                    check(
                        target.id not in found,
                        "DUPLICATE_ASSIGNMENT:"
                        + target.id
                    )
                    found[target.id] = node

    check(
        set(found) == set(changes),
        "MISSING_ASSIGNMENTS:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[node.lineno-1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def patch_reward(source):
    tree = ast.parse(source)

    steps = []
    appends = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):
            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "step_checked"
                and isinstance(
                    call.func.value, ast.Name
                )
                and call.func.value.id == "N"
            ):
                steps.append(node)

        if isinstance(node, ast.Expr):
            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(
                    call.func.value, ast.Name
                )
                and call.func.value.id == "replay"
            ):
                appends.append(node)

    check(
        len(steps) == 1
        and len(appends) == 1,
        "REWARD_PATCH_ANCHOR_INVALID"
    )

    step = steps[0]
    append = appends[0]

    check(
        step.end_lineno < append.lineno,
        "TRANSITION_ORDER_INVALID"
    )

    lines = source.splitlines(keepends=True)

    def indentation(node):
        line = lines[node.lineno - 1]
        return line[:len(line) - len(line.lstrip())]

    step_indent = indentation(step)
    append_indent = indentation(append)

    before_step = [
        step_indent
        + "_t55_before_m = float(env._goal_state()[1])\n"
    ]

    reward_code = """
_t55_after_m = float(env._goal_state()[1])

if not (
    math.isfinite(_t55_before_m)
    and math.isfinite(_t55_after_m)
):
    raise RuntimeError("T55_DISTANCE_INVALID")

_t55_near = max(
    0.0,
    min(1.0, (1.0 - _t55_after_m) / 0.8)
)

_t55_progress = max(
    -0.01,
    min(
        0.01,
        _t55_before_m - _t55_after_m
    )
)

# 训练奖励：临近目标仍鼓励距离缩小。
_t55_shaping = (
    35.0 * _t55_near * _t55_progress
)

# 目标附近驶离时增加负反馈。
if _t55_progress < 0.0:
    _t55_shaping += (
        20.0 * _t55_near * _t55_progress
    )

# 对近目标持续大油门作轻量惩罚。
# 不直接改变任何推理动作。
_t55_shaping -= (
    0.004
    * _t55_near
    * float(action[0]) ** 2
)

reward = float(reward) + float(_t55_shaping)

if (
    _t55_near > 0.0
    and not getattr(
        agent, "_t55_reward_announced", False
    )
):
    agent._t55_reward_announced = True

    print(
        "T55_NEAR_GOAL_REWARD_ACTIVE=PASS",
        "BEFORE_M=",
        round(_t55_before_m, 5),
        "AFTER_M=",
        round(_t55_after_m, 5),
        "SHAPING=",
        round(_t55_shaping, 7),
        flush=True
    )
"""

    reward_lines = [
        append_indent + line + "\n"
        if line else "\n"
        for line in reward_code.strip().splitlines()
    ]

    inserts = {
        step.lineno - 1: before_step,
        append.lineno - 1: reward_lines,
    }

    for position in sorted(inserts, reverse=True):
        lines[position:position] = inserts[position]

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count(
            "T55_NEAR_GOAL_REWARD_ACTIVE=PASS"
        ) == 1,
        "REWARD_PATCH_NOT_UNIQUE"
    )

    return result


def main():
    print("T55_STAGE_ONLY=TRUE")
    print("T55_PARENT_FROZEN=TRUE")
    print("T55_PLANNER=FALSE")
    print("T55_ACTION_OVERRIDE=FALSE")
    print("T55_GOAL_RADIUS_MODIFIED=FALSE")
    print("T55_OBSTACLES_MODIFIED=FALSE")

    check(
        PARENT.is_file()
        and sha(PARENT) == PARENT_SHA,
        "T53_CHECKPOINT_CHANGED"
    )

    check(
        CORE.is_file()
        and sha(CORE) == CORE_SHA,
        "T53_CORE_CHANGED"
    )

    parent_manifest = json.loads(
        (BASE / "prepare.json").read_text()
    )

    for name, digest in parent_manifest["files"].items():
        path = BASE / name

        check(
            path.is_file()
            and sha(path) == digest,
            "T53_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and int(parent["frames"]) == START
        and int(parent["episode_index"]) == 10
        and parent["manifest"]["source_core"]
            == CORE_SHA,
        "T53_PARENT_METADATA_INVALID"
    )

    source = replace_assignments(
        TRAIN.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 3000",
            "WALL": "WALL = 2400",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    source = patch_reward(source)

    old_tag = '"experiment": "STAGE_TD3_T53"'
    new_tag = '"experiment": "STAGE_TD3_T55"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_TAG_INVALID"
    )

    source = source.replace(
        old_tag, new_tag, 1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": CORE.read_bytes(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    manifest = {
        "experiment": "STAGE_TD3_T55",
        "parent_sha256": PARENT_SHA,
        "source_core_sha256": CORE_SHA,
        "initial_frames": START,
        "target_frames": TARGET,
        "episode_cap": 3000,
        "tasks": [list(x) for x in TASKS],
        "near_goal_reward_enabled": True,
        "success_radius_unchanged": True,
        "obstacles_unchanged": True,
        "automatic_promotion": False,
        "files": hashes,
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():
            check(
                (JOB / name).is_file()
                and sha(JOB / name) == digest,
                "EXISTING_SOURCE_CHANGED:" + name
            )

        check(
            status_path.is_file(),
            "EXISTING_STATUS_MISSING"
        )

        status = json.loads(
            status_path.read_text()
        )

        checkpoint = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint)
                == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T55_EXISTING_BRANCH=VERIFIED")

    else:
        check(
            not status_path.exists(),
            "PARTIAL_STATUS_FOUND"
        )

        for name in files:
            check(
                not (JOB / name).exists(),
                "PARTIAL_SOURCE_FOUND:" + name
            )

        checkpoint_dir = JOB / "checkpoints"

        check(
            not (
                checkpoint_dir / "episode_0010.pt"
            ).exists(),
            "PARTIAL_CHECKPOINT_FOUND"
        )

        for name, data in files.items():
            (JOB / name).write_bytes(data)

        for name, digest in hashes.items():
            check(
                sha(JOB / name) == digest,
                "GENERATED_SHA_INVALID:" + name
            )

        branch = copy.deepcopy(parent)
        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T55"
        )

        branch["manifest"]["target_frames"] = TARGET

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = (
            checkpoint_dir / "episode_0010.pt"
        )

        temporary = (
            checkpoint_dir / "parent.partial"
        )

        torch.save(branch, temporary)
        os.replace(temporary, checkpoint)

        save_json(
            status_path,
            {
                "state": "RUNNING",
                "frames": START,
                "target_frames": TARGET,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        save_json(manifest_path, manifest)

        print("T55_BRANCH_CREATED=TRUE")

    print("T55_NEW_FRAMES=", TARGET - START)
    print("T55_EPISODE_CAP=3000")
    print("T55_TASKS=", TASKS)
    print("T55_REWARD_SOURCE_PATCH=PASS")
    print("T55_READY=TRUE")


try:
    main()

except Exception as error:
    print("T55_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
