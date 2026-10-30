from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
J = H / "mkmini_stage_td3_t49_random3m_focus_20261027_v1"
P = H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"

SRC = P / "train.py"
CORE = P / "td3_core_goal_contrast.py"
PAIRS = P / "goal_pairs.npz"
BASE = P / "checkpoints/episode_0009.pt"

BASE_HASH = (
    "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e"
)

CORE_HASH = (
    "b28b45562d79dff52090a24e6b21a74bcd24d830ab89a706cb46467df8d531ea"
)

# 新训练任务全部为3米。
# 左右目标方向交替，起点仍由Stage原生任务生成器确定。
TASKS = (
    (-10.0, 3.0),
    (10.0, 3.0),
    (-20.0, 3.0),
    (20.0, 3.0),
    (-30.0, 3.0),
    (30.0, 3.0),
    (-40.0, 3.0),
    (40.0, 3.0),
)

START = 16491
TARGET = 28491


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for part in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(part)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def write_json(path, obj):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


def patch_assignments(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in changes:
                    check(
                        target.id not in found,
                        "DUPLICATE:" + target.id
                    )
                    found[target.id] = node

    check(
        set(found) == set(changes),
        "ASSIGNMENTS_MISSING:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("T49_STAGE_ONLY=TRUE")
    print("T49_PLANNER=FALSE")
    print("T49_ACTION_OVERRIDE=FALSE")
    print("T49_PARENT_FROZEN=TRUE")

    check(
        BASE.is_file()
        and sha(BASE) == BASE_HASH,
        "T46R_CHECKPOINT_CHANGED"
    )

    check(
        CORE.is_file()
        and sha(CORE) == CORE_HASH,
        "T46R_CORE_CHANGED"
    )

    original = json.loads(
        (P / "prepare.json").read_text()
    )

    for name, digest in original["files"].items():
        check(
            (P / name).is_file()
            and sha(P / name) == digest,
            "TRAINER_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        BASE,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and parent.get("frames") == START
        and parent.get("episode_index") == 9
        and parent["manifest"]["source_core"] == CORE_HASH,
        "T46R_METADATA_INVALID"
    )

    source = patch_assignments(
        SRC.read_text(),
        {
            "JOB": f"JOB = Path({str(J)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "WALL": "WALL = 4800",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    old_tag = '"experiment": "STAGE_TD3_T46R"'
    new_tag = '"experiment": "STAGE_TD3_T49S"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_ANCHOR_INVALID"
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
        "experiment": "STAGE_TD3_T49S",
        "source_parent_sha256": BASE_HASH,
        "source_core_sha256": CORE_HASH,
        "initial_frames": START,
        "target_frames": TARGET,
        "tasks": [list(item) for item in TASKS],
        "files": hashes,
        "automatic_promotion": False,
        "evaluation_required": True,
    }

    manifest_path = J / "prepare.json"
    status_path = J / "status.json"

    if manifest_path.exists():

        check(
            json.loads(manifest_path.read_text()) == manifest,
            "EXISTING_PREPARE_CHANGED"
        )

        for name, digest in hashes.items():
            check(
                (J / name).is_file()
                and sha(J / name) == digest,
                "EXISTING_SOURCE_CHANGED:" + name
            )

        status = json.loads(status_path.read_text())

        checkpoint = (
            J / "checkpoints"
            / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint)
            == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CORRUPT"
        )

        print("T49_EXISTING_BRANCH=VERIFIED")

    else:

        for name in files:
            check(
                not (J / name).exists(),
                "PARTIAL_FILE_FOUND:" + name
            )

        check(
            not status_path.exists(),
            "PARTIAL_STATUS_FOUND"
        )

        ckdir = J / "checkpoints"
        ckdir.mkdir(parents=True, exist_ok=True)

        check(
            not (ckdir / "episode_0009.pt").exists(),
            "PARTIAL_PARENT_FOUND"
        )

        for name, data in files.items():
            (J / name).write_bytes(data)

        for name, digest in hashes.items():
            check(
                sha(J / name) == digest,
                "COPY_FAILED:" + name
            )

        branch = copy.deepcopy(parent)

        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T49S"
        )

        branch["manifest"]["target_frames"] = TARGET

        checkpoint = ckdir / "episode_0009.pt"
        temporary = ckdir / "parent.partial"

        torch.save(branch, temporary)
        os.replace(temporary, checkpoint)

        write_json(
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

        write_json(manifest_path, manifest)

        print("T49_BRANCH_CREATED=TRUE")

    print("T49_INITIAL_FRAMES=", START)
    print("T49_TARGET_FRAMES=", TARGET)
    print("T49_NEW_FRAMES=", TARGET - START)
    print("T49_THREE_METER_TASKS=", len(TASKS))
    print("T49_READY=TRUE")


try:
    main()

except Exception as error:
    print("T49_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
