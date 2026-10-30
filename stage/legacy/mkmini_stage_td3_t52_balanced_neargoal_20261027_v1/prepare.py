from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import torch

H = Path.home()

J = (
    H / "mkmini_stage_td3_t52_balanced_neargoal_20261027_v1"
)

P = (
    H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"
)

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

START = 16491
TARGET = 28491

# episode_index从9开始。
# 任务顺序设计为右1.2、左3、右3、左1.2。
# 每个训练片段最多3000帧。
TASKS = (
    (-20.0, 1.2),
    (20.0, 1.2),
    (-20.0, 3.0),
    (20.0, 3.0),
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


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
    nodes = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in changes:
                    check(
                        target.id not in nodes,
                        "DUPLICATE:" + target.id
                    )

                    nodes[target.id] = node

    check(
        set(nodes) == set(changes),
        "MISSING_ASSIGNMENTS:"
        + repr(set(changes) - set(nodes))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        nodes.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)

    ast.parse(result)

    return result


def patch_near_goal_sampler(source):
    tree = ast.parse(source)
    found = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if not isinstance(target, ast.Name):
            continue

        if target.id in (
            "_left_ids",
            "_right_ids",
        ):
            check(
                target.id not in found,
                "DUPLICATE_SAMPLER:" + target.id
            )

            found[target.id] = node

    check(
        set(found) == {
            "_left_ids",
            "_right_ids",
        },
        "MIXED_REPLAY_SAMPLER_MISSING"
    )

    changes = {
        "_left_ids": (
            "_left_ids = _mix_rng.choice("
            "_t43_np.arange(8212, 10612), "
            "size=8, replace=False)"
        ),
        "_right_ids": (
            "_right_ids = _mix_rng.choice("
            "_t43_np.arange(17919, 20319), "
            "size=8, replace=False)"
        ),
    }

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        indent = " " * node.col_offset

        replacement = [
            indent + changes[name] + "\n"
        ]

        if name == "_right_ids":
            replacement.extend([
                indent
                + "if agent._t43_batches == 0:\n",
                indent
                + "    print("
                + "'T52_NEAR_GOAL_SAMPLING=PASS', "
                + "'LEFT_TAIL=2400', "
                + "'RIGHT_TAIL=2400', "
                + "flush=True)\n",
            ])

        lines[
            node.lineno - 1:node.end_lineno
        ] = replacement

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count(
            "T52_NEAR_GOAL_SAMPLING=PASS"
        ) == 1,
        "SAMPLER_PATCH_FAILED"
    )

    return result


def main():
    print("T52_STAGE_ONLY=TRUE")
    print("T52_PARENT_FROZEN=TRUE")
    print("T52_PLANNER=FALSE")
    print("T52_WAYPOINTS=FALSE")
    print("T52_ACTION_OVERRIDE=FALSE")

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

    parent_manifest = json.loads(
        (P / "prepare.json").read_text()
    )

    for name, digest in parent_manifest["files"].items():
        path = P / name

        check(
            path.is_file()
            and sha(path) == digest,
            "PARENT_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        BASE,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and int(parent["frames"]) == START
        and int(parent["episode_index"]) == 9
        and parent["manifest"]["source_core"]
            == CORE_HASH,
        "PARENT_METADATA_INVALID"
    )

    source = replace_assignments(
        SRC.read_text(),
        {
            "JOB": f"JOB = Path({str(J)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 3000",
            "WALL": "WALL = 2100",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    source = patch_near_goal_sampler(source)

    old_tag = '"experiment": "STAGE_TD3_T46R"'
    new_tag = '"experiment": "STAGE_TD3_T52"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_TAG_INVALID"
    )

    source = source.replace(
        old_tag,
        new_tag,
        1
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
        "experiment": "STAGE_TD3_T52",
        "parent_sha256": BASE_HASH,
        "core_sha256": CORE_HASH,
        "initial_frames": START,
        "target_frames": TARGET,
        "episode_cap": 3000,
        "tasks": [list(t) for t in TASKS],
        "near_goal_sampling": True,
        "success_radius_unchanged": True,
        "obstacle_geometry_unchanged": True,
        "files": hashes,
        "automatic_promotion": False,
    }

    manifest_path = J / "prepare.json"
    status_path = J / "status.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():
            check(
                (J / name).is_file()
                and sha(J / name) == digest,
                "EXISTING_SOURCE_CHANGED:" + name
            )

        check(
            status_path.is_file(),
            "STATUS_MISSING"
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
            "EXISTING_CHECKPOINT_INVALID"
        )

        print("T52_EXISTING_BRANCH=VERIFIED")

    else:
        check(
            not status_path.exists(),
            "PARTIAL_STATUS_FOUND"
        )

        for name in files:
            check(
                not (J / name).exists(),
                "PARTIAL_SOURCE_FOUND:" + name
            )

        checkpoint_dir = J / "checkpoints"

        check(
            not (
                checkpoint_dir / "episode_0009.pt"
            ).exists(),
            "PARTIAL_PARENT_CHECKPOINT"
        )

        for name, data in files.items():
            (J / name).write_bytes(data)

        for name, digest in hashes.items():
            check(
                sha(J / name) == digest,
                "GENERATED_HASH_INVALID:" + name
            )

        branch = copy.deepcopy(parent)

        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T52"
        )

        branch["manifest"]["target_frames"] = TARGET

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = (
            checkpoint_dir / "episode_0009.pt"
        )

        temporary = (
            checkpoint_dir / "parent.partial"
        )

        torch.save(branch, temporary)

        os.replace(
            temporary, checkpoint
        )

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

        save_json(
            manifest_path,
            manifest
        )

        print("T52_BRANCH_CREATED=TRUE")

    print("T52_START_FRAMES=", START)
    print("T52_TARGET_FRAMES=", TARGET)
    print("T52_ADDITIONAL_FRAMES=", TARGET - START)
    print("T52_EPISODE_CAP=3000")
    print("T52_TASKS=", TASKS)
    print("T52_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T52_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
