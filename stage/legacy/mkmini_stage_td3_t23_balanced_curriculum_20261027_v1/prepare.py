from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import shutil
import traceback

import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t23_balanced_curriculum_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

SOURCE = T16 / "train.py"
CORE = T16 / "td3_core_goal_contrast.py"
PAIRS = T16 / "goal_pairs.npz"
PARENT = T16 / "checkpoints/episode_0008.pt"
PREP = T16 / "prepare.json"

FROZEN = (
    H / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
)

EXPECTED_PARENT = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

EXPECTED_FROZEN = (
    "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
)

INITIAL = 13491
TARGET = 37491
EP_CAP = 6000
WALL = 4500

TASKS = (
    (-20, 1.2),
    (20, 1.2),
    (-20, 1.5),
    (20, 1.5),
)


def sha(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            digest.update(part)

    return digest.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def write_json(path, data):
    temp = Path(str(path) + ".partial")

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


def patch_assignment(source, replacement):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    nodes = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in replacement
            ):
                check(
                    target.id not in nodes,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )
                nodes[target.id] = node

    for name in replacement:
        check(
            name in nodes,
            "MISSING_ASSIGNMENT:" + name
        )

    for name, node in sorted(
        nodes.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [replacement[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("STAGE_ONLY=TRUE")
    print("NO_EXTERNAL_PLANNER=TRUE")
    print("NO_WAYPOINTS=TRUE")
    print("NO_ACTION_OVERRIDE=TRUE")
    print("FROZEN_HISTORY_WRITE=FALSE")

    for path in (SOURCE, CORE, PAIRS, PARENT, PREP, FROZEN):
        check(path.is_file(), "MISSING_FILE:" + str(path))

    check(
        sha(PARENT) == EXPECTED_PARENT,
        "PARENT_CHECKPOINT_CHANGED"
    )

    check(
        sha(FROZEN) == EXPECTED_FROZEN,
        "FROZEN_REFERENCE_CHANGED"
    )

    original_prepare = json.loads(PREP.read_text())

    for name, expected in original_prepare["files"].items():
        path = T16 / name

        check(
            path.is_file() and sha(path) == expected,
            "T16_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3",
        "PARENT_ALGORITHM_INVALID"
    )

    check(
        int(parent["frames"]) == INITIAL
        and int(parent["updates"]) == 6618
        and int(parent["episode_index"]) == 8,
        "PARENT_RESUME_POSITION_INVALID"
    )

    original_manifest = dict(parent["manifest"])

    check(
        original_manifest["experiment"] == "STAGE_TD3_T16",
        "PARENT_EXPERIMENT_INVALID"
    )

    check(
        int(original_manifest["target_frames"]) == 20000,
        "PARENT_TARGET_INVALID"
    )

    check(
        original_manifest["source_core"] == sha(CORE),
        "PARENT_CORE_PROVENANCE_INVALID"
    )

    check(
        original_manifest["goal_pair_dataset_sha256"]
        == sha(PAIRS),
        "PARENT_PAIR_PROVENANCE_INVALID"
    )

    # 修改训练入口，不修改TD3网络结构及更新算法。
    source = SOURCE.read_text()

    source = patch_assignment(
        source,
        {
            "JOB": (
                'JOB = H / '
                '"mkmini_stage_td3_t23_balanced_curriculum_20261027_v1"'
            ),
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": f"EP_CAP = {EP_CAP}",
            "WALL": f"WALL = {WALL}",
            "TASKS": (
                "TASKS = (\n"
                "    (-20, 1.2),\n"
                "    (20, 1.2),\n"
                "    (-20, 1.5),\n"
                "    (20, 1.5),\n"
                ")"
            ),
        }
    )

    old_name = '"experiment": "STAGE_TD3_T16"'
    new_name = '"experiment": "STAGE_TD3_T23"'

    check(
        source.count(old_name) == 1,
        "EXPERIMENT_PATCH_ANCHOR_INVALID"
    )

    source = source.replace(
        old_name,
        new_name,
        1
    )

    ast.parse(source)

    script_sha = hashlib.sha256(
        source.encode()
    ).hexdigest()

    prepared = JOB / "prepare.json"

    # 已建立的T23分支只允许校验，不重复初始化。
    if prepared.exists():

        existing = json.loads(prepared.read_text())

        check(
            existing["source_parent_sha256"]
            == EXPECTED_PARENT,
            "T23_PARENT_PROVENANCE_CHANGED"
        )

        for name, expected in existing["files"].items():
            path = JOB / name

            check(
                path.is_file() and sha(path) == expected,
                "T23_PREPARED_SOURCE_CHANGED:" + name
            )

        check(
            sha(JOB / "train.py") == script_sha,
            "T23_TRAIN_SCRIPT_CHANGED"
        )

        status = json.loads(
            (JOB / "status.json").read_text()
        )

        checkpoint = (
            JOB / "checkpoints" / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint) == status["checkpoint_sha256"],
            "T23_RESUME_CHECKPOINT_CHANGED"
        )

        print("T23_PREPARE=EXISTING_VERIFIED")
        print("T23_READY=TRUE")
        return

    for name in (
        "train.py",
        "td3_core_goal_contrast.py",
        "goal_pairs.npz",
        "status.json",
        "checkpoints/episode_0008.pt",
    ):
        check(
            not (JOB / name).exists(),
            "T23_PARTIAL_DIRECTORY:" + name
        )

    JOB.mkdir(parents=True, exist_ok=True)

    (JOB / "train.py").write_text(source)

    shutil.copy2(
        CORE,
        JOB / "td3_core_goal_contrast.py"
    )

    shutil.copy2(
        PAIRS,
        JOB / "goal_pairs.npz"
    )

    check(
        sha(JOB / "td3_core_goal_contrast.py") == sha(CORE),
        "CORE_COPY_FAILED"
    )

    check(
        sha(JOB / "goal_pairs.npz") == sha(PAIRS),
        "PAIR_COPY_FAILED"
    )

    new_manifest = copy.deepcopy(original_manifest)

    # 训练器仍会执行严格manifest相等校验。
    # 不加入额外字段，避免T19出现过的恢复错误。
    new_manifest["experiment"] = "STAGE_TD3_T23"
    new_manifest["target_frames"] = TARGET

    new_state = copy.deepcopy(parent)
    new_state["manifest"] = new_manifest

    checkpoint_dir = JOB / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = checkpoint_dir / "episode_0008.pt"
    temporary = checkpoint_dir / "initial.pt.partial"

    torch.save(
        new_state,
        temporary
    )

    os.replace(
        temporary,
        checkpoint
    )

    write_json(
        JOB / "status.json",
        {
            "state": "RUNNING",
            "frames": INITIAL,
            "target_frames": TARGET,
            "checkpoint": "episode_0008.pt",
            "checkpoint_sha256": sha(checkpoint),
            "automatic_promotion": False
        }
    )

    file_hashes = {
        name: sha(JOB / name)
        for name in (
            "train.py",
            "td3_core_goal_contrast.py",
            "goal_pairs.npz",
        )
    }

    write_json(
        prepared,
        {
            "experiment": "STAGE_TD3_T23",
            "source_parent_sha256": EXPECTED_PARENT,
            "original_T16_source_sha256": sha(SOURCE),
            "initial_frames": INITIAL,
            "target_frames": TARGET,
            "episode_cap": EP_CAP,
            "tasks": TASKS,
            "files": file_hashes,
            "automatic_promotion": False
        }
    )

    print("T23_NEW_BRANCH=CREATED")
    print("T23_INITIAL_FRAMES=", INITIAL)
    print("T23_TARGET_FRAMES=", TARGET)
    print("T23_ADDITIONAL_FRAMES=", TARGET - INITIAL)
    print("T23_TASKS=", TASKS)
    print("T23_CORE_UNMODIFIED=TRUE")
    print("T23_HISTORY_UNMODIFIED=TRUE")
    print("T23_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T23_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
