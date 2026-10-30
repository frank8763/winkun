from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import random
import traceback

import torch

H = Path.home()

JOB = H / "mkmini_stage_td3_t35_random3m_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T31A = (
    H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"
    / "t31a"
)

PARENT = T16 / "checkpoints/episode_0008.pt"

SOURCE = T31A / "train.py"
CORE = T31A / "td3_core_goal_contrast.py"
PAIRS = T31A / "goal_pairs.npz"

EXPECTED_PARENT = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

EXPECTED_CORE = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

INITIAL = 13491
TARGET = 61491

EP_CAP = 12000
WALL = 5400

# 可重复的随机方位样本。
# 保留固定随机种子，确保实验可复现。
rng = random.Random(2026102735)

angle_ranges = [
    (-45, -25),
    (25, 45),
    (-25, -10),
    (10, 25),
]

TASKS = tuple(
    (round(rng.uniform(low, high), 3), 3.0)
    for low, high in angle_ranges
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(chunk)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def write_json(path, value):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


def patch_assignments(source, replacements):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in replacements:
                    check(
                        target.id not in found,
                        "DUPLICATE_ASSIGNMENT:" + target.id
                    )
                    found[target.id] = node

    check(
        set(found) == set(replacements),
        "SOURCE_ASSIGNMENT_MISSING:"
        + repr(set(replacements) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [replacements[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("STAGE_ONLY=TRUE")
    print("PHASE_PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("HISTORICAL_MODEL_WRITE=FALSE")

    check(
        PARENT.is_file()
        and sha(PARENT) == EXPECTED_PARENT,
        "PARENT_CHECKPOINT_CHANGED"
    )

    check(
        CORE.is_file()
        and sha(CORE) == EXPECTED_CORE,
        "SOURCE_CORE_CHANGED"
    )

    prior_manifest = json.loads(
        (T31A / "prepare.json").read_text()
    )

    for filename, expected in prior_manifest["files"].items():
        file = T31A / filename
        check(
            file.is_file() and sha(file) == expected,
            "T31_SOURCE_INTEGRITY_FAILED:" + filename
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and parent.get("frames") == INITIAL
        and parent.get("updates") == 6618
        and parent.get("episode_index") == 8,
        "PARENT_METADATA_INVALID"
    )

    check(
        parent["manifest"]["experiment"]
        == "STAGE_TD3_T16",
        "PARENT_EXPERIMENT_INVALID"
    )

    check(
        parent["manifest"]["source_core"]
        == EXPECTED_CORE,
        "PARENT_PROVENANCE_INVALID"
    )

    source = SOURCE.read_text()

    source = patch_assignments(
        source,
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": f"EP_CAP = {EP_CAP}",
            "WALL": f"WALL = {WALL}",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    old_experiment = '"experiment": "STAGE_TD3_T31A"'
    new_experiment = '"experiment": "STAGE_TD3_T35"'

    check(
        source.count(old_experiment) == 1,
        "EXPERIMENT_SOURCE_ANCHOR_INVALID"
    )

    source = source.replace(
        old_experiment,
        new_experiment,
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

    manifest_path = JOB / "prepare.json"

    if manifest_path.exists():
        old = json.loads(
            manifest_path.read_text()
        )

        check(
            old["experiment"] == "STAGE_TD3_T35"
            and old["parent_sha256"] == EXPECTED_PARENT
            and old["target_frames"] == TARGET
            and old["tasks"] == [list(x) for x in TASKS]
            and old["files"] == hashes,
            "EXISTING_T35_MANIFEST_CHANGED"
        )

        for name, expected in hashes.items():
            check(
                (JOB / name).is_file()
                and sha(JOB / name) == expected,
                "EXISTING_T35_SOURCE_CHANGED:" + name
            )

        status_path = JOB / "status.json"
        check(
            status_path.is_file(),
            "EXISTING_T35_STATUS_MISSING"
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
            "EXISTING_T35_CHECKPOINT_INVALID"
        )

        print("T35_EXISTING_BRANCH=VERIFIED")

    else:
        for filename in (
            "train.py",
            "td3_core_goal_contrast.py",
            "goal_pairs.npz",
            "status.json",
            "checkpoints/episode_0008.pt",
        ):
            check(
                not (JOB / filename).exists(),
                "PARTIAL_BRANCH:" + filename
            )

        for filename, data in files.items():
            (JOB / filename).write_bytes(data)

        ckpt_dir = JOB / "checkpoints"

        ckpt_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        new_state = copy.deepcopy(parent)

        manifest = copy.deepcopy(
            parent["manifest"]
        )

        manifest["experiment"] = "STAGE_TD3_T35"
        manifest["target_frames"] = TARGET
        manifest["source_core"] = EXPECTED_CORE

        new_state["manifest"] = manifest

        checkpoint = ckpt_dir / "episode_0008.pt"

        temp = ckpt_dir / "parent.partial"

        torch.save(new_state, temp)
        os.replace(temp, checkpoint)

        write_json(
            JOB / "status.json",
            {
                "state": "RUNNING",
                "frames": INITIAL,
                "target_frames": TARGET,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        write_json(
            manifest_path,
            {
                "experiment": "STAGE_TD3_T35",
                "parent_sha256": EXPECTED_PARENT,
                "initial_frames": INITIAL,
                "target_frames": TARGET,
                "episode_cap": EP_CAP,
                "randomization_seed": 2026102735,
                "tasks": TASKS,
                "files": hashes,
                "automatic_promotion": False,
            }
        )

        print("T35_BRANCH_CREATED=TRUE")

    print("T35_RANDOM_3M_TASKS=")
    for index, (angle, distance) in enumerate(TASKS, 1):
        print(
            "TASK",
            index,
            "ANGLE_DEG=", angle,
            "DISTANCE_M=", distance
        )

    print("T35_INITIAL_FRAMES=", INITIAL)
    print("T35_TARGET_FRAMES=", TARGET)
    print("T35_NEW_TRAINING_FRAMES=", TARGET - INITIAL)
    print("T35_EP_CAP=", EP_CAP)
    print("T35_PARENT_FROZEN=TRUE")
    print("T35_READY=TRUE")


try:
    main()

except Exception as error:
    print("T35_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
