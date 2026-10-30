from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
ROOT = H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T26A = (
    H / "mkmini_stage_td3_t26_direction_guard_20261027_v1"
    / "t26a"
)

PARENT = T16 / "checkpoints/episode_0008.pt"
SOURCE = T26A / "train.py"
CORE = T26A / "td3_core_goal_contrast.py"
PAIRS = T26A / "goal_pairs.npz"

EXPECTED = {
    PARENT:
        "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
    SOURCE:
        "4f139284ee6ddcc078e53a997ddcf6f15be034c5d178ffb5270ac43cd8a93203",
    CORE:
        "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    PAIRS:
        "3cfeedea79a983c9afdeab780dae8f199caea18ad77c89fe5a85d6346e19ce89",
}

INITIAL = 13491
TARGET = 37491
EP_CAP = 12000
WALL = 1700

TASKS = (
    (-20, 3.0),
    (20, 3.0),
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def save_json(path, data):
    temporary = path.with_name(path.name + ".partial")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temporary, path)


def patch_assignments(source, values):
    tree = ast.parse(source)

    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in values
            ):
                check(
                    target.id not in found,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )

                found[target.id] = node

    check(
        set(found) == set(values),
        "TRAIN_ASSIGNMENT_MISSING:"
        + repr(set(values) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [values[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def add_risk_loss(source):
    """
    只修改Actor训练目标。
    不修改Actor网络、推理动作或MuJoCo环境。
    """

    anchor = "actor_loss.backward()"

    lines = source.splitlines(keepends=True)

    matches = [
        i for i, line in enumerate(lines)
        if line.strip() == anchor
    ]

    check(
        len(matches) == 1,
        "ACTOR_LOSS_PATCH_ANCHOR_INVALID"
    )

    index = matches[0]

    indent = lines[index][
        :len(lines[index]) - len(lines[index].lstrip())
    ]

    patch = [
        "# T31: training-only LiDAR risk-conditioned throttle loss",
        "# 48 rays span -135..135 degrees.",
        "# indices 8:40 approximately span -90..90 degrees.",
        "t31_near_lidar = state[:, 8:40].detach().amin(dim=1)",
        "t31_risk = (",
        "    (0.10 - t31_near_lidar) / 0.07",
        ").clamp(0.0, 1.0)",
        "t31_throttle = 0.5 * (",
        "    td3_t9_actions[:, 0] + 1.0",
        ")",
        "t31_risk_loss = (",
        "    t31_risk * t31_throttle.square()",
        ").mean()",
        "actor_loss = actor_loss + 0.04 * t31_risk_loss",
    ]

    addition = "".join(
        indent + line + "\n"
        for line in patch
    )

    lines.insert(index, addition)

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count("t31_risk_loss =") == 1,
        "RISK_LOSS_PATCH_FAILED"
    )

    return result


def prepare_branch(name, parent):
    job = ROOT / name
    experiment = "STAGE_TD3_" + name.upper()

    risk_enabled = name == "t31b"

    base_core = CORE.read_text()

    new_core = (
        add_risk_loss(base_core)
        if risk_enabled
        else base_core
    )

    core_sha = hashlib.sha256(
        new_core.encode()
    ).hexdigest()

    source = SOURCE.read_text()

    source = patch_assignments(
        source,
        {
            "JOB": f"JOB = Path({str(job)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": f"EP_CAP = {EP_CAP}",
            "WALL": f"WALL = {WALL}",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    old_exp = '"experiment": "STAGE_TD3_T26A"'

    check(
        source.count(old_exp) == 1,
        "EXPERIMENT_ANCHOR_INVALID"
    )

    source = source.replace(
        old_exp,
        f'"experiment": "{experiment}"',
        1
    )

    original_core_sha = EXPECTED[CORE]

    if risk_enabled:
        check(
            source.count(original_core_sha) == 1,
            "CORE_HASH_PATCH_ANCHOR_INVALID"
        )

        source = source.replace(
            original_core_sha,
            core_sha,
            1
        )

    ast.parse(source)

    expected_files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": new_core.encode(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    expected_hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in expected_files.items()
    }

    manifest_file = job / "prepare.json"

    if manifest_file.exists():
        manifest = json.loads(
            manifest_file.read_text()
        )

        check(
            manifest["experiment"] == experiment
            and manifest["parent_sha256"]
            == EXPECTED[PARENT]
            and manifest["target_frames"] == TARGET
            and manifest["episode_cap"] == EP_CAP
            and manifest["files"] == expected_hashes,
            "EXISTING_BRANCH_MANIFEST_INVALID:" + name
        )

        for filename, expected in expected_hashes.items():
            path = job / filename

            check(
                path.is_file()
                and sha(path) == expected,
                "EXISTING_BRANCH_SOURCE_CHANGED:" + filename
            )

        status_file = job / "status.json"

        check(
            status_file.is_file(),
            "EXISTING_STATUS_MISSING:" + name
        )

        status = json.loads(
            status_file.read_text()
        )

        checkpoint = (
            job / "checkpoints"
            / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint)
            == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED:" + name
        )

        print(
            "T31_BRANCH_EXISTING_VERIFIED=",
            name
        )

        return

    for filename in (
        "train.py",
        "td3_core_goal_contrast.py",
        "goal_pairs.npz",
        "status.json",
        "checkpoints/episode_0008.pt",
    ):
        check(
            not (job / filename).exists(),
            "PARTIAL_BRANCH_DETECTED:" + name
        )

    job.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename, data in expected_files.items():
        (job / filename).write_bytes(data)

    for filename, expected in expected_hashes.items():
        check(
            sha(job / filename) == expected,
            "FILE_COPY_VERIFICATION_FAILED:" + filename
        )

    state = copy.deepcopy(parent)

    new_manifest = copy.deepcopy(
        parent["manifest"]
    )

    new_manifest["experiment"] = experiment
    new_manifest["target_frames"] = TARGET
    new_manifest["source_core"] = core_sha

    state["manifest"] = new_manifest

    checkpoint_dir = job / "checkpoints"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = checkpoint_dir / "episode_0008.pt"

    temporary = checkpoint_dir / "parent.partial"

    torch.save(
        state,
        temporary
    )

    os.replace(
        temporary,
        checkpoint
    )

    save_json(
        job / "status.json",
        {
            "state": "RUNNING",
            "frames": INITIAL,
            "target_frames": TARGET,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha(checkpoint),
            "automatic_promotion": False,
        }
    )

    save_json(
        manifest_file,
        {
            "experiment": experiment,
            "parent_sha256": EXPECTED[PARENT],
            "initial_frames": INITIAL,
            "target_frames": TARGET,
            "episode_cap": EP_CAP,
            "tasks": TASKS,
            "risk_loss_enabled": risk_enabled,
            "files": expected_hashes,
            "automatic_promotion": False,
        }
    )

    print(
        "T31_BRANCH_CREATED=",
        name,
        "RISK_LOSS=", risk_enabled,
        "CORE_SHA256=", core_sha
    )


def main():
    print("STAGE_ONLY=TRUE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    for path, expected in EXPECTED.items():
        check(
            path.is_file()
            and sha(path) == expected,
            "SOURCE_HASH_FAILED:" + str(path)
        )

    previous_manifest = json.loads(
        (T26A / "prepare.json").read_text()
    )

    for filename, expected in previous_manifest["files"].items():
        path = T26A / filename

        check(
            path.is_file() and sha(path) == expected,
            "T26A_SOURCE_CHANGED:" + filename
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and parent["frames"] == INITIAL
        and parent["updates"] == 6618
        and parent["episode_index"] == 8,
        "PARENT_METADATA_INVALID"
    )

    check(
        parent["manifest"]["experiment"]
        == "STAGE_TD3_T16",
        "PARENT_EXPERIMENT_INVALID"
    )

    check(
        parent["manifest"]["source_core"]
        == EXPECTED[CORE],
        "PARENT_CORE_INVALID"
    )

    for name in ("t31a", "t31b"):
        prepare_branch(
            name,
            parent
        )

    print("T31_TRAINING_TASKS=", TASKS)
    print("T31_ENV_EPISODE_CAP=", EP_CAP)
    print("T31_NEW_FRAMES_PER_BRANCH=", TARGET - INITIAL)
    print("T31_PARENT_FROZEN=TRUE")
    print("T31_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T31_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
