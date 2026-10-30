from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import re
import shutil
import traceback

import torch

H = Path.home()
ROOT = H / "mkmini_stage_td3_t26_direction_guard_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T23 = H / "mkmini_stage_td3_t23_balanced_curriculum_20261027_v1"

PARENT = T16 / "checkpoints/episode_0008.pt"
SOURCE = T23 / "train.py"
CORE = T16 / "td3_core_goal_contrast.py"
PAIRS = T16 / "goal_pairs.npz"

PARENT_SHA = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

INITIAL = 13491
TARGET = 25491
EP_CAP = 6000
WALL = 2700

TASKS = (
    (-20, 1.2),
    (20, 1.2),
    (-20, 1.5),
    (20, 1.5),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def atomic_json(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(temp, path)


def patch_assignments(source, replacements):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in replacements:
                require(
                    target.id not in found,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )
                found[target.id] = node

    require(
        set(found) == set(replacements),
        "MISSING_TOP_LEVEL_ASSIGNMENT"
    )

    for key, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            replacements[key] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


def patch_real_episode_cap(source):
    """
    在创建Stage环境之后立即设置真实max_steps。
    只定位helper.make_goal_task的实际调用位置。
    不改变MuJoCo物理参数或控制动作。
    """
    tree = ast.parse(source)
    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        call = node.value

        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "make_goal_task"
        ):
            targets = {
                item.id
                for target in node.targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            }

            if "env" in targets:
                matches.append(node)

    require(
        len(matches) == 1,
        "ENV_CREATION_ANCHOR_NOT_UNIQUE"
    )

    node = matches[0]
    lines = source.splitlines(keepends=True)

    original_line = lines[node.lineno - 1]
    indent = re.match(r"^\s*", original_line).group(0)

    addition = [
        f"{indent}env.max_steps = EP_CAP\n",
        f"{indent}check(\n",
        f"{indent}    int(env.max_steps) == EP_CAP,\n",
        f'{indent}    "T26_ENV_HORIZON_INVALID"\n',
        f"{indent})\n",
        f'{indent}print("T26_ENV_MAX_STEPS=", env.max_steps)\n',
    ]

    lines[node.end_lineno:node.end_lineno] = addition

    result = "".join(lines)
    ast.parse(result)

    require(
        result.count("T26_ENV_HORIZON_INVALID") == 1,
        "ENV_HORIZON_PATCH_FAILED"
    )

    return result


def patch_direction_loss(source):
    """
    仅增加训练损失，不改变Actor结构和执行动作。
    真实状态约束只适用于目标在前方、具有一定净空的区域。
    """
    anchor = "actor_loss = actor_loss + 0.05 * t12_goal_loss"

    matches = [
        line for line in source.splitlines()
        if anchor in line
    ]

    require(
        len(matches) == 1,
        "ACTOR_LOSS_ANCHOR_INVALID"
    )

    line = matches[0]
    indent = line[:len(line) - len(line.lstrip())]

    addition = [
        "t26_anchor_loss = (",
        "    torch.relu(0.20 + t12_l).square().mean()",
        "    + torch.relu(0.20 - t12_r).square().mean()",
        ")",
        "t26_goal_sin = state[:, 49].detach()",
        "t26_goal_cos = state[:, 50].detach()",
        "t26_goal_d10 = state[:, 48].detach()",
        "t26_lidar_min = state[:, :48].detach().amin(dim=1)",
        "t26_mask = (",
        "    (t26_goal_cos > 0.84)",
        "    & (t26_goal_sin.abs() > 0.17)",
        "    & (t26_goal_d10 >= 0.06)",
        "    & (t26_goal_d10 <= 0.16)",
        "    & (t26_lidar_min > 0.12)",
        ").float()",
        "t26_margin_error = torch.relu(",
        "    0.18",
        "    - torch.sign(t26_goal_sin)",
        "    * td3_t9_actions[:, 1]",
        ").square()",
        "t26_dynamic_loss = (",
        "    t26_mask * t26_margin_error",
        ").sum() / t26_mask.sum().clamp_min(1.0)",
        "actor_loss = (",
        "    actor_loss",
        "    + 0.04 * t26_anchor_loss",
        "    + 0.025 * t26_dynamic_loss",
        ")",
    ]

    replacement = (
        line + "\n"
        + "\n".join(indent + item for item in addition)
    )

    result = source.replace(line, replacement, 1)
    ast.parse(result)

    require(
        result.count("t26_anchor_loss =") == 1
        and result.count("t26_dynamic_loss =") == 1,
        "DIRECTION_LOSS_PATCH_FAILED"
    )

    return result


def prepare_branch(name, parent):
    job = ROOT / name

    experiment = "STAGE_TD3_" + name.upper()
    modified = name == "t26b"

    old_core = CORE.read_text()
    new_core = (
        patch_direction_loss(old_core)
        if modified else old_core
    )

    old_core_sha = sha(CORE)
    new_core_sha = hashlib.sha256(new_core.encode()).hexdigest()

    source = SOURCE.read_text()

    source = patch_assignments(
        source,
        {
            "JOB": f"JOB = H / {str(job.relative_to(H))!r}",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": f"EP_CAP = {EP_CAP}",
            "WALL": f"WALL = {WALL}",
            "TASKS": f"TASKS = {repr(TASKS)}",
        }
    )

    source = patch_real_episode_cap(source)

    old_experiment = '"experiment": "STAGE_TD3_T23"'
    new_experiment = f'"experiment": "{experiment}"'

    require(
        source.count(old_experiment) == 1,
        "TRAINER_EXPERIMENT_ANCHOR_INVALID"
    )

    source = source.replace(
        old_experiment,
        new_experiment,
        1
    )

    if modified:
        require(
            source.count(old_core_sha) == 1,
            "TRAINER_CORE_SHA_ANCHOR_INVALID"
        )
        source = source.replace(
            old_core_sha,
            new_core_sha,
            1
        )

    ast.parse(source)

    source_sha = hashlib.sha256(source.encode()).hexdigest()

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": new_core.encode(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    job.mkdir(parents=True, exist_ok=True)

    manifest_path = job / "prepare.json"

    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())

        require(
            saved["parent_sha256"] == PARENT_SHA
            and saved["target_frames"] == TARGET
            and saved["experiment"] == experiment,
            "PREPARED_MANIFEST_CHANGED:" + name
        )

        for filename, digest in saved["files"].items():
            require(
                (job / filename).is_file()
                and sha(job / filename) == digest,
                "PREPARED_SOURCE_CHANGED:" + filename
            )

        require(
            sha(job / "train.py") == source_sha
            and sha(job / "td3_core_goal_contrast.py") == new_core_sha,
            "PREPARED_PATCH_CHANGED:" + name
        )

        status = json.loads((job / "status.json").read_text())

        checkpoint = (
            job / "checkpoints" / status["checkpoint"]
        )

        require(
            checkpoint.is_file()
            and sha(checkpoint) == status["checkpoint_sha256"],
            "RESUME_CHECKPOINT_INVALID:" + name
        )

        print("T26_EXISTING_BRANCH_VERIFIED=", name)
        return

    for filename in (
        "train.py",
        "td3_core_goal_contrast.py",
        "goal_pairs.npz",
        "status.json",
        "checkpoints/episode_0008.pt",
    ):
        require(
            not (job / filename).exists(),
            "PARTIAL_BRANCH_DIRECTORY:" + name
        )

    for filename, data in files.items():
        (job / filename).write_bytes(data)

    new_state = copy.deepcopy(parent)

    new_manifest = copy.deepcopy(parent["manifest"])

    new_manifest["experiment"] = experiment
    new_manifest["target_frames"] = TARGET
    new_manifest["source_core"] = new_core_sha

    # 不往严格恢复manifest中添加训练算法说明字段。
    new_state["manifest"] = new_manifest

    checkpoint_dir = job / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_dir / "episode_0008.pt"
    temp = checkpoint_dir / "parent.partial"

    torch.save(new_state, temp)
    os.replace(temp, checkpoint)

    atomic_json(
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

    hashes = {
        filename: sha(job / filename)
        for filename in files
    }

    atomic_json(
        manifest_path,
        {
            "experiment": experiment,
            "parent_sha256": PARENT_SHA,
            "initial_frames": INITIAL,
            "target_frames": TARGET,
            "episode_cap": EP_CAP,
            "direction_constraint": modified,
            "files": hashes,
            "automatic_promotion": False,
        }
    )

    print(
        "T26_BRANCH_CREATED=",
        name,
        "CORE_SHA=", new_core_sha,
        "TRAINER_SHA=", source_sha
    )


def main():
    print("STAGE_ONLY=TRUE")
    print("NO_EXTERNAL_PLANNER=TRUE")
    print("NO_WAYPOINTS=TRUE")
    print("ACTION_OVERRIDE=FALSE")

    require(
        PARENT.is_file()
        and sha(PARENT) == PARENT_SHA,
        "PARENT_CHECKPOINT_CHANGED"
    )

    prep = json.loads(
        (T23 / "prepare.json").read_text()
    )

    for filename, expected in prep["files"].items():
        path = T23 / filename

        require(
            path.is_file() and sha(path) == expected,
            "T23_SOURCE_CHANGED:" + filename
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    require(
        parent["algorithm"] == "TD3"
        and parent["frames"] == INITIAL
        and parent["updates"] == 6618
        and parent["episode_index"] == 8,
        "PARENT_METADATA_INVALID"
    )

    require(
        parent["manifest"]["source_core"] == sha(CORE),
        "PARENT_CORE_PROVENANCE_INVALID"
    )

    require(
        parent["manifest"]["goal_pair_dataset_sha256"]
        == sha(PAIRS),
        "PARENT_GOAL_PAIRS_INVALID"
    )

    require(
        parent["manifest"]["experiment"] == "STAGE_TD3_T16",
        "PARENT_EXPERIMENT_INVALID"
    )

    for name in ("t26a", "t26b"):
        prepare_branch(name, parent)

    print("T26_PARENT=VERIFIED")
    print("T26_EPISODE_ENV_MAX_STEPS=", EP_CAP)
    print("T26_ADDITIONAL_TRAINING_FRAMES=", TARGET-INITIAL)
    print("T26_DIRECTION_CONSTRAINT_COMPARISON=ENABLED")
    print("T26_READY=TRUE")


try:
    main()
except Exception as error:
    print("T26_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
