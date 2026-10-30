from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import textwrap
import traceback

import numpy as np
import torch

H = Path.home()

JOB = H / "mkmini_stage_td3_t39_retention_pilot_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T31A = (
    H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"
    / "t31a"
)

T38 = H / "mkmini_stage_td3_t38_policy_retention_20261027_v1"

PARENT = T16 / "checkpoints/episode_0008.pt"
SOURCE = T31A / "train.py"
CORE = T31A / "td3_core_goal_contrast.py"
PAIRS = T31A / "goal_pairs.npz"
BANK = T38 / "t16_reference_bank.npz"
BANK_META = T38 / "reference_manifest.json"

PARENT_SHA = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

CORE_SHA = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

BANK_SHA = (
    "37d100a34140bbb3ab986297c1f5db786f8b2f1a5862b91efea8bc761efa1732"
)

INITIAL = 13491
TARGET = 37491
EP_CAP = 12000
WALL = 3600

TASKS = ((-20, 3.0), (20, 3.0))


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def atomic_json(path, data):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            data, indent=2, ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


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
                require(
                    target.id not in found,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )
                found[target.id] = node

    require(
        set(found) == set(values),
        "ASSIGNMENT_MISSING:"
        + repr(set(values) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[node.lineno-1:node.end_lineno] = [
            values[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def patch_core(source):

    lines = source.splitlines(keepends=True)

    indices = [
        i for i, line in enumerate(lines)
        if line.strip() == "actor_loss.backward()"
    ]

    require(
        len(indices) == 1,
        "ACTOR_LOSS_ANCHOR_INVALID"
    )

    index = indices[0]
    original_line = lines[index]

    indent = original_line[
        :len(original_line)-len(original_line.lstrip())
    ]

    block = textwrap.dedent("""
    # T39: training-only reference-policy retention.
    # No inference-time action override is introduced.
    if not hasattr(self, "_t39_reference_obs"):
        import hashlib as _t39_hashlib
        import numpy as _t39_np

        _t39_path = BANK_PATH_LITERAL
        _t39_hasher = _t39_hashlib.sha256()

        with open(_t39_path, "rb") as _t39_file:
            for _t39_chunk in iter(
                lambda: _t39_file.read(1048576), b""
            ):
                _t39_hasher.update(_t39_chunk)

        if _t39_hasher.hexdigest() != BANK_SHA_LITERAL:
            raise RuntimeError("T39_REFERENCE_SHA_FAILED")

        with _t39_np.load(_t39_path) as _t39_npz:
            _t39_obs = _t39_np.asarray(
                _t39_npz["observations"],
                dtype=_t39_np.float32
            )
            _t39_actions = _t39_np.asarray(
                _t39_npz["teacher_raw_actions"],
                dtype=_t39_np.float32
            )

        if (
            _t39_obs.shape != (8192, 112)
            or _t39_actions.shape != (8192, 2)
            or not _t39_np.isfinite(_t39_obs).all()
            or not _t39_np.isfinite(_t39_actions).all()
        ):
            raise RuntimeError("T39_REFERENCE_SHAPE_INVALID")

        self._t39_reference_obs = torch.as_tensor(
            _t39_obs,
            dtype=state.dtype,
            device=state.device
        )

        self._t39_reference_actions = torch.as_tensor(
            _t39_actions,
            dtype=state.dtype,
            device=state.device
        )

        print(
            "T39_REFERENCE_LOADED=PASS",
            "SAMPLES=8192",
            "LOSS_WEIGHT=0.05",
            flush=True
        )

    _t39_indices = torch.randint(
        0,
        self._t39_reference_obs.shape[0],
        (256,),
        device=state.device
    )

    _t39_student = self.actor(
        self._t39_reference_obs[_t39_indices]
    )

    _t39_teacher = (
        self._t39_reference_actions[_t39_indices]
    )

    _t39_loss = (
        _t39_student - _t39_teacher
    ).square().mean()

    actor_loss = actor_loss + 0.05 * _t39_loss
    """).strip("\n")

    block = block.replace(
        "BANK_PATH_LITERAL", repr(str(BANK))
    ).replace(
        "BANK_SHA_LITERAL", repr(BANK_SHA)
    )

    insertion = "".join(
        indent + line + "\n"
        for line in block.splitlines()
    )

    lines.insert(index, insertion)

    result = "".join(lines)
    ast.parse(result)

    require(
        result.count("T39_REFERENCE_LOADED=PASS") == 1
        and result.count("actor_loss.backward()") == 1,
        "T39_CORE_PATCH_FAILED"
    )

    return result


def main():
    print("T39_STAGE_ONLY=TRUE")
    print("T39_PARENT_FROZEN=TRUE")
    print("T39_EXTERNAL_PLANNER=FALSE")
    print("T39_ACTION_OVERRIDE=FALSE")

    require(
        PARENT.is_file() and sha(PARENT) == PARENT_SHA,
        "PARENT_CHECKPOINT_CHANGED"
    )

    require(
        CORE.is_file() and sha(CORE) == CORE_SHA,
        "PARENT_CORE_CHANGED"
    )

    require(
        BANK.is_file() and sha(BANK) == BANK_SHA,
        "REFERENCE_BANK_CHANGED"
    )

    require(
        BANK_META.is_file(),
        "REFERENCE_MANIFEST_MISSING"
    )

    bank_meta = json.loads(BANK_META.read_text())

    require(
        bank_meta["reference_bank_sha256"] == BANK_SHA
        and bank_meta["reference_count"] == 8192
        and bank_meta["parent_checkpoint_sha256"] == PARENT_SHA,
        "REFERENCE_PROVENANCE_INVALID"
    )

    with np.load(BANK) as data:
        require(
            data["observations"].shape == (8192, 112)
            and data["teacher_raw_actions"].shape == (8192, 2),
            "BANK_SHAPE_INVALID"
        )

    original_manifest = json.loads(
        (T31A / "prepare.json").read_text()
    )

    for filename, expected in original_manifest["files"].items():
        path = T31A / filename

        require(
            path.is_file() and sha(path) == expected,
            "BASELINE_SOURCE_CHANGED:" + filename
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    require(
        parent.get("algorithm") == "TD3"
        and parent.get("frames") == INITIAL
        and parent.get("updates") == 6618
        and parent.get("episode_index") == 8,
        "PARENT_METADATA_INVALID"
    )

    require(
        parent["manifest"]["experiment"] == "STAGE_TD3_T16"
        and parent["manifest"]["source_core"] == CORE_SHA,
        "PARENT_PROVENANCE_INVALID"
    )

    new_core = patch_core(CORE.read_text())

    new_core_sha = hashlib.sha256(
        new_core.encode()
    ).hexdigest()

    source = patch_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": f"EP_CAP = {EP_CAP}",
            "WALL": f"WALL = {WALL}",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    old_exp = '"experiment": "STAGE_TD3_T31A"'

    require(
        source.count(old_exp) == 1,
        "EXPERIMENT_ANCHOR_INVALID"
    )

    source = source.replace(
        old_exp,
        '"experiment": "STAGE_TD3_T39B"',
        1
    )

    require(
        source.count(CORE_SHA) == 1,
        "TRAINER_CORE_HASH_ANCHOR_INVALID"
    )

    source = source.replace(
        CORE_SHA,
        new_core_sha,
        1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": new_core.encode(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    file_hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    manifest = {
        "experiment": "STAGE_TD3_T39B",
        "parent_checkpoint_sha256": PARENT_SHA,
        "reference_bank_sha256": BANK_SHA,
        "core_sha256": new_core_sha,
        "initial_frames": INITIAL,
        "target_frames": TARGET,
        "episode_cap": EP_CAP,
        "tasks": TASKS,
        "retention_loss_weight": 0.05,
        "reference_batch_size": 256,
        "files": file_hashes,
        "automatic_promotion": False,
    }

    if manifest_path.exists():

        require(
            json.loads(manifest_path.read_text())
            == json.loads(json.dumps(manifest)),
            "EXISTING_MANIFEST_CHANGED"
        )

        for filename, expected in file_hashes.items():
            path = JOB / filename

            require(
                path.is_file() and sha(path) == expected,
                "EXISTING_SOURCE_CHANGED:" + filename
            )

        require(
            status_path.is_file(),
            "EXISTING_STATUS_MISSING"
        )

        status = json.loads(status_path.read_text())

        checkpoint = (
            JOB / "checkpoints" / status["checkpoint"]
        )

        require(
            checkpoint.is_file()
            and sha(checkpoint) == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T39_EXISTING_BRANCH=VERIFIED")

    else:

        for filename in (
            "train.py",
            "td3_core_goal_contrast.py",
            "goal_pairs.npz",
            "status.json",
            "checkpoints/episode_0008.pt",
        ):
            require(
                not (JOB / filename).exists(),
                "PARTIAL_BRANCH_FOUND:" + filename
            )

        for filename, data in files.items():
            (JOB / filename).write_bytes(data)

        for filename, expected in file_hashes.items():
            require(
                sha(JOB / filename) == expected,
                "COPIED_FILE_SHA_FAILED:" + filename
            )

        new_state = copy.deepcopy(parent)

        new_state["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        new_state["manifest"]["experiment"] = (
            "STAGE_TD3_T39B"
        )

        new_state["manifest"]["target_frames"] = TARGET
        new_state["manifest"]["source_core"] = new_core_sha

        checkpoint_dir = JOB / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = checkpoint_dir / "episode_0008.pt"
        temporary = checkpoint_dir / "parent.partial"

        torch.save(new_state, temporary)
        os.replace(temporary, checkpoint)

        atomic_json(
            status_path,
            {
                "state": "RUNNING",
                "frames": INITIAL,
                "target_frames": TARGET,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        atomic_json(manifest_path, manifest)

        print("T39_BRANCH_CREATED=TRUE")

    print("T39_CORE_SHA256=", new_core_sha)
    print("T39_REFERENCE_SHA256=", BANK_SHA)
    print("T39_TASKS=", TASKS)
    print("T39_ADDITIONAL_FRAMES=", TARGET - INITIAL)
    print("T39_READY=TRUE")


try:
    main()

except Exception as error:
    print("T39_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
