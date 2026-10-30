from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import numpy as np
import torch

H = Path.home()

JOB = H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
BASE = H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"
REF = H / "mkmini_stage_td3_t38_policy_retention_20261027_v1"

PARENT = BASE / "checkpoints/episode_0009.pt"
TRAIN = BASE / "train.py"
CORE = BASE / "td3_core_goal_contrast.py"
PAIRS = BASE / "goal_pairs.npz"
BANK = REF / "t16_reference_bank.npz"

PARENT_SHA = (
    "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e"
)
CORE_SHA = (
    "b28b45562d79dff52090a24e6b21a74bcd24d830ab89a706cb46467df8d531ea"
)
BANK_SHA = (
    "37d100a34140bbb3ab986297c1f5db786f8b2f1a5862b91efea8bc761efa1732"
)

START = 16491
TARGET = 22491

TASKS = (
    (20.0, 1.2),
    (-20.0, 1.2),
    (20.0, 3.0),
    (-20.0, 3.0),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def save_json(path, data):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


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
                        "DUPLICATE_ASSIGNMENT:" + target.id
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
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


ANCHOR_CODE = """
if not hasattr(self, "_t53_ref_obs"):
    import numpy as _t53_np
    import hashlib as _t53_hashlib

    _t53_path = BANK_PATH
    _t53_digest = _t53_hashlib.sha256()

    with open(_t53_path, "rb") as _t53_file:
        for _t53_chunk in iter(
            lambda: _t53_file.read(1048576), b""
        ):
            _t53_digest.update(_t53_chunk)

    if _t53_digest.hexdigest() != BANK_HASH:
        raise RuntimeError("T53_REFERENCE_HASH_CHANGED")

    with _t53_np.load(_t53_path) as _t53_data:
        _t53_obs = _t53_np.asarray(
            _t53_data["observations"],
            dtype=_t53_np.float32
        ).copy()

        _t53_actions = _t53_np.asarray(
            _t53_data["teacher_raw_actions"],
            dtype=_t53_np.float32
        ).copy()

    if (
        _t53_obs.shape != (8192, 112)
        or _t53_actions.shape != (8192, 2)
        or not _t53_np.isfinite(_t53_obs).all()
        or not _t53_np.isfinite(_t53_actions).all()
    ):
        raise RuntimeError("T53_REFERENCE_INVALID")

    self._t53_left_ids = _t53_np.flatnonzero(
        _t53_actions[:, 1] < -0.2
    )
    self._t53_right_ids = _t53_np.flatnonzero(
        _t53_actions[:, 1] > 0.2
    )

    if (
        len(self._t53_left_ids) < 128
        or len(self._t53_right_ids) < 128
    ):
        raise RuntimeError("T53_DIRECTION_COVERAGE_INVALID")

    self._t53_ref_obs = torch.as_tensor(
        _t53_obs,
        dtype=torch.float32,
        device=self.device
    )
    self._t53_ref_actions = torch.as_tensor(
        _t53_actions,
        dtype=torch.float32,
        device=self.device
    )

    print(
        "T53_REFERENCE_LOADED=PASS",
        "LEFT=", len(self._t53_left_ids),
        "RIGHT=", len(self._t53_right_ids),
        flush=True
    )

import numpy as _t53_np

_t53_rng = _t53_np.random.default_rng(
    202610275300000 + int(self.total_updates)
)

_t53_left = _t53_rng.choice(
    self._t53_left_ids,
    size=128,
    replace=False
)

_t53_right = _t53_rng.choice(
    self._t53_right_ids,
    size=128,
    replace=False
)

_t53_indices = torch.as_tensor(
    _t53_np.concatenate(
        [_t53_left, _t53_right]
    ),
    dtype=torch.long,
    device=self.device
)

_t53_states = self._t53_ref_obs[_t53_indices]
_t53_teacher = self._t53_ref_actions[_t53_indices]

_t53_pred = self.actor(_t53_states)

_t53_ref_loss = torch.nn.functional.smooth_l1_loss(
    _t53_pred,
    _t53_teacher
)

_t53_sign = torch.sign(
    _t53_teacher[:, 1]
)

_t53_direction_loss = (
    torch.relu(
        0.15 - _t53_sign * _t53_pred[:, 1]
    ).square().mean()
)

actor_loss = (
    actor_loss
    + 0.8 * _t53_ref_loss
    + 0.5 * _t53_direction_loss
)

if not hasattr(self, "_t53_announced"):
    self._t53_announced = True
    print(
        "T53_ACTOR_ANCHOR_ACTIVE=PASS",
        "LEFT=128",
        "RIGHT=128",
        "REFERENCE_WEIGHT=0.8",
        "DIRECTION_WEIGHT=0.5",
        flush=True
    )
"""


def patch_core(source):
    lines = source.splitlines(keepends=True)

    before = [
        i for i, line in enumerate(lines)
        if line.strip() == "actor_loss.backward()"
    ]

    step = [
        i for i, line in enumerate(lines)
        if line.strip() == "self.actor_optimizer.step()"
    ]

    check(
        len(before) == 1 and len(step) == 1,
        "ACTOR_UPDATE_ANCHOR_INVALID"
    )

    index = before[0]
    indentation = lines[index][
        :len(lines[index]) - len(lines[index].lstrip())
    ]

    block = ANCHOR_CODE.replace(
        "BANK_PATH", repr(str(BANK))
    ).replace(
        "BANK_HASH", repr(BANK_SHA)
    )

    inserted = "".join(
        indentation + line + "\n"
        if line else "\n"
        for line in block.strip().splitlines()
    )

    lines.insert(index, inserted)
    modified = "".join(lines)

    # 仅在训练期间限制Actor的梯度和学习率，
    # 不在推理期间修改动作。
    modified_lines = modified.splitlines(keepends=True)

    steps = [
        i for i, line in enumerate(modified_lines)
        if line.strip() == "self.actor_optimizer.step()"
    ]

    check(
        len(steps) == 1,
        "ACTOR_STEP_NOT_UNIQUE"
    )

    i = steps[0]
    indent = modified_lines[i][
        :len(modified_lines[i])
        - len(modified_lines[i].lstrip())
    ]

    modified_lines.insert(
        i,
        indent
        + "torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)\n"
        + indent
        + "for _t53_group in self.actor_optimizer.param_groups:\n"
        + indent
        + "    _t53_group['lr'] = min(float(_t53_group['lr']), 1e-5)\n"
    )

    result = "".join(modified_lines)

    ast.parse(result)

    check(
        result.count("T53_ACTOR_ANCHOR_ACTIVE=PASS") == 1,
        "T53_PATCH_INVALID"
    )

    return result


def main():
    print("T53_STAGE_ONLY=TRUE")
    print("T53_PARENT_FROZEN=TRUE")
    print("T53_PLANNER=FALSE")
    print("T53_ACTION_OVERRIDE=FALSE")

    for path, digest in (
        (PARENT, PARENT_SHA),
        (CORE, CORE_SHA),
        (BANK, BANK_SHA),
    ):
        check(
            path.is_file() and sha(path) == digest,
            "SOURCE_INTEGRITY_INVALID:" + str(path)
        )

    old_manifest = json.loads(
        (BASE / "prepare.json").read_text()
    )

    for name, digest in old_manifest["files"].items():
        check(
            (BASE / name).is_file()
            and sha(BASE / name) == digest,
            "BASE_SOURCE_CHANGED:" + name
        )

    with np.load(BANK) as data:
        obs = data["observations"]
        actions = data["teacher_raw_actions"]

        check(
            obs.shape == (8192, 112)
            and actions.shape == (8192, 2)
            and int((actions[:, 1] < -0.2).sum()) >= 128
            and int((actions[:, 1] > 0.2).sum()) >= 128,
            "REFERENCE_BANK_INVALID"
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and int(parent["frames"]) == START
        and int(parent["episode_index"]) == 9
        and parent["manifest"]["source_core"] == CORE_SHA,
        "PARENT_METADATA_INVALID"
    )

    core_text = patch_core(CORE.read_text())
    new_core_sha = hashlib.sha256(
        core_text.encode()
    ).hexdigest()

    source = replace_assignments(
        TRAIN.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 12000",
            "WALL": "WALL = 2400",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    check(
        source.count(CORE_SHA) == 1,
        "TRAINER_CORE_SHA_ANCHOR_INVALID"
    )

    source = source.replace(
        CORE_SHA,
        new_core_sha,
        1
    )

    old_tag = '"experiment": "STAGE_TD3_T46R"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_TAG_INVALID"
    )

    source = source.replace(
        old_tag,
        '"experiment": "STAGE_TD3_T53"',
        1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": core_text.encode(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    manifest = {
        "experiment": "STAGE_TD3_T53",
        "parent_sha256": PARENT_SHA,
        "reference_bank_sha256": BANK_SHA,
        "initial_frames": START,
        "target_frames": TARGET,
        "actor_lr_cap": 1e-5,
        "actor_grad_clip": 0.5,
        "balanced_direction_anchors": True,
        "success_radius_modified": False,
        "planner": False,
        "action_override": False,
        "files": hashes,
        "automatic_promotion": False,
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text()) == manifest,
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():
            check(
                (JOB / name).is_file()
                and sha(JOB / name) == digest,
                "EXISTING_SOURCE_CHANGED:" + name
            )

        status = json.loads(status_path.read_text())

        ckpt = JOB / "checkpoints" / status["checkpoint"]

        check(
            ckpt.is_file()
            and sha(ckpt) == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T53_EXISTING_BRANCH=VERIFIED")

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
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        check(
            not (checkpoint_dir / "episode_0009.pt").exists(),
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
            "STAGE_TD3_T53"
        )
        branch["manifest"]["source_core"] = new_core_sha
        branch["manifest"]["target_frames"] = TARGET

        checkpoint = checkpoint_dir / "episode_0009.pt"
        temporary = checkpoint_dir / "parent.partial"

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

        print("T53_BRANCH_CREATED=TRUE")

    print("T53_NEW_CORE_SHA256=", new_core_sha)
    print("T53_NEW_TRAINING_FRAMES=", TARGET - START)
    print("T53_READY=TRUE")


try:
    main()

except Exception as error:
    print("T53_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
