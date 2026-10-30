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
    H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"
)

SRC = (
    H / "mkmini_stage_td3_t46_success_anchored_20261027_v1"
)

T16 = (
    H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
)

PARENT = T16 / "checkpoints/episode_0008.pt"

PARENT_SHA = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

OLD_CORE_SHA = (
    "2a0d8c627026cb41fed87375cca1f821ee0d10cd32a4c47ab0eca94b3ce1d45a"
)

INITIAL = 13491
TARGET = 16491


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def write_json(path, data):
    temporary = path.with_name(
        path.name + ".partial"
    )

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(temporary, path)


def patch_job(source):
    tree = ast.parse(source)

    found = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "JOB"
            for target in node.targets
        )
    ]

    check(
        len(found) == 1,
        "JOB_ANCHOR_INVALID"
    )

    node = found[0]
    lines = source.splitlines(keepends=True)

    lines[node.lineno - 1:node.end_lineno] = [
        f"JOB = Path({str(JOB)!r})\n"
    ]

    result = "".join(lines)

    ast.parse(result)

    return result


def main():
    print("T46R_STAGE_ONLY=TRUE")
    print("T46R_PARENT_FROZEN=TRUE")
    print("T46R_PHASE_CODE_USED=FALSE")
    print("T46R_PLANNER=FALSE")
    print("T46R_ACTION_OVERRIDE=FALSE")

    check(
        PARENT.is_file()
        and sha(PARENT) == PARENT_SHA,
        "FROZEN_PARENT_CHANGED"
    )

    original_manifest_path = SRC / "prepare.json"

    check(
        original_manifest_path.is_file(),
        "T46_PREPARE_MANIFEST_MISSING"
    )

    old_manifest = json.loads(
        original_manifest_path.read_text()
    )

    check(
        old_manifest["experiment"] == "STAGE_TD3_T46B"
        and old_manifest["parent_checkpoint_sha256"]
            == PARENT_SHA,
        "T46_PROVENANCE_INVALID"
    )

    for name, digest in old_manifest["files"].items():
        path = SRC / name

        check(
            path.is_file()
            and sha(path) == digest,
            "T46_SOURCE_CHANGED:" + name
        )

    core_path = SRC / "td3_core_goal_contrast.py"

    check(
        sha(core_path) == OLD_CORE_SHA,
        "T46_CORE_UNEXPECTED"
    )

    original_core = core_path.read_text()

    lines = original_core.splitlines(keepends=True)

    matches = [
        index
        for index, line in enumerate(lines)
        if (
            "_bc_rng = _bc_numpy.random.default_rng("
            in line
        )
        and not line.lstrip().startswith("#")
    ]

    check(
        len(matches) == 1,
        "BC_NUMPY_ANCHOR_INVALID"
    )

    index = matches[0]
    line = lines[index]

    indent = line[
        :len(line) - len(line.lstrip())
    ]

    # 修复点：
    # 每次调用Actor更新都显式绑定_bc_numpy，
    # 而不是只在首次加载参考库时定义。
    lines.insert(
        index,
        indent + "import numpy as _bc_numpy\n"
    )

    new_core = "".join(lines)

    ast.parse(new_core)

    check(
        new_core.count(
            "T46_SUCCESS_BC_ACTIVE=PASS"
        ) == 1,
        "SUCCESS_RETENTION_PATCH_CHANGED"
    )

    new_core_sha = sha_bytes(
        new_core.encode()
    )

    train_path = SRC / "train.py"
    original_train = train_path.read_text()

    check(
        original_train.count(OLD_CORE_SHA) == 1,
        "TRAINER_CORE_HASH_ANCHOR_INVALID"
    )

    old_tag = '"experiment": "STAGE_TD3_T46B"'
    new_tag = '"experiment": "STAGE_TD3_T46R"'

    check(
        original_train.count(old_tag) == 1,
        "TRAINER_EXPERIMENT_ANCHOR_INVALID"
    )

    train = original_train.replace(
        OLD_CORE_SHA,
        new_core_sha,
        1
    )

    train = train.replace(
        old_tag,
        new_tag,
        1
    )

    train = patch_job(train)

    ast.parse(train)

    files = {
        "train.py": train.encode(),
        "td3_core_goal_contrast.py": new_core.encode(),
        "goal_pairs.npz": (
            SRC / "goal_pairs.npz"
        ).read_bytes(),
    }

    hashes = {
        name: sha_bytes(data)
        for name, data in files.items()
    }

    manifest = copy.deepcopy(old_manifest)

    manifest["experiment"] = "STAGE_TD3_T46R"
    manifest["files"] = hashes
    manifest["source_branch"] = "STAGE_TD3_T46B"
    manifest["repair"] = "BC_NUMPY_SCOPE"
    manifest["repaired_core_sha256"] = new_core_sha
    manifest["automatic_promotion"] = False

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():

        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():
            path = JOB / name

            check(
                path.is_file()
                and sha(path) == digest,
                "EXISTING_FILE_CHANGED:" + name
            )

        check(
            status_path.is_file(),
            "EXISTING_STATUS_MISSING"
        )

        status = json.loads(
            status_path.read_text()
        )

        ckpt = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            ckpt.is_file()
            and sha(ckpt)
                == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T46R_EXISTING_BRANCH=VERIFIED")

    else:

        check(
            not status_path.exists(),
            "PARTIAL_STATUS_FOUND"
        )

        for name in files:
            check(
                not (JOB / name).exists(),
                "PARTIAL_FILE_FOUND:" + name
            )

        ckdir = JOB / "checkpoints"

        check(
            not (
                ckdir / "episode_0008.pt"
            ).exists(),
            "PARTIAL_CHECKPOINT_FOUND"
        )

        for name, data in files.items():
            (JOB / name).write_bytes(data)

        for name, digest in hashes.items():
            check(
                sha(JOB / name) == digest,
                "COPY_VERIFICATION_FAILED:" + name
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
            "T16_PARENT_METADATA_INVALID"
        )

        branch = copy.deepcopy(parent)

        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T46R"
        )

        branch["manifest"]["source_core"] = (
            new_core_sha
        )

        branch["manifest"]["target_frames"] = (
            TARGET
        )

        ckdir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = (
            ckdir / "episode_0008.pt"
        )

        temporary = (
            ckdir / "parent.partial"
        )

        torch.save(branch, temporary)

        os.replace(
            temporary,
            checkpoint
        )

        write_json(
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

        write_json(
            manifest_path,
            manifest
        )

        print("T46R_BRANCH_CREATED=TRUE")

    print("T46R_OLD_CORE_SHA256=", OLD_CORE_SHA)
    print("T46R_NEW_CORE_SHA256=", new_core_sha)
    print("T46R_BC_SCOPE_REPAIR=PASS")
    print("T46R_NEW_TRAINING_FRAMES=3000")
    print("T46R_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T46R_PREPARE=BLOCKED",
        repr(error)
    )

    traceback.print_exc()

    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
