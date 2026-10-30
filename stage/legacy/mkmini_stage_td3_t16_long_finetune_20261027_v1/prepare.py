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
JOB = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T12 = H / "mkmini_stage_td3_t12_goal_contrast_20261027_v1"

SOURCE = T12 / "train.py"
CORE = T12 / "td3_core_goal_contrast.py"
PAIRS = T12 / "goal_pairs.npz"
PARENT = T12 / "checkpoints/episode_0005.pt"

FROZEN = (
    H / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
)

PARENT_SHA = (
    "ab7ca56e87e238615c7601789f40a8ef571eb45617b584c2bb5860b15560a146"
)

FROZEN_SHA = (
    "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
)

TARGET = 20000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def replace_one(text, old, new):
    check(
        text.count(old) == 1,
        "PATCH_ANCHOR_INVALID:" + old[:100]
    )
    return text.replace(old, new, 1)


def write_json(path, data):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(tmp, path)


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("EXTERNAL_PLANNER=FALSE")
    print("ARTIFICIAL_WAYPOINTS=FALSE")
    print("FROZEN_CHECKPOINT_WRITE=FALSE")

    for path in (SOURCE, CORE, PAIRS, PARENT, FROZEN):
        check(path.is_file(), "MISSING:" + str(path))

    check(sha(PARENT) == PARENT_SHA, "T12_PARENT_CHANGED")
    check(sha(FROZEN) == FROZEN_SHA, "FROZEN_SAC_CHANGED")

    t12_prep = json.loads(
        (T12 / "prepare.json").read_text()
    )

    for name, expected in t12_prep["files"].items():
        path = T12 / name
        check(
            path.is_file() and sha(path) == expected,
            "T12_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(parent["algorithm"] == "TD3", "BAD_ALGORITHM")
    check(parent["frames"] == 4000, "BAD_PARENT_FRAMES")
    check(parent["updates"] == 1873, "BAD_PARENT_UPDATES")
    check(parent["episode_index"] == 5, "BAD_PARENT_EPISODES")
    check(len(parent["replay"]) == 4000, "BAD_PARENT_REPLAY")

    check(
        parent["manifest"]["source_core"] == sha(CORE),
        "CORE_PROVENANCE_INVALID"
    )

    check(
        parent["manifest"]["goal_pair_dataset_sha256"]
        == sha(PAIRS),
        "GOAL_PAIR_PROVENANCE_INVALID"
    )

    script = SOURCE.read_text()

    script = replace_one(
        script,
        'JOB = H / "mkmini_stage_td3_t12_goal_contrast_20261027_v1"',
        'JOB = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"'
    )

    script = replace_one(
        script,
        '"experiment": "STAGE_TD3_T12"',
        '"experiment": "STAGE_TD3_T16"'
    )

    script = replace_one(
        script,
        "TARGET = 4000",
        "TARGET = 20000"
    )

    script = replace_one(
        script,
        "EP_CAP = 1000",
        "EP_CAP = 4000"
    )

    script = replace_one(
        script,
        "WALL = 2100",
        "WALL = 6300"
    )

    # 仅使用最终目标；不引入中间目标或规划路径。
    tree = ast.parse(script)

    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "TASKS"
            for target in node.targets
        )
    ]

    check(len(assignments) == 1, "TASKS_AST_INVALID")

    tasks = (
        (20, 1.2),
        (20, 1.0),
        (-20, 1.0),
        (-20, 1.2),
    )

    node = assignments[0]
    lines = script.splitlines(keepends=True)

    replacement = (
        "TASKS = (\n"
        + "".join(
            f"    ({angle}, {distance}),\n"
            for angle, distance in tasks
        )
        + ")\n"
    )

    lines[node.lineno - 1:node.end_lineno] = [replacement]
    script = "".join(lines)

    ast.parse(script)

    script_sha = hashlib.sha256(script.encode()).hexdigest()

    manifest = copy.deepcopy(parent["manifest"])
    manifest["experiment"] = "STAGE_TD3_T16"
    manifest["target_frames"] = TARGET

    paths = {
        "train.py": JOB / "train.py",
        "td3_core_goal_contrast.py":
            JOB / "td3_core_goal_contrast.py",
        "goal_pairs.npz": JOB / "goal_pairs.npz",
    }

    prepared = JOB / "prepare.json"

    if prepared.exists():

        info = json.loads(prepared.read_text())

        for name, expected in info["files"].items():
            path = JOB / name
            check(
                path.is_file() and sha(path) == expected,
                "T16_EXISTING_FILE_CHANGED:" + name
            )

        check(
            info["files"]["train.py"] == script_sha,
            "T16_SCRIPT_VERSION_CHANGED"
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
            "T16_RESUME_CHECKPOINT_INVALID"
        )

        print("TD3_T16_RESUME=VERIFIED")

    else:

        for name in (
            "train.py",
            "td3_core_goal_contrast.py",
            "goal_pairs.npz",
            "status.json",
            "manifest.json",
            "checkpoints/episode_0005.pt",
        ):
            check(
                not (JOB / name).exists(),
                "PARTIAL_T16_DIRECTORY:" + name
            )

        paths["train.py"].write_text(script)

        shutil.copy2(
            CORE,
            paths["td3_core_goal_contrast.py"]
        )

        shutil.copy2(
            PAIRS,
            paths["goal_pairs.npz"]
        )

        check(
            sha(paths["td3_core_goal_contrast.py"]) == sha(CORE)
            and sha(paths["goal_pairs.npz"]) == sha(PAIRS),
            "T16_COPY_VERIFICATION_FAILED"
        )

        state = copy.deepcopy(parent)
        state["manifest"] = manifest

        checkpoint_dir = JOB / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = checkpoint_dir / "episode_0005.pt"
        temp = checkpoint_dir / "initial.pt.partial"

        torch.save(state, temp)
        os.replace(temp, checkpoint)

        write_json(JOB / "manifest.json", manifest)

        write_json(
            JOB / "status.json",
            {
                "state": "RUNNING",
                "frames": 4000,
                "target_frames": TARGET,
                "checkpoint": "episode_0005.pt",
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        files = {
            name: sha(path)
            for name, path in paths.items()
        }

        write_json(
            prepared,
            {
                "experiment": "STAGE_TD3_T16",
                "source_T12_checkpoint_sha256": PARENT_SHA,
                "initial_frames": 4000,
                "target_frames": TARGET,
                "tasks": tasks,
                "episode_cap": 4000,
                "files": files,
                "automatic_promotion": False,
            }
        )

        print("TD3_T16_NEW_BRANCH=CREATED")

    print("TD3_T16_SOURCE_SHA256=", script_sha)
    print("TD3_T16_PARENT_FRAMES=4000")
    print("TD3_T16_TARGET_FRAMES=20000")
    print("TD3_T16_ADDITIONAL_FRAMES_MAX=16000")
    print("TD3_T16_EPISODE_CAP=4000")
    print("TD3_T16_TRAINING_TASKS=", tasks)
    print("TD3_T16_FROZEN_PARENT_UNCHANGED=TRUE")
    print("TD3_T16_READY=TRUE")


try:
    main()

except Exception as error:
    print("TD3_T16_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
