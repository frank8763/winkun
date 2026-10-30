from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import torch

HOME = Path.home()
JOB = Path(os.environ["T71_JOB"])

PARENT = (
    HOME
    / "mkmini_stage_td3_t67_3m_nightrelay_20261027_kEPAW0"
    / "T67C022"
)

T60 = (
    HOME
    / "mkmini_stage_td3_t60_safety_pair_20261027_B8xAZp"
)

T53 = (
    HOME
    / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
    / "checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22"
    "c0d51c035744d25259560d1e936e4e8e0"
)

PILOT = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

OLD_NAME = "STAGE_TD3_T67C022"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(part)

    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    tmp = Path(str(path) + ".partial")

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(tmp, path)


def replace_one(source, old, new):
    check(
        source.count(old) == 1,
        "PATCH_ANCHOR_MISMATCH:" + old
    )

    result = source.replace(old, new, 1)
    ast.parse(result)

    return result


def replace_top(source, changes):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    hits = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in changes
            ):
                check(
                    target.id not in hits,
                    "DUPLICATE_CONST:" + target.id
                )

                hits[target.id] = node

    check(
        set(hits) == set(changes),
        "CONST_MISSING:"
        + repr(set(changes) - set(hits))
    )

    for name, node in sorted(
        hits.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def parent_valid():
    meta = read_json(PARENT / "prepare.json")

    for name, expected in meta["files"].items():
        path = PARENT / name

        check(
            path.is_file()
            and sha(path) == expected,
            "PARENT_SOURCE_CHANGED:" + name
        )

    status = read_json(PARENT / "status.json")

    check(
        status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == 134491,
        "PARENT_STATE_INVALID"
    )

    checkpoint = (
        PARENT
        / "checkpoints"
        / status["checkpoint"]
    )

    check(
        checkpoint.is_file()
        and sha(checkpoint)
            == status["checkpoint_sha256"],
        "PARENT_CHECKPOINT_CHANGED"
    )

    model = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )

    check(
        model.get("algorithm") == "TD3"
        and int(model["frames"]) == 134491,
        "PARENT_TD3_METADATA_INVALID"
    )

    original_core_sha = sha(
        PARENT / "td3_core_goal_contrast.py"
    )

    check(
        model["manifest"]["source_core"]
            == original_core_sha,
        "PARENT_CORE_MISMATCH"
    )

    check(
        sha(T53) == T53_SHA,
        "T53_FROZEN_CHANGED"
    )

    evaluator_info = read_json(
        T60 / "t60_setup.json"
    )

    check(
        sha(T60 / "evaluate.py")
            == evaluator_info["evaluator_sha256"],
        "T60_EVALUATOR_CHANGED"
    )

    return model, status, original_core_sha


def patch_core(source, initial_updates):
    original = (
        "actor_loss = actor_loss + 1.0 * _bc_loss"
    )

    modified = (
        "_t71_factor = max(0.35, 1.0 - 0.65 * "
        "min(1.0, max(0.0, "
        f"(float(self.total_updates) - {int(initial_updates)}) "
        "/ 28000.0)))\n"
        "            actor_loss = actor_loss "
        "+ _t71_factor * _bc_loss\n"
        "            if self.total_updates % 2000 == 0:\n"
        "                print("
        '"T71_DECAY_ACTIVE=PASS", '
        '"FACTOR=", round(_t71_factor, 5), '
        "flush=True)"
    )

    source = replace_one(
        source,
        original,
        modified
    )

    source = replace_one(
        source,
        "+ 0.8 * _t53_ref_loss",
        "+ (0.8 * _t71_factor) * _t53_ref_loss"
    )

    source = replace_one(
        source,
        "+ 0.5 * _t53_direction_loss",
        "+ (0.5 * _t71_factor) * _t53_direction_loss"
    )

    return source


def prepare():
    print("T71_STAGE_ONLY=TRUE")
    print("T71_PARENT_FROZEN=TRUE")
    print("T71_PLANNER=FALSE")
    print("T71_ACTION_OVERRIDE=FALSE")

    model, status, original_sha = parent_valid()

    source = (PARENT / "train.py").read_text()

    original_core = (
        PARENT / "td3_core_goal_contrast.py"
    ).read_text()

    pairs = (
        PARENT / "goal_pairs.npz"
    ).read_bytes()

    tree = ast.parse(source)
    values = {}

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id

            if name in ("TASKS", "TRAIN_SEEDS"):
                values[name] = ast.literal_eval(
                    node.value
                )

    check(
        set(values) == {"TASKS", "TRAIN_SEEDS"},
        "PARENT_TASKS_UNKNOWN"
    )

    check(
        len(values["TASKS"]) == 4
        and len(values["TRAIN_SEEDS"]) == 4,
        "PARENT_TRAIN_PAIRS_INVALID"
    )

    initial_updates = int(status["updates"])

    for label in ("T71A", "T71B"):
        folder = JOB / label

        check(
            not folder.exists(),
            "BRANCH_ALREADY_EXISTS:" + label
        )

        folder.mkdir()

        tag = "STAGE_TD3_" + label

        core = (
            original_core
            if label == "T71A"
            else patch_core(
                original_core,
                initial_updates
            )
        )

        core_sha = hashlib.sha256(
            core.encode()
        ).hexdigest()

        train_source = replace_top(
            source,
            {
                "JOB": f"JOB = Path({str(folder)!r})",
                "TARGET": "TARGET = 190491",
                "EP_CAP": "EP_CAP = 14000",
                "WALL": "WALL = 5400",
            }
        )

        train_source = replace_one(
            train_source,
            OLD_NAME,
            tag
        )

        if label == "T71B":
            train_source = replace_one(
                train_source,
                original_sha,
                core_sha
            )

        files = {
            "train.py": train_source.encode(),
            "td3_core_goal_contrast.py":
                core.encode(),
            "goal_pairs.npz": pairs,
        }

        for filename, content in files.items():
            (folder / filename).write_bytes(
                content
            )

        checkpoint_dir = folder / "checkpoints"
        checkpoint_dir.mkdir()

        child = copy.deepcopy(model)
        child["manifest"] = copy.deepcopy(
            model["manifest"]
        )

        child["manifest"]["experiment"] = tag
        child["manifest"]["target_frames"] = 190491
        child["manifest"]["source_core"] = core_sha

        checkpoint = (
            checkpoint_dir
            / status["checkpoint"]
        )

        temporary = (
            checkpoint_dir / "initial.partial"
        )

        torch.save(child, temporary)
        os.replace(temporary, checkpoint)

        new_status = copy.deepcopy(status)

        new_status.update(
            state="RUNNING",
            frames=134491,
            target_frames=190491,
            checkpoint=checkpoint.name,
            checkpoint_sha256=sha(checkpoint),
            automatic_promotion=False
        )

        write_json(
            folder / "status.json",
            new_status
        )

        write_json(
            folder / "prepare.json",
            {
                "experiment": tag,
                "parent": "T67C022",
                "parent_checkpoint_sha256":
                    status["checkpoint_sha256"],
                "parent_frames": 134491,
                "target_frames": 190491,
                "training_tasks": values["TASKS"],
                "training_seeds":
                    values["TRAIN_SEEDS"],
                "changed_variable":
                    (
                        "historical_actor_loss_decay"
                        if label == "T71B"
                        else "none"
                    ),
                "files": {
                    name:
                        hashlib.sha256(data).hexdigest()
                    for name, data in files.items()
                },
                "automatic_promotion": False,
            }
        )

        print(
            "T71_BRANCH_READY=",
            label,
            "CORE_SHA256=",
            core_sha,
            flush=True
        )

    print("T71_PREFLIGHT=PASS", flush=True)


def run_script(script, logfile, seconds):
    with Path(logfile).open("w") as output:
        result = subprocess.run(
            [
                "timeout",
                "--foreground",
                "--signal=TERM",
                "--kill-after=30s",
                f"{seconds}s",
                sys.executable,
                "-u",
                str(script),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False
        )

    print(
        "T71_RUN=",
        script,
        "RC=",
        result.returncode,
        flush=True
    )

    output_text = Path(logfile).read_text(
        errors="replace"
    )

    check(
        result.returncode == 0
        and "TD3_T6_BLOCKED=" not in output_text
        and "T71E_BLOCKED=" not in output_text,
        "SUBPROCESS_FAILED_OR_INTERNAL_ERROR:"
        + str(script)
    )


def evaluate(
    label,
    checkpoint,
    checkpoint_sha,
    core_sha,
    tasks,
    split
):
    folder = JOB / (label + "_" + split)
    folder.mkdir()

    models = {
        "T53": {
            "path": str(T53),
            "sha": T53_SHA,
            "frames": 22491,
            "core_sha":
                "35f97fa8f9a88c0807339801691b4e01"
                "f06f58967d9520fb4f136a6a09c79a07",
        },
        label: {
            "path": str(checkpoint),
            "sha": checkpoint_sha,
            "frames": 190491,
            "core_sha": core_sha,
        },
    }

    model_source = "MODELS = {\n"

    for name, info in models.items():
        model_source += (
            f"    {name!r}: {{"
            f"'path': Path({info['path']!r}), "
            f"'sha': {info['sha']!r}, "
            f"'frames': {info['frames']}, "
            f"'core_sha': {info['core_sha']!r}"
            "},\n"
        )

    model_source += "}"

    source = (
        T60 / "evaluate.py"
    ).read_text()

    source = replace_top(
        source,
        {
            "JOB": f"JOB = Path({str(folder)!r})",
            "MODELS": model_source,
            "TASKS": f"TASKS = {tasks!r}",
            "WALL": "WALL = 4800",
        }
    )

    source = (
        source.replace("T59", label)
        .replace("T60", "T71E")
    )

    ast.parse(source)

    evaluator = folder / "evaluate.py"
    evaluator.write_text(source)

    run_script(
        evaluator,
        folder / "live.log",
        4800
    )

    log = (
        folder / "live.log"
    ).read_text(errors="replace")

    check(
        "T71E_PAIR_GATE=PASS" in log
        and "T71E_STATE=EVALUATED" in log,
        "EVALUATOR_INCOMPLETE"
    )

    success = 0
    collisions = 0
    retained = 0

    for i, task in enumerate(tasks, 1):
        result_path = (
            folder / f"task_{i:02d}_{label}.json"
        )

        check(
            result_path.is_file(),
            "RESULT_MISSING:" + str(result_path)
        )

        row = read_json(result_path)

        check(
            row["checkpoint_sha256"]
                == checkpoint_sha
            and int(row["actual_seed"]) == task[0],
            "EVALUATION_IDENTITY_MISMATCH"
        )

        good = bool(
            row["success"]
            and not row["collision"]
        )

        success += int(good)
        collisions += int(row["collision"])

        if i <= 2:
            retained += int(good)

        if split == "pilot4":
            print(
                "T71_SCENE",
                label,
                i,
                row["end"],
                "CLOSEST_M=",
                round(float(row["closest_m"]), 4),
                flush=True
            )

    if split == "pilot4":
        passed = (
            success == 4
            and collisions == 0
            and retained == 2
        )
    else:
        passed = (
            len(tasks) == 40
            and success >= 36
            and collisions == 0
        )

    result = {
        "model": label,
        "split": split,
        "success": success,
        "runs": len(tasks),
        "collisions": collisions,
        "old_success_retained": retained,
        "passed": passed,
        "checkpoint_sha256": checkpoint_sha,
    }

    write_json(
        folder / "results.json",
        result
    )

    print(
        "T71_GATE",
        json.dumps(result),
        flush=True
    )

    return passed


def run():
    for label in ("T71A", "T71B"):
        folder = JOB / label

        try:
            meta = read_json(
                folder / "prepare.json"
            )

            for filename, expected in (
                meta["files"].items()
            ):
                check(
                    sha(folder / filename) == expected,
                    "GENERATED_SOURCE_CHANGED:"
                    + filename
                )

            print(
                "T71_TRAINING_START=",
                label,
                flush=True
            )

            run_script(
                folder / "train.py",
                folder / "train.log",
                5400
            )

            log = (
                folder / "train.log"
            ).read_text(errors="replace")

            check(
                "T53_ACTOR_ANCHOR_ACTIVE=PASS" in log
                and "T46_SUCCESS_BC_ACTIVE=PASS"
                    in log,
                "ACTOR_LOSS_NOT_ACTIVE"
            )

            if label == "T71B":
                check(
                    "T71_DECAY_ACTIVE=PASS" in log,
                    "DECAY_NOT_CONFIRMED"
                )

            status = read_json(
                folder / "status.json"
            )

            check(
                status["state"]
                    == "PILOT_COMPLETED_NONCOLLAPSED"
                and int(status["frames"]) == 190491,
                "TRAINING_STATE_INVALID:"
                + status["state"]
            )

            checkpoint = (
                folder
                / "checkpoints"
                / status["checkpoint"]
            )

            check(
                checkpoint.is_file()
                and sha(checkpoint)
                    == status["checkpoint_sha256"],
                "FINAL_CHECKPOINT_CHANGED"
            )

            model = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False
            )

            actual_core_sha = sha(
                folder / "td3_core_goal_contrast.py"
            )

            check(
                model["manifest"]["source_core"]
                    == actual_core_sha,
                "CHILD_CORE_PROVENANCE_INVALID"
            )

            print(
                "T71_TRAINING_COMPLETE=",
                label,
                flush=True
            )

            pilot_pass = evaluate(
                label,
                checkpoint,
                status["checkpoint_sha256"],
                actual_core_sha,
                PILOT,
                "pilot4"
            )

            if pilot_pass:
                rng = np.random.default_rng(
                    20261028071
                )

                angles = rng.uniform(
                    8.0, 40.0, 40
                )

                heldout = [
                    (
                        2030710000 + i,
                        round(
                            float(
                                -angles[i]
                                if i % 2 == 0
                                else angles[i]
                            ),
                            3
                        ),
                        3.0,
                    )
                    for i in range(40)
                ]

                evaluate(
                    label,
                    checkpoint,
                    status["checkpoint_sha256"],
                    actual_core_sha,
                    heldout,
                    "heldout40"
                )

        except Exception as error:
            print(
                "T71_BRANCH_BLOCKED=",
                label,
                repr(error),
                flush=True
            )

            traceback.print_exc()

    print(
        "T71_PAIR_EXPERIMENT_FINISHED=TRUE",
        flush=True
    )

    print("MODEL_PROMOTED=FALSE")
    print("LONG_DISTANCE=LOCKED")


try:
    mode = sys.argv[1]

    if mode == "--prepare":
        prepare()

    elif mode == "--run":
        run()

    else:
        raise RuntimeError("INVALID_MODE")

except Exception as error:
    print(
        "T71_BLOCKED=",
        repr(error),
        flush=True
    )

    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
        flush=True
    )
