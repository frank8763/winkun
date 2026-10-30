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

H = Path.home()
JOB = Path(os.environ["T62_JOB"])

T53 = H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
T59 = H / "mkmini_stage_td3_t59_wheel_clearance_20261027_v1"
T60 = H / "mkmini_stage_td3_t60_safety_pair_20261027_B8xAZp"

PARENT = T53 / "checkpoints/episode_0010.pt"

PARENT_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22"
    "c0d51c035744d25259560d1e936e4e8e0"
)

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01"
    "f06f58967d9520fb4f136a6a09c79a07"
)

START = 22491
TARGET = 50491

# 前两个旧成功场景 + 两个困难场景。
# episode_index=10，故首先训练右+18.8度。
TASKS = (
    (-17.3, 3.0),
    (13.6, 3.0),
    (18.8, 3.0),
    (-12.5, 3.0),
)

SEEDS = (
    2029653601,
    2029653602,
    2029654802,
    2029654801,
)


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(part)

    return h.hexdigest()


def save(path, data):
    temp = Path(str(path) + ".partial")

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    os.replace(temp, path)


def patch_assignments(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in changes
            ):
                check(
                    target.id not in found,
                    "DUPLICATE:" + target.id
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


def patch_seed(source):
    tree = ast.parse(source)

    matches = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id == "seed"
                for t in node.targets
            )
        )
    ]

    check(
        len(matches) == 1,
        "SEED_ASSIGNMENT_NOT_UNIQUE:"
        + str(len(matches))
    )

    node = matches[0]
    lines = source.splitlines(keepends=True)

    indent = lines[node.lineno - 1][:node.col_offset]

    lines[node.lineno - 1:node.end_lineno] = [
        indent
        + "seed = TRAIN_SEEDS[episode_index % len(TASKS)]\n"
    ]

    result = "".join(lines)
    ast.parse(result)

    return result


def check_sources(folder):
    meta = json.loads(
        (folder / "prepare.json").read_text()
    )

    for name, expected in meta["files"].items():
        path = folder / name

        check(
            path.is_file()
            and sha(path) == expected,
            "SOURCE_CHANGED:" + str(path)
        )


def read_model(path, frames):
    check(path.is_file(), "CHECKPOINT_MISSING")

    model = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    check(
        model.get("algorithm") == "TD3"
        and int(model["frames"]) == frames
        and model["manifest"]["source_core"] == CORE_SHA,
        "CHECKPOINT_METADATA_INVALID"
    )

    return model


def create_branch(
    folder, label, parent, initial, source_dir,
    target, cap, tasks, seeds, modify_reward, wall
):
    check(
        len(tasks) == len(seeds),
        "TASK_SEED_MISMATCH"
    )

    folder.mkdir(parents=True, exist_ok=True)

    check(
        not (folder / "prepare.json").exists(),
        "BRANCH_ALREADY_EXISTS"
    )

    ckpt = read_model(parent, initial)

    core = source_dir / "td3_core_goal_contrast.py"
    pairs = source_dir / "goal_pairs.npz"

    check(
        sha(core) == CORE_SHA,
        "CORE_CHANGED"
    )

    source = (source_dir / "train.py").read_text()

    changes = {
        "JOB": f"JOB = Path({str(folder)!r})",
        "TARGET": f"TARGET = {target}",
        "EP_CAP": f"EP_CAP = {cap}",
        "WALL": f"WALL = {wall}",
        "TASKS": f"TASKS = {tuple(tasks)!r}",
    }

    existing_seed_table = any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == "TRAIN_SEEDS"
            for t in n.targets
        )
        for n in ast.parse(source).body
    )

    if existing_seed_table:
        changes["TRAIN_SEEDS"] = (
            f"TRAIN_SEEDS = {tuple(seeds)!r}"
        )

    source = patch_assignments(source, changes)
    source = patch_seed(source)

    if not existing_seed_table:
        anchor = f"TASKS = {tuple(tasks)!r}"

        check(
            source.count(anchor) == 1,
            "TASK_ANCHOR_INVALID"
        )

        source = source.replace(
            anchor,
            anchor
            + f"\nTRAIN_SEEDS = {tuple(seeds)!r}",
            1
        )

    check(
        source.count("TRAIN_SEEDS = ") == 1,
        "SEED_TABLE_INVALID"
    )

    if modify_reward:
        replacements = {
            "max_range = 0.35":
                "max_range = 0.60",

            "0.018 * risk(front, 0.20)":
                "0.12 * risk(front, 0.48)",

            "0.010 * risk(rear, 0.16)":
                "0.05 * risk(rear, 0.35)",

            "0.012 * risk(chassis, 0.20)":
                "0.08 * risk(chassis, 0.45)",
        }

        for old, new in replacements.items():
            check(
                source.count(old) == 1,
                "REWARD_ANCHOR_INVALID:" + old
            )

            source = source.replace(old, new, 1)

        old = "reward = float(reward) - _t59_penalty"

        new = (
            "reward = float(reward) - "
            "_t59_penalty * "
            "(0.6 + 1.6 * "
            "max(0.0, float(action[0]))**2)"
        )

        check(
            source.count(old) == 1,
            "REWARD_OUTPUT_ANCHOR_INVALID"
        )

        source = source.replace(old, new, 1)

    if label == "T62":
        old_tag = '"experiment": "STAGE_TD3_T59"'
    else:
        old_tag = '"experiment": "STAGE_TD3_T62"'

    check(
        old_tag in source,
        "EXPERIMENT_TAG_MISSING"
    )

    source = source.replace(
        old_tag,
        f'"experiment": "STAGE_TD3_{label}"'
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": core.read_bytes(),
        "goal_pairs.npz": pairs.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    for name, data in files.items():
        path = folder / name

        check(
            not path.exists(),
            "OUTPUT_ALREADY_EXISTS:" + name
        )

        path.write_bytes(data)

    for name, expected in hashes.items():
        check(
            sha(folder / name) == expected,
            "GENERATED_SHA_FAILED:" + name
        )

    new_model = copy.deepcopy(ckpt)

    new_model["manifest"] = copy.deepcopy(
        ckpt["manifest"]
    )

    new_model["manifest"]["experiment"] = (
        "STAGE_TD3_" + label
    )

    new_model["manifest"]["target_frames"] = target

    ckdir = folder / "checkpoints"
    ckdir.mkdir(exist_ok=True)

    checkpoint = (
        ckdir
        / f"episode_{int(new_model['episode_index']):04d}.pt"
    )

    temporary = ckdir / "initial.partial"

    torch.save(new_model, temporary)
    os.replace(temporary, checkpoint)

    save(
        folder / "status.json",
        {
            "state": "RUNNING",
            "frames": initial,
            "target_frames": target,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha(checkpoint),
            "automatic_promotion": False,
        }
    )

    save(
        folder / "prepare.json",
        {
            "experiment": "STAGE_TD3_" + label,
            "parent_sha256": sha(parent),
            "initial_frames": initial,
            "target_frames": target,
            "episode_cap": cap,
            "tasks": tasks,
            "training_seeds": seeds,
            "files": hashes,
            "automatic_promotion": False,
        }
    )

    print(
        "BRANCH_READY=",
        label,
        "SOURCE_SHA256=",
        sha(folder / "train.py"),
        flush=True
    )


def run_script(script, logfile):
    print(
        "START_SCRIPT=",
        script,
        flush=True
    )

    with logfile.open("w") as f:
        process = subprocess.run(
            [sys.executable, "-u", str(script)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False
        )

    output = logfile.read_text(errors="replace")

    print(
        "SCRIPT_RC=",
        process.returncode,
        flush=True
    )

    print(
        "\n".join(output.splitlines()[-12:]),
        flush=True
    )

    check(
        process.returncode == 0
        and "TD3_T6_BLOCKED=" not in output
        and "T63_BLOCKED=" not in output,
        "SCRIPT_FAILED:" + str(script)
    )

    return output


def completed_model(folder, target):
    status = json.loads(
        (folder / "status.json").read_text()
    )

    check(
        int(status["frames"]) == target,
        "TRAINING_BUDGET_NOT_COMPLETED"
    )

    check(
        status["state"] == "PILOT_COMPLETED_NONCOLLAPSED",
        "TRAINING_FINAL_STATE_INVALID"
    )

    path = (
        folder / "checkpoints"
        / status["checkpoint"]
    )

    check(
        sha(path) == status["checkpoint_sha256"],
        "FINAL_CHECKPOINT_SHA_INVALID"
    )

    read_model(path, target)

    return path, status["checkpoint_sha256"]


def build_evaluator(
    folder, label, tasks, checkpoint,
    checkpoint_sha, frames, wall
):
    folder.mkdir(parents=True, exist_ok=True)

    source = T60 / "evaluate.py"

    meta = json.loads(
        (T60 / "t60_setup.json").read_text()
    )

    check(
        sha(source) == meta["evaluator_sha256"],
        "T60_EVALUATOR_CHANGED"
    )

    models = {
        "T53": {
            "path": str(PARENT),
            "sha": PARENT_SHA,
            "frames": START,
            "core_sha": CORE_SHA,
        },
        label: {
            "path": str(checkpoint),
            "sha": checkpoint_sha,
            "frames": frames,
            "core_sha": CORE_SHA,
        },
    }

    model_lines = ["MODELS = {"]

    for name, cfg in models.items():
        model_lines += [
            f"    {name!r}: {{",
            f"        'path': Path({cfg['path']!r}),",
            f"        'sha': {cfg['sha']!r},",
            f"        'frames': {cfg['frames']},",
            f"        'core_sha': {cfg['core_sha']!r},",
            "    },",
        ]

    model_lines.append("}")

    generated = patch_assignments(
        source.read_text(),
        {
            "JOB": f"JOB = Path({str(folder)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {list(tasks)!r}",
            "WALL": f"WALL = {wall}",
        }
    )

    generated = (
        generated
        .replace("T59", label)
        .replace("T60", "T63")
    )

    ast.parse(generated)

    target = folder / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(generated)

    save(
        folder / "source.json",
        {
            "evaluator_sha256": sha(target),
            "models": models,
            "tasks": tasks,
        }
    )

    return target


def check_eval(folder, label, tasks, heldout):
    log = (folder / "live.log").read_text()

    check(
        "T63_PAIR_GATE=PASS" in log
        and "T63_STATE=EVALUATED" in log
        and "T63_BLOCKED=" not in log,
        "EVALUATION_INCOMPLETE"
    )

    totals = {
        "T53": {"success": 0, "collision": 0},
        label: {"success": 0, "collision": 0},
    }

    retained = 0

    setup = json.loads(
        (folder / "source.json").read_text()
    )

    for index, task in enumerate(tasks, 1):
        rows = {}

        for model in totals:
            path = (
                folder
                / f"task_{index:02d}_{model}.json"
            )

            check(
                path.is_file(),
                "RESULT_MISSING:" + str(path)
            )

            row = json.loads(path.read_text())
            rows[model] = row

            check(
                row["checkpoint_sha256"]
                == setup["models"][model]["sha"],
                "CHECKPOINT_PROVENANCE_FAILED"
            )

            check(
                int(row["actual_seed"]) == int(task[0])
                and abs(
                    float(row["angle_deg"]) - float(task[1])
                ) <= 2.0
                and abs(float(row["distance_m"]) - 3.0) < 1e-4,
                "TASK_PROVENANCE_FAILED"
            )

            check(
                row["training"] is False
                and row["action_override"] is False,
                "EVALUATION_MODE_INVALID"
            )

            totals[model]["success"] += int(
                row["success"] and not row["collision"]
            )

            totals[model]["collision"] += int(
                row["collision"]
            )

        check(
            rows["T53"]["start_xy"]
            == rows[label]["start_xy"]
            and rows["T53"]["goal_xy"]
            == rows[label]["goal_xy"],
            "PAIRED_GEOMETRY_INVALID"
        )

        if not heldout:
            if index <= 2:
                retained += int(
                    rows[label]["success"]
                    and not rows[label]["collision"]
                )

            print(
                "T62_PILOT_CASE",
                index,
                "OLD=", rows["T53"]["end"],
                "NEW=", rows[label]["end"],
                "CLOSEST_M=",
                round(float(rows[label]["closest_m"]), 4),
                flush=True
            )

    print(
        "T62_EVALUATION_TOTALS=",
        totals,
        flush=True
    )

    if heldout:
        check(
            len(tasks) == 40
            and totals[label]["success"] >= 36
            and totals[label]["collision"] == 0,
            "HELDOUT_40_GATE_FAILED"
        )
    else:
        check(
            len(tasks) == 4
            and retained == 2
            and totals[label]["success"] == 4
            and totals[label]["collision"] == 0,
            "PILOT_4_GATE_FAILED"
        )


def preflight():
    print("T62_STAGE_ONLY=TRUE")
    print("T62_PARENT_FROZEN=TRUE")
    print("T62_PLANNER=FALSE")
    print("T62_ACTION_OVERRIDE=FALSE")

    check_sources(T53)
    check_sources(T59)

    check(
        sha(PARENT) == PARENT_SHA,
        "T53_PARENT_CHECKPOINT_CHANGED"
    )

    read_model(PARENT, START)

    create_branch(
        JOB / "stage_t62_train",
        "T62",
        PARENT,
        START,
        T59,
        TARGET,
        14000,
        TASKS,
        SEEDS,
        True,
        7200
    )

    print("T62_PREFLIGHT=PASS")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")


def night():
    print("T62_OVERNIGHT_STAGE_ONLY=TRUE", flush=True)

    training = JOB / "stage_t62_train"

    log = run_script(
        training / "train.py",
        training / "night_train.log"
    )

    check(
        "T59_EXTERNAL_SAFETY_REWARD_ACTIVE=PASS" in log
        and "T53_ACTOR_ANCHOR_ACTIVE=PASS" in log,
        "SAFETY_OR_ANCHOR_NOT_ACTIVE"
    )

    checkpoint, checkpoint_sha = completed_model(
        training, TARGET
    )

    print("T62_3M_TRAINING_COMPLETE=TRUE", flush=True)

    pilot_tasks = [
        (2029653601, -17.3, 3.0),
        (2029653602, 13.6, 3.0),
        (2029654801, -12.5, 3.0),
        (2029654802, 18.8, 3.0),
    ]

    pilot = JOB / "pilot_4"

    evaluator = build_evaluator(
        pilot,
        "T62",
        pilot_tasks,
        checkpoint,
        checkpoint_sha,
        TARGET,
        6000
    )

    run_script(evaluator, pilot / "live.log")

    check_eval(
        pilot,
        "T62",
        pilot_tasks,
        heldout=False
    )

    print("T62_PILOT_4_GATE=PASS", flush=True)

    # 预先固定40个新种子，不能根据结果筛选。
    rng = np.random.default_rng(2026102762)

    degrees = rng.uniform(
        8.0, 43.0, size=40
    )

    random_tasks = [
        (
            2029670001 + i,
            round(
                float(
                    -degrees[i]
                    if i % 2 == 0
                    else degrees[i]
                ),
                3
            ),
            3.0,
        )
        for i in range(40)
    ]

    heldout = JOB / "heldout_40"

    evaluator = build_evaluator(
        heldout,
        "T62",
        random_tasks,
        checkpoint,
        checkpoint_sha,
        TARGET,
        24000
    )

    run_script(evaluator, heldout / "live.log")

    check_eval(
        heldout,
        "T62",
        random_tasks,
        heldout=True
    )

    print(
        "T62_RANDOM_3M_40_GATE=PASS",
        flush=True
    )

    # 只有40场景通过，才创建长距离训练分支。
    long_job = JOB / "stage_t64_long_5m"

    long_tasks = (
        (-15.0, 5.0),
        (15.0, 5.0),
        (-30.0, 5.0),
        (30.0, 5.0),
    )

    long_seeds = (
        2029688101,
        2029688102,
        2029688103,
        2029688104,
    )

    create_branch(
        long_job,
        "T64",
        checkpoint,
        TARGET,
        training,
        110491,
        20000,
        long_tasks,
        long_seeds,
        False,
        28800
    )

    print(
        "LONG_DISTANCE_5M_STARTED=TRUE",
        flush=True
    )

    run_script(
        long_job / "train.py",
        long_job / "night_train.log"
    )

    print(
        "LONG_DISTANCE_5M_TRAINING_COMPLETED=TRUE",
        flush=True
    )

    print(
        "LONG_DISTANCE_INDEPENDENT_EVALUATION_REQUIRED=TRUE",
        flush=True
    )


try:
    if sys.argv[1] == "--preflight":
        preflight()

    elif sys.argv[1] == "--night":
        night()

    else:
        raise RuntimeError("INVALID_MODE")

except Exception as error:
    print(
        "T62_BLOCKED=",
        repr(error),
        flush=True
    )

    traceback.print_exc()
    raise
