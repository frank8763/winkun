from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path(os.environ["T67_JOB"])
T66 = Path(os.environ["T66_JOB"])

T65B = (
    H / "mkmini_stage_td3_t65_relay_20261027_kFHbLY"
    / "T65B"
)

T60 = (
    H / "mkmini_stage_td3_t60_safety_pair_20261027_B8xAZp"
)

T53 = (
    H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
    / "checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22"
    "c0d51c035744d25259560d1e936e4e8e0"
)

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01"
    "f06f58967d9520fb4f136a6a09c79a07"
)

LOCK = H / ".mkmini_stage_td3_independent.lock"

CST = timezone(timedelta(hours=8))

DEADLINE = datetime(
    2026, 10, 28, 8, 0, 0,
    tzinfo=CST
).timestamp()

PARENT_FRAMES = 78491

PILOT = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

TRAIN_PAIRS = [
    ((18.8, 3.0), 2029654802),
    ((-12.5, 3.0), 2029654801),
    ((-17.3, 3.0), 2029653601),
    ((13.6, 3.0), 2029653602),
]


def log(*args):
    print(*args, flush=True)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def remaining():
    return max(
        0,
        int(DEADLINE - time.time())
    )


def digest(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    tmp = Path(str(path) + ".partial")

    tmp.write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2
        ) + "\n"
    )

    os.replace(tmp, path)


def read_checkpoint(path, frames):
    check(
        Path(path).is_file(),
        "CHECKPOINT_NOT_FOUND"
    )

    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    check(
        ck.get("algorithm") == "TD3"
        and int(ck["frames"]) == frames
        and ck["manifest"]["source_core"] == CORE_SHA,
        "CHECKPOINT_METADATA_INVALID"
    )

    return ck


def verify_sources(folder):
    meta = load_json(folder / "prepare.json")

    for name, expected in meta["files"].items():
        path = folder / name

        check(
            path.is_file()
            and digest(path) == expected,
            "SOURCE_INTEGRITY_FAILED:"
            + str(path)
        )


def verify_parent():
    verify_sources(T65B)
    verify_sources(T66)

    check(
        digest(T53) == T53_SHA,
        "T53_FROZEN_CHANGED"
    )

    status = load_json(
        T65B / "status.json"
    )

    check(
        status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == PARENT_FRAMES,
        "T65B_PARENT_STATE_INVALID"
    )

    ck = (
        T65B / "checkpoints"
        / status["checkpoint"]
    )

    check(
        digest(ck)
            == status["checkpoint_sha256"],
        "T65B_PARENT_SHA_INVALID"
    )

    read_checkpoint(ck, PARENT_FRAMES)

    eval_meta = load_json(
        T60 / "t60_setup.json"
    )

    check(
        digest(T60 / "evaluate.py")
            == eval_meta["evaluator_sha256"],
        "EVALUATOR_SOURCE_CHANGED"
    )

    return ck


def patch_constants(source, changes):
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
                    "DUPLICATE_CONSTANT:"
                    + target.id
                )

                found[target.id] = node

    check(
        set(found) == set(changes),
        "MISSING_CONSTANTS:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno-1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def replace_once(source, old, new):
    check(
        source.count(old) == 1,
        "PATCH_ANCHOR_INVALID:" + old
    )

    return source.replace(old, new, 1)


def trial_parameters(index):
    """
    固定的实验参数网格。
    不根据测试结果临时挑选有利参数。
    """

    bonuses = (10.0, 16.0, 22.0, 30.0, 38.0)
    safety = (0.65, 0.80, 0.95, 1.10, 1.20)

    position = (index - 1) % 25
    generation = (index - 1) // 25

    bonus = bonuses[position % 5]
    scale = safety[position // 5]

    # 每轮覆盖所有四个训练场景。
    training_frames = (
        56000
        if generation % 2 == 0
        else 84000
    )

    return {
        "bonus": bonus,
        "safety_scale": scale,
        "training_frames": training_frames,
        "rotation": index % 4,
    }


def make_training_branch(index, parent):
    name = f"T67C{index:03d}"
    folder = JOB / name

    check(
        not folder.exists(),
        "TRIAL_DIR_ALREADY_EXISTS"
    )

    folder.mkdir()

    cfg = trial_parameters(index)

    target = (
        PARENT_FRAMES
        + cfg["training_frames"]
    )

    pairs = TRAIN_PAIRS[
        cfg["rotation"]:
    ] + TRAIN_PAIRS[
        :cfg["rotation"]
    ]

    tasks = tuple(x[0] for x in pairs)
    seeds = tuple(x[1] for x in pairs)

    source = (
        T66 / "train.py"
    ).read_text()

    source = patch_constants(
        source,
        {
            "JOB": f"JOB = Path({str(folder)!r})",
            "TARGET": f"TARGET = {target}",
            "EP_CAP": "EP_CAP = 14000",
            "WALL": "WALL = 7500",
            "TASKS": f"TASKS = {tasks!r}",
            "TRAIN_SEEDS":
                f"TRAIN_SEEDS = {seeds!r}",
        }
    )

    source = replace_once(
        source,
        "_t66_bonus = 18.0 * _t66_delta",
        (
            "_t66_bonus = "
            f"{cfg['bonus']!r} * _t66_delta"
        )
    )

    scale = cfg["safety_scale"]

    for old_weight, margin in (
        (0.19, "front, 0.55"),
        (0.09, "rear, 0.46"),
        (0.15, "chassis, 0.56"),
    ):

        old = (
            f"{old_weight} * risk({margin})"
        )

        new = (
            f"{old_weight * scale:.6f}"
            f" * risk({margin})"
        )

        source = replace_once(
            source, old, new
        )

    source = replace_once(
        source,
        '"experiment": "STAGE_TD3_T66"',
        f'"experiment": "STAGE_TD3_{name}"'
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py":
            (
                T66 / "td3_core_goal_contrast.py"
            ).read_bytes(),
        "goal_pairs.npz":
            (
                T66 / "goal_pairs.npz"
            ).read_bytes(),
    }

    check(
        hashlib.sha256(
            files["td3_core_goal_contrast.py"]
        ).hexdigest() == CORE_SHA,
        "CORE_SOURCE_CHANGED"
    )

    for filename, data in files.items():
        (folder / filename).write_bytes(data)

    hashes = {
        filename:
            hashlib.sha256(data).hexdigest()
        for filename, data in files.items()
    }

    initial = read_checkpoint(
        parent, PARENT_FRAMES
    )

    child = copy.deepcopy(initial)

    child["manifest"] = copy.deepcopy(
        initial["manifest"]
    )

    child["manifest"]["experiment"] = (
        "STAGE_TD3_" + name
    )

    child["manifest"]["target_frames"] = target

    ckdir = folder / "checkpoints"
    ckdir.mkdir()

    checkpoint = ckdir / "episode_0016.pt"

    temporary = ckdir / "initial.partial"

    torch.save(child, temporary)
    os.replace(temporary, checkpoint)

    save_json(
        folder / "status.json",
        {
            "state": "RUNNING",
            "frames": PARENT_FRAMES,
            "target_frames": target,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256":
                digest(checkpoint),
            "automatic_promotion": False,
        }
    )

    save_json(
        folder / "prepare.json",
        {
            "experiment":
                "STAGE_TD3_" + name,
            "parent": "T65B",
            "parent_frames": PARENT_FRAMES,
            "target_frames": target,
            "tasks": tasks,
            "seeds": seeds,
            "parameters": cfg,
            "files": hashes,
            "automatic_promotion": False,
        }
    )

    verify_sources(folder)

    log(
        "T67_BRANCH_READY=",
        name,
        "BONUS=", cfg["bonus"],
        "SAFETY_SCALE=", scale,
        "TARGET=", target
    )

    return name, folder, target


def run_script(script, logfile, budget):
    budget = min(
        budget,
        remaining() - 60
    )

    check(
        budget >= 240,
        "INSUFFICIENT_TIME"
    )

    log(
        "T67_RUN_BEGIN=",
        script,
        "TIME_BUDGET_S=",
        budget
    )

    with Path(logfile).open("w") as output:
        result = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{budget}s",
                sys.executable,
                "-u",
                str(script),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False
        )

    text = Path(logfile).read_text(
        errors="replace"
    )

    log(
        "T67_RUN_RC=",
        result.returncode,
        "SCRIPT=", script
    )

    check(
        result.returncode == 0
        and "TD3_T6_BLOCKED=" not in text
        and "T68_BLOCKED=" not in text,
        "SCRIPT_FAILED_OR_BLOCKED"
    )

    return text


def finished_model(folder, target):
    status = load_json(
        folder / "status.json"
    )

    check(
        status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == target,
        "TRAINING_STATE_NOT_ACCEPTABLE:"
        + str(status["state"])
    )

    checkpoint = (
        folder / "checkpoints"
        / status["checkpoint"]
    )

    check(
        checkpoint.is_file()
        and digest(checkpoint)
            == status["checkpoint_sha256"],
        "FINAL_CHECKPOINT_CHANGED"
    )

    read_checkpoint(checkpoint, target)

    return (
        checkpoint,
        status["checkpoint_sha256"]
    )


def make_tasks(seed, count=40):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(
        8.0, 43.0, count
    )

    return [
        (
            seed * 100 + i,
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
        for i in range(count)
    ]


def evaluate(
    name, checkpoint, checkpoint_sha,
    frames, tasks, label
):
    folder = JOB / f"{name}_{label}"

    check(
        not folder.exists(),
        "EVALUATION_DIR_ALREADY_EXISTS"
    )

    folder.mkdir()

    models = {
        "T53": {
            "path": T53,
            "sha": T53_SHA,
            "frames": 22491,
        },
        name: {
            "path": checkpoint,
            "sha": checkpoint_sha,
            "frames": frames,
        },
    }

    model_lines = ["MODELS = {"]

    for model_name, item in models.items():

        model_lines.extend([
            f"    {model_name!r}: {{",
            f"        'path': Path({str(item['path'])!r}),",
            f"        'sha': {item['sha']!r},",
            f"        'frames': {item['frames']},",
            f"        'core_sha': {CORE_SHA!r},",
            "    },",
        ])

    model_lines.append("}")

    source = (
        T60 / "evaluate.py"
    ).read_text()

    source = patch_constants(
        source,
        {
            "JOB":
                f"JOB = Path({str(folder)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {tasks!r}",
            "WALL": "WALL = 14400",
        }
    )

    source = (
        source.replace("T59", name)
        .replace("T60", "T68")
    )

    ast.parse(source)

    evaluator = folder / "evaluate.py"
    evaluator.write_text(source)

    save_json(
        folder / "source.json",
        {
            "sha256": digest(evaluator),
            "candidate": name,
            "checkpoint_sha256":
                checkpoint_sha,
            "tasks": tasks,
            "automatic_promotion": False,
        }
    )

    run_script(
        evaluator,
        folder / "live.log",
        14400
    )

    output = (
        folder / "live.log"
    ).read_text(errors="replace")

    check(
        "T68_PAIR_GATE=PASS" in output
        and "T68_STATE=EVALUATED" in output
        and "T68_BLOCKED=" not in output,
        "EVALUATION_INCOMPLETE"
    )

    success = 0
    collisions = 0
    retained = 0

    for index, task in enumerate(
        tasks, 1
    ):
        row_path = (
            folder /
            f"task_{index:02d}_{name}.json"
        )

        base_path = (
            folder /
            f"task_{index:02d}_T53.json"
        )

        check(
            row_path.is_file()
            and base_path.is_file(),
            "PAIR_RESULT_MISSING"
        )

        row = load_json(row_path)
        baseline = load_json(base_path)

        check(
            row["checkpoint_sha256"]
                == checkpoint_sha
            and baseline["checkpoint_sha256"]
                == T53_SHA
            and row["start_xy"]
                == baseline["start_xy"]
            and row["goal_xy"]
                == baseline["goal_xy"]
            and int(row["actual_seed"])
                == int(task[0])
            and abs(
                float(row["distance_m"]) - 3.0
            ) < 1e-4
            and row["training"] is False
            and row["action_override"] is False,
            "PAIR_PROVENANCE_FAILED"
        )

        good = (
            row["success"]
            and not row["collision"]
        )

        success += int(good)
        collisions += int(row["collision"])

        if label == "pilot4" and index <= 2:
            retained += int(good)

        if label == "pilot4":
            log(
                "T67_PILOT_CASE",
                name,
                index,
                "BASE=", baseline["end"],
                "NEW=", row["end"],
                "CLOSEST_M=",
                round(float(row["closest_m"]), 4)
            )

    result = {
        "name": name,
        "label": label,
        "runs": len(tasks),
        "success": success,
        "collisions": collisions,
        "old_success_retained": retained,
        "checkpoint_sha256": checkpoint_sha,
    }

    if label == "pilot4":
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

    result["passed"] = passed

    save_json(
        folder / "results.json",
        result
    )

    log(
        "T67_GATE",
        json.dumps(result)
    )

    return passed


def existing_t66_candidate():
    log_path = T66 / "runner.log"
    status_path = T66 / "status.json"

    if (
        not log_path.is_file()
        or not status_path.is_file()
    ):
        return None

    contents = log_path.read_text(
        errors="replace"
    )

    if "T66_PILOT_GATE= PASS" not in contents:
        return None

    status = load_json(status_path)

    if (
        status["state"]
            != "PILOT_COMPLETED_NONCOLLAPSED"
        or int(status["frames"]) != 106491
    ):
        return None

    checkpoint = (
        T66 / "checkpoints"
        / status["checkpoint"]
    )

    if (
        not checkpoint.is_file()
        or digest(checkpoint)
            != status["checkpoint_sha256"]
    ):
        return None

    read_checkpoint(checkpoint, 106491)

    return (
        "T66",
        checkpoint,
        status["checkpoint_sha256"],
        106491
    )


def stress_test(candidate):
    """
    达标后继续使用独立种子验证3米稳定性。
    不继续修改已经达标的模型。
    """

    name, checkpoint, sha, frames = candidate

    batch = 1

    while remaining() > 600:

        tasks = make_tasks(
            2026108000 + batch
        )

        try:
            passed = evaluate(
                name,
                checkpoint,
                sha,
                frames,
                tasks,
                f"confirmation_{batch:03d}"
            )

            log(
                "T67_CONFIRMATION",
                batch,
                "PASS=", passed
            )

        except Exception as error:
            log(
                "T67_CONFIRMATION_BLOCKED=",
                repr(error)
            )

        batch += 1

    log(
        "T67_CONFIRMATION_END=TRUE"
    )


def try_candidate(candidate):
    name, checkpoint, sha, frames = candidate

    heldout = make_tasks(
        2026102767
    )

    passed = evaluate(
        name, checkpoint, sha,
        frames, heldout, "heldout40"
    )

    if passed:
        log(
            "T67_RANDOM_3M_40_GATE=PASS",
            name
        )

        save_json(
            JOB / "eligible_candidate.json",
            {
                "candidate": name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha,
                "frames": frames,
                "random40_pass": True,
                "automatic_promotion": False,
                "long_distance_started": False,
            }
        )

        stress_test(candidate)

        return True

    return False


def work():
    log("T67_STAGE_ONLY=TRUE")
    log("T67_TRAINING_ONLY_3M=TRUE")
    log("T67_PLANNER=FALSE")
    log("T67_ACTION_OVERRIDE=FALSE")
    log("T67_MODEL_PROMOTION=FALSE")

    log(
        "T67_DEADLINE_CST=",
        datetime.fromtimestamp(
            DEADLINE, CST
        ).isoformat()
    )

    parent = verify_parent()

    log(
        "T67_T66_WAIT_FOR_STAGE_LOCK=TRUE"
    )

    fd = os.open(
        LOCK,
        os.O_RDWR | os.O_CREAT,
        0o600
    )

    acquired = False

    try:
        last_heartbeat = 0

        while remaining() > 0:

            try:
                fcntl.flock(
                    fd,
                    fcntl.LOCK_EX | fcntl.LOCK_NB
                )

                acquired = True

            except BlockingIOError:
                pass

            if acquired:
                break

            if time.time() - last_heartbeat >= 300:
                log(
                    "T67_WAIT_HEARTBEAT",
                    "REMAINING_S=",
                    remaining()
                )

                last_heartbeat = time.time()

            time.sleep(
                min(15, remaining())
            )

        if not acquired:
            log(
                "T67_DEADLINE_WHILE_WAITING=TRUE"
            )
            return

        log(
            "T67_STAGE_LOCK_ACQUIRED=TRUE"
        )

        # 先检查正在运行的T66是否已经达标。
        candidate = existing_t66_candidate()

        if candidate:
            log(
                "T67_T66_PILOT_SUCCESS_DETECTED=TRUE"
            )

            if try_candidate(candidate):
                return

        index = 1

        while remaining() > 900:

            name = f"T67C{index:03d}"

            log(
                "T67_BRANCH_START=",
                name,
                "REMAINING_S=",
                remaining()
            )

            try:
                branch_name, folder, target = (
                    make_training_branch(
                        index, parent
                    )
                )

                run_script(
                    folder / "train.py",
                    folder / "train.log",
                    7500
                )

                checkpoint, sha = (
                    finished_model(
                        folder, target
                    )
                )

                log(
                    "T67_TRAINING_COMPLETE=",
                    name
                )

                pilot_ok = evaluate(
                    name,
                    checkpoint,
                    sha,
                    target,
                    PILOT,
                    "pilot4"
                )

                if pilot_ok:

                    log(
                        "T67_PILOT_4_GATE=PASS",
                        name
                    )

                    if try_candidate(
                        (
                            name,
                            checkpoint,
                            sha,
                            target
                        )
                    ):
                        return

                else:
                    log(
                        "T67_PILOT_4_GATE=FAIL",
                        name
                    )

            except Exception as error:

                log(
                    "T67_BRANCH_BLOCKED=",
                    name,
                    repr(error)
                )

                traceback.print_exc()

            index += 1

        log(
            "T67_NO_NEW_TRAINING_BEFORE_DEADLINE=TRUE"
        )

    finally:
        if acquired:
            fcntl.flock(
                fd,
                fcntl.LOCK_UN
            )

        os.close(fd)

        log(
            "T67_STAGE_LOCK_RELEASED=TRUE"
        )


def preflight():
    check(
        remaining() > 0,
        "DEADLINE_ALREADY_PASSED"
    )

    parent = verify_parent()

    source = (T66 / "train.py").read_text()

    check(
        source.count(
            "_t66_bonus = 18.0 * _t66_delta"
        ) == 1,
        "PROGRESS_REWARD_ANCHOR_INVALID"
    )

    for anchor in (
        "0.19 * risk(front, 0.55)",
        "0.09 * risk(rear, 0.46)",
        "0.15 * risk(chassis, 0.56)",
        '"experiment": "STAGE_TD3_T66"',
    ):
        check(
            source.count(anchor) == 1,
            "SOURCE_ANCHOR_INVALID:" + anchor
        )

    log(
        "T67_PARENT_VERIFIED=",
        parent
    )

    log(
        "T67_DEADLINE=",
        datetime.fromtimestamp(
            DEADLINE, CST
        ).isoformat()
    )

    log("T67_PREFLIGHT=PASS")


try:
    mode = sys.argv[1]

    if mode == "--check":
        preflight()

    elif mode == "--run":
        work()

    else:
        raise RuntimeError(
            "INVALID_MODE"
        )

except Exception as error:
    log(
        "T67_BLOCKED=",
        repr(error)
    )

    traceback.print_exc()

    raise

finally:
    log(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
