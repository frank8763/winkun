from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback

import torch

HOME = Path.home()
JOB = Path(os.environ["T66_JOB"])

BASE = (
    HOME /
    "mkmini_stage_td3_t65_relay_20261027_kFHbLY/T65B"
)

EVAL_BASE = (
    HOME /
    "mkmini_stage_td3_t60_safety_pair_20261027_B8xAZp"
)

T53 = (
    HOME /
    "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
    "/checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22"
    "c0d51c035744d25259560d1e936e4e8e0"
)

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01"
    "f06f58967d9520fb4f136a6a09c79a07"
)

INITIAL = 78491
TARGET = 106491

TASKS = (
    (18.8, 3.0),
    (-12.5, 3.0),
    (-17.3, 3.0),
    (13.6, 3.0),
)

SEEDS = (
    2029654802,
    2029654801,
    2029653601,
    2029653602,
)

EVAL_TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    temporary = Path(str(path) + ".partial")

    temporary.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(temporary, path)


def replace_assignments(source, replacements):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in replacements
            ):
                check(
                    target.id not in found,
                    "DUPLICATE_ASSIGNMENT:"
                    + target.id
                )
                found[target.id] = node

    check(
        set(found) == set(replacements),
        "ASSIGNMENT_MISSING:"
        + repr(set(replacements) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno-1:node.end_lineno
        ] = [replacements[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def patch_progress(source):
    tree = ast.parse(source)
    steps = []
    appends = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):
            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "N"
                and call.func.attr == "step_checked"
            ):
                steps.append(node)

        if isinstance(node, ast.Expr):
            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "replay"
                and call.func.attr == "append"
            ):
                appends.append(node)

    check(
        len(steps) == 1 and len(appends) == 1,
        "TRANSITION_ANCHOR_INVALID"
    )

    check(
        steps[0].end_lineno < appends[0].lineno,
        "TRANSITION_ORDER_INVALID"
    )

    lines = source.splitlines(keepends=True)

    def indentation(node):
        line = lines[node.lineno - 1]
        return line[:len(line) - len(line.lstrip())]

    before = indentation(steps[0]) + (
        "_t66_before = float(env._goal_state()[1])\n"
    )

    after_indent = indentation(appends[0])

    after = "".join(
        after_indent + statement + "\n"
        for statement in [
            "_t66_after = float(env._goal_state()[1])",
            (
                "_t66_delta = max(-0.02, min(0.02, "
                "_t66_before - _t66_after))"
            ),
            "_t66_bonus = 18.0 * _t66_delta",
            "reward = float(reward) + _t66_bonus",
            (
                "if not globals().get("
                "'_T66_REWARD_LOGGED', False):"
            ),
            (
                "    print('T66_PROGRESS_REWARD_ACTIVE=PASS', "
                "'BONUS=', round(_t66_bonus, 7), "
                "flush=True)"
            ),
            "    globals()['_T66_REWARD_LOGGED'] = True",
        ]
    )

    insertions = [
        (steps[0].lineno - 1, before),
        (appends[0].lineno - 1, after),
    ]

    for index, block in sorted(
        insertions,
        key=lambda x: x[0],
        reverse=True
    ):
        lines.insert(index, block)

    result = "".join(lines)
    ast.parse(result)

    check(
        result.count(
            "T66_PROGRESS_REWARD_ACTIVE=PASS"
        ) == 1,
        "PROGRESS_REWARD_PATCH_FAILED"
    )

    return result


def verify_parent():
    manifest = json.loads(
        (BASE / "prepare.json").read_text()
    )

    for filename, digest in manifest["files"].items():
        path = BASE / filename

        check(
            path.is_file() and sha(path) == digest,
            "PARENT_SOURCE_CHANGED:" + filename
        )

    status = json.loads(
        (BASE / "status.json").read_text()
    )

    check(
        status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == INITIAL,
        "PARENT_STATUS_INVALID"
    )

    checkpoint = (
        BASE / "checkpoints"
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
        and int(model["frames"]) == INITIAL
        and int(model["episode_index"]) == 16
        and model["manifest"]["source_core"]
            == CORE_SHA,
        "PARENT_MODEL_METADATA_INVALID"
    )

    check(
        sha(T53) == T53_SHA,
        "T53_FROZEN_CHECKPOINT_CHANGED"
    )

    return model, status


def prepare():
    print("T66_STAGE_ONLY=TRUE")
    print("T66_PHASE_ISOLATED=TRUE")
    print("T66_PARENT_FROZEN=TRUE")
    print("T66_PLANNER=FALSE")
    print("T66_ACTION_OVERRIDE=FALSE")
    print("T66_ENV_MODIFIED=FALSE")

    parent, old_status = verify_parent()

    check(
        not (JOB / "prepare.json").exists(),
        "T66_BRANCH_ALREADY_PREPARED"
    )

    source = (BASE / "train.py").read_text()

    source = replace_assignments(
        source,
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 14000",
            "WALL": "WALL = 4200",
            "TASKS": f"TASKS = {TASKS!r}",
            "TRAIN_SEEDS": (
                f"TRAIN_SEEDS = {SEEDS!r}"
            ),
        }
    )

    source = patch_progress(source)

    old_tag = '"experiment": "STAGE_TD3_T65B"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_TAG_INVALID"
    )

    source = source.replace(
        old_tag,
        '"experiment": "STAGE_TD3_T66"',
        1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py":
            (BASE / "td3_core_goal_contrast.py")
            .read_bytes(),
        "goal_pairs.npz":
            (BASE / "goal_pairs.npz").read_bytes(),
    }

    check(
        hashlib.sha256(
            files["td3_core_goal_contrast.py"]
        ).hexdigest() == CORE_SHA,
        "CORE_CHANGED"
    )

    for filename, content in files.items():
        path = JOB / filename

        check(
            not path.exists(),
            "TARGET_SOURCE_ALREADY_EXISTS:"
            + filename
        )

        path.write_bytes(content)

    model = copy.deepcopy(parent)

    model["manifest"] = copy.deepcopy(
        parent["manifest"]
    )

    model["manifest"]["experiment"] = (
        "STAGE_TD3_T66"
    )

    model["manifest"]["target_frames"] = TARGET

    checkpoints = JOB / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    checkpoint = checkpoints / "episode_0016.pt"
    temporary = checkpoints / "initial.partial"

    check(
        not checkpoint.exists(),
        "INITIAL_CHECKPOINT_ALREADY_EXISTS"
    )

    torch.save(model, temporary)
    os.replace(temporary, checkpoint)

    status = copy.deepcopy(old_status)
    status["state"] = "RUNNING"
    status["frames"] = INITIAL
    status["target_frames"] = TARGET
    status["checkpoint"] = checkpoint.name
    status["checkpoint_sha256"] = sha(checkpoint)
    status["automatic_promotion"] = False

    save_json(JOB / "status.json", status)

    save_json(
        JOB / "prepare.json",
        {
            "experiment": "STAGE_TD3_T66",
            "parent": "T65B",
            "parent_frames": INITIAL,
            "target_frames": TARGET,
            "tasks": TASKS,
            "seeds": SEEDS,
            "goal_progress_weight": 18.0,
            "reward_clip_distance_delta_m": 0.02,
            "automatic_promotion": False,
            "files": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in files.items()
            },
        }
    )

    print("T66_START_FRAMES=", INITIAL)
    print("T66_TARGET_FRAMES=", TARGET)
    print("T66_NEW_FRAMES=", TARGET - INITIAL)
    print("T66_SOURCE_SHA256=", sha(JOB / "train.py"))
    print("T66_READY=TRUE")


def run_program(script, logfile, seconds):
    print("T66_START_SCRIPT=", script, flush=True)

    with Path(logfile).open("w") as output:
        result = subprocess.run(
            [sys.executable, "-u", str(script)],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=seconds,
            check=False
        )

    print(
        "T66_SUBPROCESS_RC=",
        result.returncode,
        "SCRIPT=",
        script,
        flush=True
    )

    check(
        result.returncode == 0,
        "SUBPROCESS_FAILED:" + str(script)
    )


def build_evaluator(checkpoint, digest):
    original = EVAL_BASE / "evaluate.py"

    old_meta = json.loads(
        (EVAL_BASE / "t60_setup.json").read_text()
    )

    check(
        sha(original)
            == old_meta["evaluator_sha256"],
        "EVALUATOR_SOURCE_CHANGED"
    )

    models = {
        "T53": {
            "path": T53,
            "sha": T53_SHA,
            "frames": 22491,
        },
        "T66": {
            "path": checkpoint,
            "sha": digest,
            "frames": TARGET,
        },
    }

    lines = ["MODELS = {"]

    for name, info in models.items():
        lines.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(info['path'])!r}),",
            f"        'sha': {info['sha']!r},",
            f"        'frames': {info['frames']},",
            f"        'core_sha': {CORE_SHA!r},",
            "    },",
        ])

    lines.append("}")

    output_dir = JOB / "pilot4"
    output_dir.mkdir(exist_ok=True)

    generated = replace_assignments(
        original.read_text(),
        {
            "JOB": (
                f"JOB = Path({str(output_dir)!r})"
            ),
            "MODELS": "\n".join(lines),
            "TASKS": f"TASKS = {EVAL_TASKS!r}",
            "WALL": "WALL = 4200",
        }
    )

    generated = (
        generated.replace("T59", "T66")
        .replace("T60", "T67")
    )

    ast.parse(generated)

    evaluator = output_dir / "evaluate.py"

    check(
        not evaluator.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    evaluator.write_text(generated)

    save_json(
        output_dir / "source.json",
        {
            "sha256": sha(evaluator),
            "models": {
                name: info["sha"]
                for name, info in models.items()
            },
            "tasks": EVAL_TASKS,
        }
    )

    return evaluator, output_dir


def audit_results(folder, candidate_state):
    log = (folder / "live.log").read_text()

    check(
        "T67_PAIR_GATE=PASS" in log
        and "T67_STATE=EVALUATED" in log
        and "T67_BLOCKED=" not in log,
        "PAIR_EVALUATION_INCOMPLETE"
    )

    total_success = 0
    total_collision = 0
    retained = 0

    for index, task in enumerate(EVAL_TASKS, 1):
        path = (
            folder
            / f"task_{index:02d}_T66.json"
        )

        check(
            path.is_file(),
            "EVAL_RESULT_MISSING:" + str(index)
        )

        row = json.loads(path.read_text())

        check(
            int(row["actual_seed"]) == task[0]
            and abs(
                float(row["angle_deg"]) - task[1]
            ) < 1e-5
            and abs(
                float(row["distance_m"]) - 3.0
            ) < 1e-5,
            "SCENE_PROVENANCE_INVALID:"
            + str(index)
        )

        success = bool(
            row["success"] and not row["collision"]
        )

        total_success += int(success)
        total_collision += int(row["collision"])

        if index <= 2:
            retained += int(success)

        print(
            "T66_PILOT_CASE",
            index,
            "ANGLE=",
            task[1],
            "END=",
            row["end"],
            "CLOSEST_M=",
            round(float(row["closest_m"]), 4),
            "COLLISION=",
            row["collision"],
            flush=True
        )

    print("T66_SUCCESS=", total_success, "/4")
    print("T66_COLLISIONS=", total_collision)
    print("T66_OLD_SUCCESS_RETAINED=", retained, "/2")

    gate = (
        candidate_state
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and retained == 2
        and total_success == 4
        and total_collision == 0
    )

    print(
        "T66_PILOT_GATE=",
        "PASS" if gate else "FAIL",
        flush=True
    )

    print("T66_RANDOM_40_GATE=NOT_RUN")
    print("T66_MODEL_PROMOTED=FALSE")
    print("LONG_DISTANCE=LOCKED")


def run():
    manifest = json.loads(
        (JOB / "prepare.json").read_text()
    )

    for name, expected in manifest["files"].items():
        check(
            sha(JOB / name) == expected,
            "GENERATED_SOURCE_CHANGED:" + name
        )

    run_program(
        JOB / "train.py",
        JOB / "train.log",
        3900
    )

    log = (JOB / "train.log").read_text()

    check(
        "T66_PROGRESS_REWARD_ACTIVE=PASS" in log
        and "T59_EXTERNAL_SAFETY_REWARD_ACTIVE=PASS"
            in log
        and "T53_ACTOR_ANCHOR_ACTIVE=PASS" in log,
        "TRAINING_MECHANISM_INACTIVE"
    )

    status = json.loads(
        (JOB / "status.json").read_text()
    )

    check(
        int(status["frames"]) == TARGET,
        "TRAINING_BUDGET_INCOMPLETE"
    )

    checkpoint = (
        JOB / "checkpoints"
        / status["checkpoint"]
    )

    digest = sha(checkpoint)

    check(
        digest == status["checkpoint_sha256"],
        "FINAL_CHECKPOINT_CHANGED"
    )

    print(
        "T66_TRAINING_STATE=",
        status["state"],
        flush=True
    )

    evaluator, output_dir = build_evaluator(
        checkpoint, digest
    )

    run_program(
        evaluator,
        output_dir / "live.log",
        2400
    )

    audit_results(
        output_dir,
        status["state"]
    )

    print(
        "T66_EXPERIMENT_COMPLETED=TRUE",
        flush=True
    )


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
        "T66_BLOCKED=",
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
