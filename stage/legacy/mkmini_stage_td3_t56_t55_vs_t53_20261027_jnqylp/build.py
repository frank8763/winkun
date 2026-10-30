from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

import torch

H = Path.home()
JOB = Path(sys.argv[1])

T54 = H / "mkmini_stage_td3_t54_validation_20261027_jawTOr"
T53 = H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
T55 = H / "mkmini_stage_td3_t55_neargoal_reward_20261027_v1"

SOURCE = T54 / "evaluate.py"
SOURCE_META = T54 / "t54_setup.json"

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07"
)

MODELS = {
    "T53": {
        "path": T53 / "checkpoints/episode_0010.pt",
        "sha": (
            "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0"
        ),
        "frames": 22491,
        "core_sha": CORE_SHA,
    },
    "T55": {
        "path": T55 / "checkpoints/episode_0015.pt",
        "sha": (
            "e093967ce1191e0ca1a3d72e15a20338413e9e4d308d90ba99ad0fccb133bbac"
        ),
        "frames": 34491,
        "core_sha": CORE_SHA,
    },
}

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def replace_assignments(source, changes):
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
        lines[node.lineno-1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    print("T56_STAGE_ONLY=TRUE")
    print("T56_TRAINING=FALSE")
    print("T56_MODEL_WRITE=FALSE")
    print("T56_PLANNER=FALSE")
    print("T56_WAYPOINTS=FALSE")
    print("T56_ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file()
        and SOURCE_META.is_file(),
        "T54_VERIFIED_EVALUATOR_MISSING"
    )

    source_info = json.loads(
        SOURCE_META.read_text()
    )

    check(
        sha(SOURCE)
        == source_info["evaluator_sha256"],
        "T54_EVALUATOR_SHA_FAILED"
    )

    check(
        source_info["source_sha256"]
        == (
            "cf2b2c7ec9fceebc46b62e9098962e34f52eff26bdd6cbc050740726105f668c"
        ),
        "T54_SOURCE_PROVENANCE_FAILED"
    )

    for name, cfg in MODELS.items():
        path = cfg["path"]

        check(
            path.is_file()
            and sha(path) == cfg["sha"],
            "CHECKPOINT_SHA_FAILED:" + name
        )

        ck = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            ck.get("algorithm") == "TD3"
            and int(ck["frames"]) == cfg["frames"]
            and ck["manifest"]["source_core"]
                == cfg["core_sha"],
            "CHECKPOINT_METADATA_FAILED:" + name
        )

        print("T56_MODEL_VERIFIED=", name)

    # 额外核对T55训练器本身没有被修改。
    prepare_path = T55 / "prepare.json"

    check(
        prepare_path.is_file(),
        "T55_PREPARE_MISSING"
    )

    prepare = json.loads(
        prepare_path.read_text()
    )

    for filename, expected in prepare["files"].items():
        path = T55 / filename

        check(
            path.is_file()
            and sha(path) == expected,
            "T55_TRAINER_CHANGED:" + filename
        )

    status = json.loads(
        (T55 / "status.json").read_text()
    )

    check(
        int(status["frames"]) == 34491
        and status["checkpoint_sha256"]
            == MODELS["T55"]["sha"],
        "T55_STATUS_INVALID"
    )

    model_lines = ["MODELS = {"]

    for name, cfg in MODELS.items():
        model_lines.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(cfg['path'])!r}),",
            f"        'sha': {cfg['sha']!r},",
            f"        'frames': {cfg['frames']},",
            f"        'core_sha': {cfg['core_sha']!r},",
            "    },",
        ])

    model_lines.append("}")

    generated = replace_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 4800",
        }
    )

    generated = generated.replace(
        "T54", "T56"
    )

    ast.parse(generated)

    evaluator = JOB / "evaluate.py"

    check(
        not evaluator.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    evaluator.write_text(generated)

    setup = {
        "experiment": "STAGE_TD3_T56",
        "source_evaluator_sha256": sha(SOURCE),
        "generated_evaluator_sha256": sha(evaluator),
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    # 不创建manifest.json，
    # 避免与评估器自身的清单发生冲突。
    (JOB / "t56_setup.json").write_text(
        json.dumps(
            setup,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    print("T56_CASE_COUNT=4")
    print("T56_REAL_MUJOCO_RUNS=8")
    print("T56_OLD_SUCCESS_CASES=2")
    print("T56_DEVELOPMENT_CASES=2")
    print("T56_READY=TRUE")


try:
    main()

except Exception as error:
    print("T56_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
