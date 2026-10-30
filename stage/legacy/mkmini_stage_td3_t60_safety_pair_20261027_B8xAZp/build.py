from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

import torch

H = Path.home()
JOB = Path(sys.argv[1])

BASE = (
    H / "mkmini_stage_td3_t56_t55_vs_t53_20261027_jnqylp"
)

T53 = (
    H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
)

T59 = (
    H / "mkmini_stage_td3_t59_wheel_clearance_20261027_v1"
)

SOURCE = BASE / "evaluate.py"
SOURCE_META = BASE / "t56_setup.json"

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
    "T59": {
        "path": T59 / "checkpoints/episode_0012.pt",
        "sha": (
            "d04fb9645bde46ffc5e06b10175406d66111edc0fc97fc3fb4af254e4e5a4f65"
        ),
        "frames": 30491,
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


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch(source, changes):
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
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("T60_STAGE_ONLY=TRUE")
    print("T60_TRAINING=FALSE")
    print("T60_PLANNER=FALSE")
    print("T60_WAYPOINTS=FALSE")
    print("T60_ACTION_OVERRIDE=FALSE")
    print("T60_MODEL_WRITE=FALSE")

    check(
        SOURCE.is_file()
        and SOURCE_META.is_file(),
        "T56_EVALUATOR_MISSING"
    )

    source_meta = json.loads(
        SOURCE_META.read_text()
    )

    check(
        sha(SOURCE)
        == source_meta["generated_evaluator_sha256"],
        "T56_EVALUATOR_CHANGED"
    )

    check(
        [tuple(x) for x in source_meta["tasks"]]
        == TASKS,
        "BASELINE_TASKS_CHANGED"
    )

    for name, info in MODELS.items():
        path = info["path"]

        check(
            path.is_file()
            and sha(path) == info["sha"],
            "MODEL_SHA_FAILED:" + name
        )

        ck = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            ck.get("algorithm") == "TD3"
            and int(ck["frames"]) == info["frames"]
            and ck["manifest"]["source_core"]
                == info["core_sha"],
            "MODEL_METADATA_FAILED:" + name
        )

        print("T60_MODEL_VERIFIED=", name)

    # 检查T59训练源文件仍与清单一致。
    meta = json.loads(
        (T59 / "prepare.json").read_text()
    )

    for filename, expected in meta["files"].items():
        path = T59 / filename

        check(
            path.is_file()
            and sha(path) == expected,
            "T59_SOURCE_CHANGED:" + filename
        )

    status = json.loads(
        (T59 / "status.json").read_text()
    )

    check(
        int(status["frames"]) == 30491
        and status["checkpoint_sha256"]
            == MODELS["T59"]["sha"]
        and status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED",
        "T59_TRAINING_STATUS_INVALID"
    )

    model_lines = ["MODELS = {"]

    for name, info in MODELS.items():
        model_lines.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(info['path'])!r}),",
            f"        'sha': {info['sha']!r},",
            f"        'frames': {info['frames']},",
            f"        'core_sha': {info['core_sha']!r},",
            "    },",
        ])

    model_lines.append("}")

    generated = patch(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 5400",
        }
    )

    generated = generated.replace(
        "T56", "T60"
    ).replace(
        "T55", "T59"
    )

    ast.parse(generated)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(generated)

    setup = {
        "experiment": "STAGE_TD3_T60",
        "source_evaluator_sha256": sha(SOURCE),
        "evaluator_sha256": sha(target),
        "models": {
            name: info["sha"]
            for name, info in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    (JOB / "t60_setup.json").write_text(
        json.dumps(
            setup,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    print("T60_CASES=4")
    print("T60_MUJOCO_RUNS=8")
    print("T60_SUCCESS_RADIUS_M=0.2")
    print("T60_READY=TRUE")


try:
    main()

except Exception as error:
    print("T60_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
