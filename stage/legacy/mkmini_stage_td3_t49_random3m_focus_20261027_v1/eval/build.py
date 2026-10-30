from pathlib import Path
import ast
import hashlib
import json
import traceback

import torch

H = Path.home()
J = H / "mkmini_stage_td3_t49_random3m_focus_20261027_v1"
E = J / "eval"

T47 = H / "mkmini_stage_td3_t47_retention_20261027_ZlRnPq"
T46 = H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"

CORE = (
    "b28b45562d79dff52090a24e6b21a74bcd24d830ab89a706cb46467df8d531ea"
)

T46_SHA = (
    "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e"
)

SRC_SHA = (
    "cf2b2c7ec9fceebc46b62e9098962e34f52eff26bdd6cbc050740726105f668c"
)

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
    (2029654803, -33.7, 3.0),
    (2029654804, 37.9, 3.0),
]


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def patch(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in changes:
                        check(
                            target.id not in found,
                            "DUPLICATE:" + target.id
                        )
                        found[target.id] = node

    check(
        set(changes) == set(found),
        "ANCHORS_MISSING:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("T49_EVAL_TRAINING=FALSE")
    print("T49_STAGE_ONLY=TRUE")
    print("T49_PLANNER=FALSE")

    status = json.loads(
        (J / "status.json").read_text()
    )

    checkpoint = (
        J / "checkpoints"
        / status["checkpoint"]
    )

    check(
        int(status["frames"]) == 28491
        and checkpoint.is_file()
        and sha(checkpoint)
            == status["checkpoint_sha256"],
        "NO_COMPLETED_T49_CHECKPOINT"
    )

    ck = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )

    check(
        ck["manifest"]["source_core"] == CORE
        and ck["frames"] == 28491,
        "NEW_CHECKPOINT_META_INVALID"
    )

    baseline = T46 / "checkpoints/episode_0009.pt"

    check(
        baseline.is_file()
        and sha(baseline) == T46_SHA,
        "FROZEN_T46_INVALID"
    )

    source = T47 / "evaluate.py"

    check(
        source.is_file()
        and sha(source) == SRC_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    models = {
        "T46R": {
            "path": baseline,
            "sha": T46_SHA,
            "frames": 16491,
            "core_sha": CORE,
        },
        "T49S": {
            "path": checkpoint,
            "sha": status["checkpoint_sha256"],
            "frames": 28491,
            "core_sha": CORE,
        },
    }

    lines = ["MODELS = {"]

    for name, value in models.items():
        lines.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(value['path'])!r}),",
            f"        'sha': {value['sha']!r},",
            f"        'frames': {value['frames']},",
            f"        'core_sha': {value['core_sha']!r},",
            "    },",
        ])

    lines.append("}")

    generated = patch(
        source.read_text(),
        {
            "JOB": f"JOB = Path({str(E)!r})",
            "MODELS": "\n".join(lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 6000",
        }
    )

    generated = generated.replace(
        "T47", "T49E"
    )

    ast.parse(generated)

    target = E / "evaluate.py"
    setup = E / "t49e_setup.json"

    meta = {
        "experiment": "STAGE_TD3_T49E",
        "source_sha256": SRC_SHA,
        "evaluator_sha256": hashlib.sha256(
            generated.encode()
        ).hexdigest(),
        "models": {
            name: value["sha"]
            for name, value in models.items()
        },
        "tasks": TASKS,
        "training": False,
        "model_promotion": False,
    }

    if target.exists() or setup.exists():

        check(
            target.is_file()
            and setup.is_file()
            and sha(target) == meta["evaluator_sha256"]
            and json.loads(setup.read_text())
                == json.loads(json.dumps(meta)),
            "PREEXISTING_EVALUATION_CHANGED"
        )

    else:

        target.write_text(generated)

        setup.write_text(
            json.dumps(meta, indent=2) + "\n"
        )

    print("T49_EVAL_CASES=6")
    print("T49_PAIRED_RUNS=12")
    print("T49_OLD_RETENTION_CASES=2")
    print("T49_EVAL_READY=TRUE")


try:
    main()

except Exception as error:
    print("T49_EVAL_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
