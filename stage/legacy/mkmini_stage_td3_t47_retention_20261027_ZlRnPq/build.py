from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

import torch

H = Path.home()
JOB = Path(sys.argv[1])

T40 = H / "mkmini_stage_td3_t40_forgetting_regression_20261027_v1"
T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T46R = H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"

SOURCE = T40 / "evaluate.py"
SOURCE_META = T40 / "preflight.json"

MODELS = {
    "T16": {
        "path": T16 / "checkpoints/episode_0008.pt",
        "sha": "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
        "frames": 13491,
        "core_sha": "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    },
    "T46R": {
        "path": T46R / "checkpoints/episode_0009.pt",
        "sha": "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e",
        "frames": 16491,
        "core_sha": "b28b45562d79dff52090a24e6b21a74bcd24d830ab89a706cb46467df8d531ea",
    },
}

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653601, 31.7, 3.0),
    (2029653602, -39.4, 3.0),
    (2029653602, 13.6, 3.0),
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1048576), b""):
            h.update(data)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def modify(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in changes:
                    check(
                        target.id not in found,
                        "DUPLICATE_" + target.id
                    )
                    found[target.id] = node

    check(
        set(found) == set(changes),
        "SOURCE_ASSIGNMENTS_MISSING:"
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
    print("T47_STAGE_ONLY=TRUE")
    print("T47_TRAINING=FALSE")
    print("T47_PLANNER=FALSE")
    print("T47_ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file()
        and SOURCE_META.is_file(),
        "T40_EVALUATOR_MISSING"
    )

    old = json.loads(SOURCE_META.read_text())

    check(
        sha(SOURCE) == old["evaluator_sha256"],
        "T40_EVALUATOR_CHANGED"
    )

    for name, cfg in MODELS.items():
        path = cfg["path"]

        check(
            path.is_file()
            and sha(path) == cfg["sha"],
            "MODEL_SHA_FAILED:" + name
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
            "MODEL_PROVENANCE_FAILED:" + name
        )

        print("T47_MODEL_VERIFIED=", name)

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

    generated = modify(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 6000",
        }
    )

    generated = generated.replace("T40", "T47")

    ast.parse(generated)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(generated)

    # 只保存自己的构建信息。
    # 不创建manifest.json，避免与评估器冲突。
    info = {
        "experiment": "STAGE_TD3_T47",
        "source_evaluator_sha256": sha(SOURCE),
        "generated_evaluator_sha256": sha(target),
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    (JOB / "t47_setup.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )

    print("T47_EVALUATOR_SHA256=", sha(target))
    print("T47_TASK_COUNT=4")
    print("T47_MUJOCO_RUNS=8")
    print("T47_NO_MANIFEST_CONFLICT=TRUE")
    print("T47_READY=TRUE")


try:
    main()

except Exception as error:
    print("T47_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
