from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

import torch

H = Path.home()

SOURCE = (
    H / "mkmini_stage_td3_t32_3m_paired_20261027_v1"
    / "evaluate.py"
)

JOB = H / "mkmini_stage_td3_t36_random3m_screen_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T31 = H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"

T35 = H / "mkmini_stage_td3_t35_random3m_20261027_v1"

CORE_SHA = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

MODELS = {
    "T16": {
        "path": T16 / "checkpoints/episode_0008.pt",
        "sha": "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
        "frames": 13491,
        "core_sha": CORE_SHA,
    },
    "T31A": {
        "path": T31 / "t31a/checkpoints/episode_0011.pt",
        "sha": "081f083010d9794d4a212a1428a04352347facfed995a4dcfb8f9957e249dc1e",
        "frames": 37491,
        "core_sha": CORE_SHA,
    },
    "T35": {
        "path": T35 / "checkpoints/episode_0016.pt",
        "sha": "50fff483557cfcd2aeeb6aee40a74148cefeda581c2e9644d26986d15b095bbe",
        "frames": 61491,
        "core_sha": CORE_SHA,
    },
}

# 训练中从未使用过的种子和目标角度。
# 同一个任务中，三个模型使用相同的实际起点和目标。
TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653601,  31.7, 3.0),
    (2029653602, -39.4, 3.0),
    (2029653602,  13.6, 3.0),
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)

    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch_assignments(source, replacements):
    tree = ast.parse(source)

    nodes = {}

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        if isinstance(node, ast.Assign):
            targets = node.targets
        else:
            targets = [node.target]

        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id in replacements
            ):
                require(
                    target.id not in nodes,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )
                nodes[target.id] = node

    require(
        set(nodes) == set(replacements),
        "MISSING_SOURCE_ASSIGNMENT:"
        + repr(set(replacements) - set(nodes))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        nodes.items(),
        key=lambda pair: pair[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            replacements[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("T36_STAGE_ONLY=TRUE")
    print("T36_READ_ONLY_EVALUATION=TRUE")
    print("T36_TRAINING=FALSE")
    print("T36_PLANNER=FALSE")
    print("T36_ACTION_OVERRIDE=FALSE")

    require(
        SOURCE.is_file(),
        "VERIFIED_T32_EVALUATOR_MISSING"
    )

    original = SOURCE.read_text()

    tree = ast.parse(original)

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    require(
        {"prepare", "run_case", "main"}.issubset(functions),
        "T32_EVALUATOR_STRUCTURE_CHANGED"
    )

    require(
        '"experiment": "STAGE_TD3_T32"' in original
        and "PAIR_START_MISMATCH" in original
        and "PAIR_GOAL_MISMATCH" in original,
        "T32_PAIR_VALIDATION_MISSING"
    )

    for name, cfg in MODELS.items():
        path = cfg["path"]

        require(
            path.is_file() and sha(path) == cfg["sha"],
            "CHECKPOINT_SHA_INVALID:" + name
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        require(
            checkpoint.get("algorithm") == "TD3"
            and int(checkpoint["frames"]) == cfg["frames"],
            "CHECKPOINT_METADATA_INVALID:" + name
        )

        require(
            checkpoint["manifest"]["source_core"]
            == cfg["core_sha"],
            "CHECKPOINT_CORE_INVALID:" + name
        )

        print(
            "T36_MODEL_VERIFIED=",
            name,
            "FRAMES=", cfg["frames"]
        )

    models_source = ["MODELS = {"]

    for name, cfg in MODELS.items():
        models_source.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(cfg['path'])!r}),",
            f"        'sha': {cfg['sha']!r},",
            f"        'frames': {cfg['frames']},",
            f"        'core_sha': {cfg['core_sha']!r},",
            "    },",
        ])

    models_source.append("}")

    replacements = {
        "JOB": f"JOB = Path({str(JOB)!r})",
        "MODELS": "\n".join(models_source),
        "SEEDS": (
            "SEEDS = (2029653601, 2029653602)"
        ),
        "TASKS": f"TASKS = {TASKS!r}",
        "WALL": "WALL = 2400",
    }

    generated = patch_assignments(
        original,
        replacements
    )

    generated = generated.replace(
        "T32",
        "T36"
    )

    ast.parse(generated)

    target = JOB / "evaluate.py"

    expected = hashlib.sha256(
        generated.encode()
    ).hexdigest()

    if target.exists():
        require(
            sha(target) == expected,
            "T36_EXISTING_SCRIPT_CHANGED"
        )

        print("T36_SCRIPT=EXISTING_VERIFIED")

    else:
        target.write_text(generated)
        print("T36_SCRIPT=CREATED")

    provenance = {
        "experiment": "STAGE_TD3_T36",
        "source_evaluator_sha256": sha(SOURCE),
        "generated_sha256": expected,
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "cap": 14000,
        "training": False,
        "automatic_promotion": False,
    }

    path = JOB / "provenance.json"

    if path.exists():
        require(
            json.loads(path.read_text()) == provenance,
            "T36_PROVENANCE_CHANGED"
        )
    else:
        temp = JOB / "provenance.partial"
        temp.write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        os.replace(temp, path)

    print("T36_TEST_TASKS=")

    for seed, angle, distance in TASKS:
        print(
            "SEED=", seed,
            "ANGLE=", angle,
            "DISTANCE=", distance
        )

    print("T36_PAIRED_TASKS=4")
    print("T36_TOTAL_MUJOCO_RUNS=12")
    print("T36_READY=TRUE")


try:
    main()
except Exception as error:
    print("T36_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
