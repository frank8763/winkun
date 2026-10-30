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
T43 = H / "mkmini_stage_td3_t43_mixed_replay_pilot_20261027_v1"

SOURCE = T40 / "evaluate.py"
REFERENCE = T40 / "preflight.json"

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
    "T43R": {
        "path": T43 / "checkpoints/episode_0011.pt",
        "sha": "45102b3257bf21e1a5809555420d0ef167a1710f545d55f3508b7aba14acb2c6",
        "frames": 25491,
        "core_sha": CORE_SHA,
    },
}

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def patch(source, changes):
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
                        "DUPLICATE:" + target.id
                    )
                    found[target.id] = node

    check(
        set(found) == set(changes),
        "MISSING_ASSIGNMENTS:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for key, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            changes[key] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    print("T45_STAGE_ONLY=TRUE")
    print("T45_TRAINING=FALSE")
    print("T45_MODEL_WRITE=FALSE")
    print("T45_PLANNER=FALSE")
    print("T45_ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file() and REFERENCE.is_file(),
        "T40_VERIFIED_SOURCE_MISSING"
    )

    old = json.loads(REFERENCE.read_text())

    check(
        sha(SOURCE) == old["evaluator_sha256"],
        "T40_SOURCE_HASH_MISMATCH"
    )

    for name, cfg in MODELS.items():
        path = cfg["path"]

        check(
            path.is_file()
            and sha(path) == cfg["sha"],
            "MODEL_SHA_MISMATCH:" + name
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
            "MODEL_PROVENANCE_MISMATCH:" + name
        )

        print("T45_MODEL_VERIFIED=", name)

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

    generated = patch(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 4000",
        }
    )

    generated = generated.replace("T40", "T45")
    ast.parse(generated)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "T45_EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(generated)

    manifest = {
        "experiment": "STAGE_TD3_T45",
        "source_sha256": sha(SOURCE),
        "evaluator_sha256": sha(target),
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    (JOB / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("T45_SCENES=2")
    print("T45_MUJOCO_RUNS=4")
    print("T45_EVALUATOR_SHA256=", sha(target))
    print("T45_READY=TRUE")


try:
    main()
except Exception as error:
    print("T45_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()
    raise
finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
