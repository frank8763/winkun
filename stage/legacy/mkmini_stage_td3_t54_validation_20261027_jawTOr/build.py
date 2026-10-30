from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

import torch

H = Path.home()
JOB = Path(sys.argv[1])

T47 = (
    H / "mkmini_stage_td3_t47_retention_20261027_ZlRnPq"
)

T46 = (
    H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"
)

T53 = (
    H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
)

SOURCE = T47 / "evaluate.py"

SOURCE_SHA = (
    "cf2b2c7ec9fceebc46b62e9098962e34f52eff26bdd6cbc050740726105f668c"
)

MODELS = {
    "T46R": {
        "path": T46 / "checkpoints/episode_0009.pt",
        "sha": "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e",
        "frames": 16491,
        "core_sha": "b28b45562d79dff52090a24e6b21a74bcd24d830ab89a706cb46467df8d531ea",
    },
    "T53": {
        "path": T53 / "checkpoints/episode_0010.pt",
        "sha": "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0",
        "frames": 22491,
        "core_sha": "35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07",
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

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

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
        lines[node.lineno - 1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)

    return result


def main():
    print("T54_STAGE_ONLY=TRUE")
    print("T54_TRAINING=FALSE")
    print("T54_PLANNER=FALSE")
    print("T54_WAYPOINTS=FALSE")
    print("T54_ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file()
        and sha(SOURCE) == SOURCE_SHA,
        "EVALUATOR_SOURCE_INVALID"
    )

    for name, cfg in MODELS.items():

        path = cfg["path"]

        check(
            path.is_file()
            and sha(path) == cfg["sha"],
            "CHECKPOINT_CHANGED:" + name
        )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3"
            and int(state["frames"]) == cfg["frames"]
            and state["manifest"]["source_core"]
                == cfg["core_sha"],
            "CHECKPOINT_METADATA_INVALID:" + name
        )

        print("T54_MODEL_VERIFIED=", name)

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
            "WALL": "WALL = 5400",
        }
    )

    generated = generated.replace(
        "T47", "T54"
    )

    ast.parse(generated)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(generated)

    meta = {
        "experiment": "STAGE_TD3_T54",
        "source_sha256": SOURCE_SHA,
        "evaluator_sha256": sha(target),
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    # 不预先创建manifest.json。
    # 该文件由评估器自行管理。
    (JOB / "t54_setup.json").write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    print("T54_CASES=4")
    print("T54_REAL_MUJOCO_RUNS=8")
    print("T54_READY=TRUE")


try:
    main()

except Exception as error:
    print("T54_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
