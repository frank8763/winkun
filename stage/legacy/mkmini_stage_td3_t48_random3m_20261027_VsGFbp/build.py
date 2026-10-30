from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

H = Path.home()
SOURCE = Path(sys.argv[1])
JOB = Path(sys.argv[2])

EXPECTED_SOURCE_SHA = (
    "cf2b2c7ec9fceebc46b62e9098962e34f52eff26bdd6cbc050740726105f668c"
)

MODELS = {
    "T16": (
        H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
        / "checkpoints/episode_0008.pt",
        "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a",
    ),
    "T46R": (
        H / "mkmini_stage_td3_t46r_scopefix_20261027_v1"
        / "checkpoints/episode_0009.pt",
        "e76b37d474cba39a8bb93f8ec6da1b9d2507110f30f9a6d617f62cb4db38330e",
    ),
}

TASKS = [
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
    (2029654803, -33.7, 3.0),
    (2029654804, 37.9, 3.0),
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def replace_assignments(source, values):
    tree = ast.parse(source)
    nodes = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in values:
                    check(
                        target.id not in nodes,
                        "DUPLICATE_" + target.id
                    )
                    nodes[target.id] = node

    check(
        set(nodes) == set(values),
        "ASSIGNMENT_MISSING:"
        + repr(set(values) - set(nodes))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        nodes.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            values[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    print("T48_STAGE_ONLY=TRUE")
    print("T48_TRAINING=FALSE")
    print("T48_PLANNER=FALSE")
    print("T48_ACTION_OVERRIDE=FALSE")
    print("T48_MODEL_WRITE=FALSE")

    evaluator = SOURCE / "evaluate.py"
    source_meta = SOURCE / "t47_setup.json"

    check(
        evaluator.is_file()
        and source_meta.is_file(),
        "T47_SOURCE_MISSING"
    )

    old_meta = json.loads(
        source_meta.read_text()
    )

    check(
        sha(evaluator) == EXPECTED_SOURCE_SHA
        and old_meta["generated_evaluator_sha256"]
            == EXPECTED_SOURCE_SHA,
        "T47_SOURCE_INTEGRITY_FAILED"
    )

    for name, (path, digest) in MODELS.items():
        check(
            path.is_file()
            and sha(path) == digest,
            "CHECKPOINT_INTEGRITY_FAILED:" + name
        )

        print("T48_MODEL_VERIFIED=", name)

    generated = replace_assignments(
        evaluator.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 6200",
        }
    )

    generated = generated.replace("T47", "T48")

    ast.parse(generated)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "T48_SCRIPT_ALREADY_EXISTS"
    )

    target.write_text(generated)

    meta = {
        "experiment": "STAGE_TD3_T48",
        "source_evaluator_sha256": EXPECTED_SOURCE_SHA,
        "generated_evaluator_sha256": sha(target),
        "models": {
            name: digest
            for name, (_, digest) in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    # 只写构建清单。
    # 不占用评估器自己的manifest.json。
    (JOB / "t48_setup.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print("T48_NEW_SCENES=", len(TASKS))
    print("T48_MUJOCO_RUNS=", len(TASKS) * 2)
    print("T48_EVALUATOR_SHA256=", sha(target))
    print("T48_READY=TRUE")


try:
    main()

except Exception as error:
    print("T48_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
