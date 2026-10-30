from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

H = Path.home()
JOB = Path(os.environ["T90S_JOB"])
OUT = Path(os.environ["T90S_EVAL"])

SRC = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j/"
    "pilot4_verified_UhtIxl/evaluate.py"
)

EXPECTED_EVAL_SHA = (
    "3d23dad58d7c84cc8ecb47a4df09fc8005ca53"
    "a0a9718ef26ae7222621fae276"
)

EXPECTED_MODEL_SHA = (
    "0f2490c418ee4905093924031987dd79d2e066"
    "a05f89553563562fe2bd44261f"
)

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def top_node(source, name):
    tree = ast.parse(source)
    found = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]
    check(len(found) == 1, "BINDING_INVALID:" + name)
    return found[0]

def replace_top(source, name, expression):
    node = top_node(source, name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        name + " = " + expression + "\n"
    ]
    result = "".join(lines)
    ast.parse(result)
    return result

try:
    print("=" * 72)
    print("T90S NATIVE MUJOCO STRICT EVALUATION")
    print("=" * 72)

    check(
        SRC.is_file() and sha(SRC) == EXPECTED_EVAL_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    status = json.loads(
        (JOB / "status.json").read_text()
    )

    manifest = json.loads(
        (JOB / "experiment.json").read_text()
    )

    check(
        status.get("state") == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status.get("frames", -1)) == 190491,
        "T90S_TRAINING_NOT_COMPLETE"
    )

    ckpt = JOB / "checkpoints" / status["checkpoint"]

    check(
        ckpt.is_file()
        and sha(ckpt) == EXPECTED_MODEL_SHA
        and status["checkpoint_sha256"] == EXPECTED_MODEL_SHA,
        "T90S_CHECKPOINT_INVALID"
    )

    core = JOB / "td3_core_goal_contrast.py"
    check(
        core.is_file()
        and sha(core) == manifest["core_sha256"],
        "T90S_CORE_INVALID"
    )

    source = SRC.read_text()

    check(
        ast.literal_eval(
            top_node(source, "TASKS").value
        ) == TASKS,
        "ORIGINAL_TASKS_CHANGED"
    )

    # Preserve the complete original T53 reference entry.
    model_node = top_node(source, "MODELS")

    check(
        isinstance(model_node.value, ast.Dict),
        "ORIGINAL_MODELS_INVALID"
    )

    entries = {}

    for key, value in zip(
        model_node.value.keys,
        model_node.value.values
    ):
        name = ast.literal_eval(key)
        entries[name] = ast.get_source_segment(source, value)

    check(
        set(entries) == {"T53", "T87"}
        and entries["T53"] is not None,
        "REFERENCE_MODELS_INVALID"
    )

    models_expression = (
        "{\n"
        "  'T53': " + entries["T53"] + ",\n"
        "  'T90S': {\n"
        f"    'path': Path({str(ckpt)!r}),\n"
        f"    'sha': {EXPECTED_MODEL_SHA!r},\n"
        "    'frames': 190491,\n"
        f"    'core_sha': {manifest['core_sha256']!r},\n"
        "  },\n"
        "}"
    )

    source = replace_top(
        source, "MODELS", models_expression
    )

    # Rename the tested model, preserving evaluator logic.
    source = source.replace("T87", "T90S")

    source = replace_top(
        source,
        "JOB",
        f"Path({str(OUT)!r})"
    )

    check(
        ast.literal_eval(
            top_node(source, "TASKS").value
        ) == TASKS,
        "GENERATED_TASKS_CHANGED"
    )

    check(
        "T87" not in source,
        "OLD_TEST_MODEL_REMAINS"
    )

    ast.parse(source)

    dest = OUT / "evaluate.py"
    check(
        not dest.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    dest.write_text(source)

    print("T90S_FINAL_MODEL=PASS")
    print("T90S_FROZEN_REFERENCE=PASS")
    print("T90S_FOUR_TASKS=PASS")
    print("T90S_EVAL_SHA256=", sha(dest))
    print("T90S_PREFLIGHT=PASS")

except Exception as error:
    print("T90S_PREFLIGHT_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
