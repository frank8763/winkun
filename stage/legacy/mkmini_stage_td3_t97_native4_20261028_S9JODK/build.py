from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T97_JOB"])
T96 = Path(os.environ["T96M_JOB"])

SOURCE = H / (
    "mkmini_stage_td3_t90s_steer_regularization_"
    "20261028_kPT54Q/native_eval_EaguFO/evaluate.py"
)

SOURCE_SHA = (
    "085f159ed672a843a49960f1f03f918c6e17d49"
    "c951f83d112efbeda3d40a4d5"
)

MODEL_SHA = (
    "97a2cc3d9999df2f658b2bdf8f8ccb33c98cbcb"
    "80ff63ef2fb140c6444e7d913"
)

CORE_SHA = (
    "59b3b27bce8c3eeefc4c1b12989139a8fbe00"
    "474cc2784df0d8f5a5ec4353a64"
)

TASKS = [
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

def get_top(source, name):
    tree = ast.parse(source)

    found = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == name
            for target in node.targets
        )
    ]

    check(
        len(found) == 1,
        "TOP_BINDING_INVALID:" + name
    )

    return found[0]

def replace_top(source, name, expression):
    node = get_top(source, name)

    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        name + " = " + expression + "\n"
    ]

    result = "".join(lines)

    ast.parse(result)

    return result

try:
    print("=" * 76)
    print("T97 — STAGE TD3 T96M NATIVE MUJOCO EVALUATION")
    print("=" * 76)

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    check(
        SOURCE.is_file()
        and sha(SOURCE) == SOURCE_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    status_path = T96 / "status.json"
    manifest_path = T96 / "experiment.json"
    core_path = T96 / "td3_core_goal_contrast.py"

    check(
        status_path.is_file()
        and manifest_path.is_file()
        and core_path.is_file(),
        "T96_EXPERIMENT_FILES_MISSING"
    )

    status = json.loads(
        status_path.read_text()
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    check(
        status.get("state")
        == "PILOT_COMPLETED_NONCOLLAPSED",
        "T96_TRAINING_NOT_COMPLETE"
    )

    check(
        int(status.get("frames", -1))
        == 190491,
        "T96_FRAMES_INVALID"
    )

    check(
        status.get("checkpoint")
        == "episode_0028.pt",
        "T96_CHECKPOINT_NAME_INVALID"
    )

    check(
        status.get("checkpoint_sha256")
        == MODEL_SHA,
        "T96_STATUS_HASH_MISMATCH"
    )

    check(
        manifest.get("experiment")
        == "STAGE_TD3_T96M",
        "T96_EXPERIMENT_INVALID"
    )

    check(
        manifest.get("core_sha256")
        == CORE_SHA,
        "T96_MANIFEST_CORE_INVALID"
    )

    check(
        sha(core_path) == CORE_SHA,
        "T96_CORE_HASH_MISMATCH"
    )

    checkpoint = (
        T96 / "checkpoints"
        / status["checkpoint"]
    )

    check(
        checkpoint.is_file()
        and sha(checkpoint) == MODEL_SHA,
        "T96_MODEL_HASH_MISMATCH"
    )

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(saved["frames"]) == 190491,
        "CHECKPOINT_FRAMES_INVALID"
    )

    check(
        saved["manifest"]["experiment"]
        == "STAGE_TD3_T96M",
        "CHECKPOINT_EXPERIMENT_INVALID"
    )

    check(
        saved["manifest"]["source_core"]
        == CORE_SHA,
        "CHECKPOINT_CORE_INVALID"
    )

    del saved

    source = SOURCE.read_text()

    check(
        ast.literal_eval(
            get_top(source, "TASKS").value
        ) == TASKS,
        "EVALUATION_TASKS_CHANGED"
    )

    model_node = get_top(
        source, "MODELS"
    )

    check(
        isinstance(model_node.value, ast.Dict),
        "MODEL_TABLE_INVALID"
    )

    entries = {}

    for key, value in zip(
        model_node.value.keys,
        model_node.value.values
    ):
        name = ast.literal_eval(key)

        entries[name] = (
            ast.get_source_segment(
                source, value
            )
        )

    check(
        set(entries) == {"T53", "T90S"}
        and entries["T53"] is not None,
        "REFERENCE_MODEL_CHANGED"
    )

    # 保留原T53基线；
    # 仅替换被测试的T90S模型为T96M。
    models_expression = (
        "{\n"
        " 'T53': "
        + entries["T53"]
        + ",\n"
        " 'T96M': {\n"
        f"  'path': Path({str(checkpoint)!r}),\n"
        f"  'sha': {MODEL_SHA!r},\n"
        "  'frames': 190491,\n"
        f"  'core_sha': {CORE_SHA!r},\n"
        " },\n"
        "}"
    )

    generated = replace_top(
        source,
        "MODELS",
        models_expression
    )

    generated = replace_top(
        generated,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    # 修改模型标签，不修改物理模型或控制动作。
    generated = generated.replace(
        "T90S", "T96M"
    )

    ast.parse(generated)

    check(
        ast.literal_eval(
            get_top(generated, "TASKS").value
        ) == TASKS,
        "GENERATED_TASKS_CHANGED"
    )

    check(
        "T90S" not in generated,
        "OLD_TEST_MODEL_REMAINS"
    )

    output = JOB / "evaluate.py"

    check(
        not output.exists(),
        "T97_EVALUATOR_ALREADY_EXISTS"
    )

    output.write_text(generated)

    check(
        not any(JOB.glob("*manifest*")),
        "T97_MANIFEST_ALREADY_EXISTS"
    )

    print("T97_T96_MODEL_SHA=PASS")
    print("T97_T96_CORE_SHA=PASS")
    print("T97_TRAINING_COMPLETE=PASS")
    print("T97_T53_REFERENCE=PASS")
    print("T97_ORIGINAL_FOUR_TASKS=PASS")
    print("T97_NEW_MANIFEST_DIRECTORY=PASS")
    print("T97_EVALUATOR_SHA256=", sha(output))
    print("T97_PREFLIGHT=PASS")

except Exception as error:
    print(
        "T97_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
