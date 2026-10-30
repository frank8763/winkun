from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
T71 = Path(os.environ["T71_JOB"])
T77 = Path(os.environ["T77_JOB"])
JOB = Path(os.environ["T78_JOB"])

SRC = T71 / "T71B_pilot4/evaluate.py"

T53_CK = H / (
    "mkmini_stage_td3_t53_actor_anchor_20261027_v1/"
    "checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d252"
    "59560d1e936e4e8e0"
)

T53_CORE = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520"
    "fb4f136a6a09c79a07"
)

EXPECTED_EVAL_SHA = (
    "81ac87df1d98900fd8568cd3a5c8dee7a20547bcf3dfd"
    "323817251593acebea5"
)

EXPECTED_TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()

def check(condition, message):
    if not condition:
        raise RuntimeError(message)

def top_assignment(source, name):
    tree = ast.parse(source)

    found = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == name
            for target in node.targets
        )
    ]

    check(
        len(found) == 1,
        "TOP_ASSIGNMENT_INVALID:" + name
    )

    return found[0]

def replace_top(source, name, expression):
    node = top_assignment(source, name)
    lines = source.splitlines(keepends=True)

    lines[node.lineno - 1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)

    return result

def main():
    print("=" * 76)
    print("T78 — STAGE TD3 T77 STRICT PILOT4 EVALUATION")
    print("=" * 76)

    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("FROZEN_MODEL_WRITE=FALSE")

    check(
        SRC.is_file()
        and sha(SRC) == EXPECTED_EVAL_SHA,
        "ORIGINAL_EVALUATOR_CHANGED"
    )

    status = json.loads(
        (T77 / "status.json").read_text()
    )

    check(
        status["state"]
            == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == 190491,
        "T77_TRAINING_STATE_INVALID"
    )

    ck = (
        T77 / "checkpoints"
        / status["checkpoint"]
    )

    check(
        ck.is_file()
        and sha(ck) == status["checkpoint_sha256"],
        "T77_CHECKPOINT_SHA_INVALID"
    )

    state = torch.load(
        ck,
        map_location="cpu",
        weights_only=False
    )

    manifest = state["manifest"]

    check(
        int(state["frames"]) == 190491
        and manifest["experiment"] == "STAGE_TD3_T77"
        and manifest["algorithm"] == "TD3",
        "T77_CHECKPOINT_IDENTITY_INVALID"
    )

    core = T77 / "td3_core_goal_contrast.py"

    check(
        core.is_file(),
        "T77_CORE_SOURCE_MISSING"
    )

    core_sha = sha(core)

    check(
        core_sha == manifest["source_core"],
        "T77_CORE_MANIFEST_MISMATCH"
    )

    check(
        T53_CK.is_file()
        and sha(T53_CK) == T53_SHA,
        "T53_FROZEN_REFERENCE_INVALID"
    )

    original = SRC.read_text()
    ast.parse(original)

    tasks_node = top_assignment(
        original,
        "TASKS"
    )

    tasks = ast.literal_eval(
        tasks_node.value
    )

    check(
        tasks == EXPECTED_TASKS,
        "ORIGINAL_FOUR_SCENES_CHANGED"
    )

    models_node = top_assignment(
        original,
        "MODELS"
    )

    model_keys = [
        ast.literal_eval(k)
        for k in models_node.value.keys
    ]

    check(
        model_keys == ["T53", "T71B"],
        "ORIGINAL_MODEL_BINDINGS_CHANGED"
    )

    check(
        original.count("T71B") >= 1,
        "ORIGINAL_MODEL_NAME_MISSING"
    )

    # 把评估标签改成T77。
    # 不改变评估函数、任务、物理模型和动作。
    updated = original.replace(
        "T71B",
        "T77"
    )

    updated = replace_top(
        updated,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    model_expression = (
        "{\n"
        "  'T53': {\n"
        f"    'path': Path({str(T53_CK)!r}),\n"
        f"    'sha': {T53_SHA!r},\n"
        "    'frames': 22491,\n"
        f"    'core_sha': {T53_CORE!r},\n"
        "  },\n"
        "  'T77': {\n"
        f"    'path': Path({str(ck)!r}),\n"
        f"    'sha': {status['checkpoint_sha256']!r},\n"
        "    'frames': 190491,\n"
        f"    'core_sha': {core_sha!r},\n"
        "  },\n"
        "}"
    )

    updated = replace_top(
        updated,
        "MODELS",
        model_expression
    )

    ast.parse(updated)

    check(
        "T71B" not in updated,
        "STALE_T71B_REFERENCE"
    )

    check(
        str(ck) in updated
        and status["checkpoint_sha256"] in updated,
        "T77_MODEL_NOT_BOUND"
    )

    check(
        str(JOB) in updated,
        "T78_OUTPUT_DIRECTORY_NOT_BOUND"
    )

    check(
        ast.literal_eval(
            top_assignment(
                updated,
                "TASKS"
            ).value
        ) == EXPECTED_TASKS,
        "T78_TASKS_CHANGED"
    )

    output = JOB / "evaluate.py"

    check(
        not output.exists(),
        "T78_EVALUATOR_ALREADY_EXISTS"
    )

    output.write_text(updated)

    report = {
        "experiment": "T78",
        "model": "T77",
        "source_model": "T71B_evaluator",
        "test": "pilot4",
        "frames": 190491,
        "checkpoint": str(ck),
        "checkpoint_sha256":
            status["checkpoint_sha256"],
        "core_sha256": core_sha,
        "original_evaluator_sha256":
            EXPECTED_EVAL_SHA,
        "generated_evaluator_sha256":
            sha(output),
        "tasks": EXPECTED_TASKS,
        "training": False,
        "action_override": False,
        "automatic_promotion": False
    }

    (JOB / "manifest.json").write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print("T78_T77_CHECKPOINT=PASS")
    print("T78_T77_CORE=PASS")
    print("T78_T53_REFERENCE=PASS")
    print("T78_ORIGINAL_EVALUATOR=PASS")
    print("T78_FOUR_SCENES=PASS")
    print("T78_NEW_EVALUATOR_SHA256=", sha(output))
    print("T78_EVALUATOR_MODEL=T77")
    print("T78_PREFLIGHT=PASS")

try:
    main()

except Exception as error:
    print(
        "T78_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
