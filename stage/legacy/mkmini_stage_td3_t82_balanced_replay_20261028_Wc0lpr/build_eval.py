from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

HOME = Path.home()
JOB = Path(os.environ["T82_JOB"])
EVAL = JOB / "pilot4"

SOURCE = HOME / (
    "mkmini_stage_td3_t71_actor_ablation_"
    "20261028_dE1arY/T71B_pilot4/evaluate.py"
)

EXPECTED_SOURCE_SHA = (
    "81ac87df1d98900fd8568cd3a5c8dee7a20547bcf3dfd"
    "323817251593acebea5"
)

T53 = HOME / (
    "mkmini_stage_td3_t53_actor_anchor_20261027_v1/"
    "checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744"
    "d25259560d1e936e4e8e0"
)

T53_CORE = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520"
    "fb4f136a6a09c79a07"
)

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def get_top(source, name):
    tree = ast.parse(source)

    found = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == name
            for t in n.targets
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
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)

    ast.parse(result)
    return result

def main():
    print("T82_EVALUATION_PREFLIGHT_START")

    status = json.loads(
        (JOB / "status.json").read_text()
    )

    check(
        status["state"] == "PILOT_COMPLETED_NONCOLLAPSED"
        and int(status["frames"]) == 190491,
        "TRAINING_NOT_COMPLETED"
    )

    ck = (
        JOB / "checkpoints"
        / status["checkpoint"]
    )

    check(
        ck.is_file()
        and sha(ck) == status["checkpoint_sha256"],
        "T82_CHECKPOINT_INVALID"
    )

    core = JOB / "td3_core_goal_contrast.py"

    core_sha = sha(core)

    experiment = json.loads(
        (JOB / "experiment.json").read_text()
    )

    check(
        core_sha == experiment["core_sha256"],
        "T82_CORE_CHANGED"
    )

    check(
        SOURCE.is_file()
        and sha(SOURCE) == EXPECTED_SOURCE_SHA,
        "NATIVE_EVALUATOR_CHANGED"
    )

    check(
        T53.is_file()
        and sha(T53) == T53_SHA,
        "T53_REFERENCE_CHANGED"
    )

    source = SOURCE.read_text()

    check(
        ast.literal_eval(
            get_top(source, "TASKS").value
        ) == TASKS,
        "FOUR_SCENES_CHANGED"
    )

    source = source.replace(
        "T71B",
        "T82"
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(EVAL)!r})"
    )

    models = (
        "{\n"
        "  'T53': {\n"
        f"    'path': Path({str(T53)!r}),\n"
        f"    'sha': {T53_SHA!r},\n"
        "    'frames': 22491,\n"
        f"    'core_sha': {T53_CORE!r},\n"
        "  },\n"
        "  'T82': {\n"
        f"    'path': Path({str(ck)!r}),\n"
        f"    'sha': {status['checkpoint_sha256']!r},\n"
        "    'frames': 190491,\n"
        f"    'core_sha': {core_sha!r},\n"
        "  },\n"
        "}"
    )

    source = replace_top(
        source,
        "MODELS",
        models
    )

    check(
        ast.literal_eval(
            get_top(source, "TASKS").value
        ) == TASKS,
        "GENERATED_TASKS_CHANGED"
    )

    check(
        "T71B" not in source,
        "STALE_MODEL_REFERENCE"
    )

    ast.parse(source)

    target = EVAL / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_PREPARED"
    )

    check(
        not (EVAL / "manifest.json").exists(),
        "MANIFEST_ALREADY_PRESENT"
    )

    target.write_text(source)

    print(
        "T82_MODEL_SHA256=",
        status["checkpoint_sha256"]
    )
    print(
        "T82_CORE_SHA256=",
        core_sha
    )
    print(
        "T82_EVALUATOR_SHA256=",
        sha(target)
    )
    print("T82_FOUR_SCENES_UNCHANGED=PASS")
    print("T82_EVAL_PREFLIGHT=PASS")

try:
    main()

except Exception as error:
    print(
        "T82_EVAL_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
