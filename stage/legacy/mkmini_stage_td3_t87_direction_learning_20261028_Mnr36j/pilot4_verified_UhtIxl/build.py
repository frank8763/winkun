from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

H = Path.home()
JOB = Path(os.environ["T87_JOB"])
OUT = Path(os.environ["T87_EVAL"])

SOURCE = H / (
    "mkmini_stage_td3_t82_balanced_replay_"
    "20261028_Wc0lpr/pilot4/evaluate.py"
)

EVAL_SHA = (
    "aea31b20c9f92ece56010360dce8f222a8ba5693"
    "b356ce04bb5715ffd3d1fa18"
)

T87_SHA = (
    "06fad7346c51c4055aff8e867d0e956f38cb9feef"
    "68009015212fe7d96f6342c"
)

T53 = H / (
    "mkmini_stage_td3_t53_actor_anchor_20261027_v1/"
    "checkpoints/episode_0010.pt"
)

T53_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744"
    "d25259560d1e936e4e8e0"
)

T53_CORE = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967"
    "d9520fb4f136a6a09c79a07"
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
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def node_for(source, name):
    tree = ast.parse(source)
    matches = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]
    check(
        len(matches) == 1,
        "BINDING_INVALID:" + name
    )
    return matches[0]

def replace_top(source, name, expression):
    node = node_for(source, name)
    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    output = "".join(lines)
    ast.parse(output)
    return output

try:
    print("=" * 78)
    print("T87 NATIVE MUJOCO PILOT4 VALIDATION")
    print("=" * 78)

    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    check(
        SOURCE.is_file()
        and sha(SOURCE) == EVAL_SHA,
        "ORIGINAL_EVALUATOR_CHANGED"
    )

    status = json.loads(
        (JOB / "status.json").read_text()
    )

    report = json.loads(
        (JOB / "experiment.json").read_text()
    )

    check(
        status.get("state")
        == "PILOT_COMPLETED_NONCOLLAPSED",
        "T87_NOT_FINISHED"
    )

    check(
        int(status.get("frames", -1)) == 190491,
        "T87_FRAME_COUNT_INVALID"
    )

    ck = (
        JOB / "checkpoints"
        / status["checkpoint"]
    )

    check(
        ck.is_file()
        and sha(ck) == T87_SHA
        and status["checkpoint_sha256"] == T87_SHA,
        "T87_CHECKPOINT_CHANGED"
    )

    core = JOB / "td3_core_goal_contrast.py"
    core_sha = sha(core)

    check(
        core_sha == report["core_sha256"],
        "T87_CORE_CHANGED"
    )

    check(
        T53.is_file()
        and sha(T53) == T53_SHA,
        "T53_REFERENCE_CHANGED"
    )

    source = SOURCE.read_text()

    check(
        ast.literal_eval(
            node_for(source, "TASKS").value
        ) == TASKS,
        "ORIGINAL_TASKS_CHANGED"
    )

    # 保持原始评估逻辑，只重新绑定实验模型。
    source = source.replace("T82", "T87")

    source = replace_top(
        source,
        "JOB",
        f"Path({str(OUT)!r})"
    )

    models = {
        "T53": {
            "path": T53,
            "sha": T53_SHA,
            "frames": 22491,
            "core_sha": T53_CORE,
        },
        "T87": {
            "path": ck,
            "sha": T87_SHA,
            "frames": 190491,
            "core_sha": core_sha,
        },
    }

    expression = "{\n"

    for name, item in models.items():
        expression += (
            f"  {name!r}: {{\n"
            f"    'path': Path({str(item['path'])!r}),\n"
            f"    'sha': {item['sha']!r},\n"
            f"    'frames': {item['frames']},\n"
            f"    'core_sha': {item['core_sha']!r},\n"
            "  },\n"
        )

    expression += "}"

    source = replace_top(
        source,
        "MODELS",
        expression
    )

    check(
        ast.literal_eval(
            node_for(source, "TASKS").value
        ) == TASKS,
        "GENERATED_TASKS_CHANGED"
    )

    check(
        "T82" not in source,
        "OLD_MODEL_BINDING_REMAINS"
    )

    ast.parse(source)

    destination = OUT / "evaluate.py"

    check(
        not destination.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    destination.write_text(source)

    print("T87_CHECKPOINT_SHA=PASS")
    print("T87_CORE_SHA=PASS")
    print("T87_ORIGINAL_TASKS=PASS")
    print("T87_EVALUATOR_SHA=", sha(destination))
    print("T87_PREFLIGHT=PASS")

except Exception as error:
    print(
        "T87_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
