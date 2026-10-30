from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

H = Path.home()
JOB = Path(os.environ["T103_JOB"])

SOURCE_DIR = H / (
    "mkmini_stage_td3_t102_policy_interpolation_"
    "20261028_pklqP9/T102_B35"
)

SOURCE = SOURCE_DIR / "evaluate.py"
IDENTITY = SOURCE_DIR / "identity.json"
CHECKPOINT = SOURCE_DIR / "candidate.pt"

EXPECTED_CKPT_SHA = (
    "699d4a1449662ccc5557b2dc689b09f41c9043ee51c41116c5b0cc4ab98ca566"
)

EXPECTED_TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def check(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()

def top(source, name):
    matches = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == name
            for target in node.targets
        )
    ]

    check(len(matches) == 1, "BINDING_INVALID:" + name)

    return matches[0]

def replace_top(source, name, expression):
    node = top(source, name)

    lines = source.splitlines(keepends=True)

    lines[node.lineno - 1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

try:
    print("=" * 76)
    print("T103 — STAGE TD3 EXTENDED-HORIZON DIAGNOSIS")
    print("=" * 76)

    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("FORMAL_SUCCESS_GATE_UNCHANGED=TRUE")

    check(
        SOURCE.is_file()
        and IDENTITY.is_file()
        and CHECKPOINT.is_file(),
        "SOURCE_FILES_MISSING"
    )

    identity = json.loads(IDENTITY.read_text())

    check(
        identity["candidate"] == "T102_B35",
        "CANDIDATE_IDENTITY_INVALID"
    )

    check(
        identity["alpha"] == 0.35,
        "ALPHA_INVALID"
    )

    check(
        identity["checkpoint_sha256"]
        == EXPECTED_CKPT_SHA,
        "IDENTITY_CHECKPOINT_SHA_INVALID"
    )

    check(
        sha(CHECKPOINT) == EXPECTED_CKPT_SHA,
        "CHECKPOINT_CHANGED"
    )

    check(
        sha(SOURCE) == identity["evaluator_sha256"],
        "EVALUATOR_CHANGED"
    )

    original = SOURCE.read_text()

    check(
        ast.literal_eval(top(original, "TASKS").value)
        == EXPECTED_TASKS,
        "ORIGINAL_TASKS_CHANGED"
    )

    check(
        ast.literal_eval(top(original, "CAP").value)
        == 14000,
        "ORIGINAL_CAP_CHANGED"
    )

    models_node = top(original, "MODELS")

    check(
        isinstance(models_node.value, ast.Dict),
        "MODEL_TABLE_INVALID"
    )

    names = {
        ast.literal_eval(key)
        for key in models_node.value.keys
    }

    check(
        names == {"T53", "T102_B35"},
        "MODEL_PAIR_CHANGED"
    )

    # 仅更改诊断副本的时间预算及输出目录。
    generated = replace_top(
        original,
        "CAP",
        "22000"
    )

    generated = replace_top(
        generated,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    check(
        ast.literal_eval(
            top(generated, "TASKS").value
        ) == EXPECTED_TASKS,
        "GENERATED_TASKS_CHANGED"
    )

    check(
        ast.literal_eval(
            top(generated, "CAP").value
        ) == 22000,
        "DIAGNOSTIC_CAP_INVALID"
    )

    # 禁止无意更改其他源码逻辑。
    reconstructed = replace_top(
        generated,
        "JOB",
        ast.get_source_segment(
            original,
            top(original, "JOB").value
        )
    )

    reconstructed = replace_top(
        reconstructed,
        "CAP",
        "14000"
    )

    check(
        reconstructed == original,
        "UNINTENDED_SOURCE_MODIFICATION"
    )

    output = JOB / "evaluate.py"

    check(
        not output.exists(),
        "OUTPUT_ALREADY_EXISTS"
    )

    output.write_text(generated)

    report = {
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "candidate": str(CHECKPOINT),
        "candidate_sha256": EXPECTED_CKPT_SHA,
        "formal_cap": 14000,
        "diagnostic_cap": 22000,
        "evaluator_sha256": sha(output),
        "training": False,
        "action_override": False,
        "eligible_for_promotion": False,
    }

    (JOB / "identity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("T103_MODEL_VERIFIED=PASS")
    print("T103_SOURCE_VERIFIED=PASS")
    print("T103_FOUR_TASKS_UNCHANGED=PASS")
    print("T103_ONLY_CAP_AND_OUTPUT_CHANGED=PASS")
    print("T103_DIAGNOSTIC_ONLY=TRUE")
    print("T103_PREFLIGHT=PASS")

except Exception as error:
    print("T103_PREFLIGHT_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("TRAINING_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
