from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import traceback

OLD = Path(os.environ["T78_JOB"])
NEW = Path(os.environ["T78R_JOB"])
T77 = Path(os.environ["T77_JOB"])

EXPECTED_EVAL_SHA = (
    "8bf441e5be375530d6234eb7aacf70c2"
    "dd097768245c5411d8669c63dfba49bc"
)

def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(block)

    return h.hexdigest()

def check(value, message):
    if not value:
        raise RuntimeError(message)

def replace_job(source, new_path):
    tree = ast.parse(source)

    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == "JOB"
            for t in n.targets
        )
    ]

    check(
        len(nodes) == 1,
        "JOB_BINDING_NOT_UNIQUE"
    )

    node = nodes[0]

    lines = source.splitlines(
        keepends=True
    )

    lines[
        node.lineno - 1:node.end_lineno
    ] = [
        "JOB = Path("
        + repr(str(new_path))
        + ")\n"
    ]

    result = "".join(lines)
    ast.parse(result)

    return result

try:
    print("=" * 75)
    print("T78R — STRICT EVALUATOR MANIFEST RECOVERY")
    print("=" * 75)

    source_path = OLD / "evaluate.py"
    old_manifest_path = OLD / "manifest.json"

    check(
        source_path.is_file()
        and sha(source_path)
            == EXPECTED_EVAL_SHA,
        "T78_EVALUATOR_HASH_CHANGED"
    )

    check(
        old_manifest_path.is_file(),
        "T78_BUILDER_MANIFEST_MISSING"
    )

    builder_manifest = json.loads(
        old_manifest_path.read_text()
    )

    check(
        builder_manifest.get("experiment")
            == "T78"
        and builder_manifest.get("model")
            == "T77",
        "BUILDER_MANIFEST_IDENTITY_INVALID"
    )

    status = json.loads(
        (T77 / "status.json").read_text()
    )

    checkpoint = (
        T77 / "checkpoints"
        / status["checkpoint"]
    )

    check(
        int(status["frames"]) == 190491
        and checkpoint.is_file()
        and sha(checkpoint)
            == status["checkpoint_sha256"],
        "T77_CHECKPOINT_CHANGED"
    )

    check(
        builder_manifest.get(
            "checkpoint_sha256"
        ) == status["checkpoint_sha256"],
        "T78_T77_BINDING_MISMATCH"
    )

    source = source_path.read_text()

    check(
        "T71E_MANIFEST_CHANGED" in source,
        "MANIFEST_GUARD_NOT_FOUND"
    )

    check(
        "T77" in source,
        "T77_MODEL_BINDING_MISSING"
    )

    check(
        "T71B" not in source,
        "OLD_MODEL_REFERENCE_REMAINS"
    )

    check(
        str(checkpoint) in source,
        "T77_CHECKPOINT_NOT_REFERENCED"
    )

    check(
        status["checkpoint_sha256"] in source,
        "T77_SHA_NOT_REFERENCED"
    )

    check(
        not (NEW / "evaluate.py").exists(),
        "T78R_ALREADY_PREPARED"
    )

    updated = replace_job(
        source,
        NEW
    )

    check(
        replace_job(updated, OLD) == source,
        "UNEXPECTED_EVALUATOR_MODIFICATION"
    )

    output = NEW / "evaluate.py"
    output.write_text(updated)

    # 保留上次构建记录，但不让它占用
    # 原始评估器使用的manifest.json名称。
    shutil.copy2(
        old_manifest_path,
        NEW / "builder_manifest_original.json"
    )

    check(
        not (NEW / "manifest.json").exists(),
        "EVALUATOR_MANIFEST_ALREADY_EXISTS"
    )

    print(
        "T77_CHECKPOINT_SHA256=",
        status["checkpoint_sha256"]
    )

    print(
        "ORIGINAL_EVALUATOR_SHA256=",
        sha(source_path)
    )

    print(
        "RECOVERED_EVALUATOR_SHA256=",
        sha(output)
    )

    print(
        "T78R_ONLY_CHANGED=JOB_OUTPUT_DIRECTORY"
    )

    print(
        "T78R_EVALUATOR_MANIFEST_PREWRITTEN=FALSE"
    )

    print("T78R_SOURCE_GATE=PASS")

except Exception as error:
    print(
        "T78R_SOURCE_GATE_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "TRAINING_LAUNCHED=FALSE"
    )
    print(
        "FROZEN_MODEL_MODIFIED=FALSE"
    )
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
