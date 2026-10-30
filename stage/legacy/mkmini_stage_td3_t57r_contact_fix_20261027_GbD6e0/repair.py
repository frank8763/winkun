from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

SOURCE = Path(sys.argv[1])
JOB = Path(sys.argv[2])

OLD_SHA = (
    "056bb7c1c7eae707dbcce9a1dc587f526f22ecd6916aea69f713844929b93436"
)

OLD_CALL = (
    "_t57_capture(N.step_checked(env, action))"
)

NEW_CALL = (
    "_t57_capture(N.step_checked, env, action)"
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    print("T57R_STAGE_ONLY=TRUE")
    print("T57R_TRAINING=FALSE")
    print("T57R_MODEL_WRITE=FALSE")
    print("T57R_PLANNER=FALSE")
    print("T57R_ACTION_OVERRIDE=FALSE")
    print("T57R_ENV_MODIFICATION=FALSE")

    original = SOURCE / "evaluate.py"

    check(
        original.is_file()
        and sha(original) == OLD_SHA,
        "ORIGINAL_EVALUATOR_CHANGED"
    )

    text = original.read_text()

    check(
        text.count(OLD_CALL) == 1,
        "ERROR_CALL_ANCHOR_INVALID"
    )

    text = text.replace(
        OLD_CALL,
        NEW_CALL,
        1
    )

    tree = ast.parse(text)

    jobs = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "JOB"
            ):
                jobs.append(node)

    check(
        len(jobs) == 1,
        "JOB_ASSIGNMENT_INVALID"
    )

    node = jobs[0]
    lines = text.splitlines(keepends=True)

    lines[
        node.lineno - 1:node.end_lineno
    ] = [
        f"JOB = Path({str(JOB)!r})\n"
    ]

    result = "".join(lines)
    tree = ast.parse(result)

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_t57_capture"
        )
    ]

    check(
        len(calls) == 1,
        "CAPTURE_CALL_COUNT_INVALID"
    )

    call = calls[0]

    check(
        len(call.args) == 3,
        "CAPTURE_ARGUMENT_COUNT_INVALID"
    )

    first = call.args[0]
    second = call.args[1]
    third = call.args[2]

    check(
        isinstance(first, ast.Attribute)
        and first.attr == "step_checked"
        and isinstance(first.value, ast.Name)
        and first.value.id == "N"
        and isinstance(second, ast.Name)
        and second.id == "env"
        and isinstance(third, ast.Name)
        and third.id == "action",
        "CAPTURE_ARGUMENT_ORDER_INVALID"
    )

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "TARGET_ALREADY_EXISTS"
    )

    target.write_text(result)

    setup = {
        "experiment": "STAGE_TD3_T57R",
        "original_sha256": OLD_SHA,
        "repaired_sha256": sha(target),
        "error_fixed": True,
        "training": False,
        "model_promotion": False,
    }

    (JOB / "t57r_setup.json").write_text(
        json.dumps(
            setup,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    print("T57R_SOURCE_SHA256=", OLD_SHA)

    print(
        "T57R_NEW_SHA256=",
        sha(target)
    )

    print("T57R_CAPTURE_CALL_REPAIR=PASS")
    print("T57R_ARGUMENT_ORDER=PASS")
    print("T57R_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T57R_REPAIR=BLOCKED",
        repr(error)
    )
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
