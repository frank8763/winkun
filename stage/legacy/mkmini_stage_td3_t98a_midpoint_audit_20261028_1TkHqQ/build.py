from pathlib import Path
import ast
import hashlib
import json
import os
import traceback
import torch

H = Path.home()
ROOT = Path(os.environ["T98A_JOB"])
T96 = Path(os.environ["T96M_JOB"])

SOURCE = H / (
    "mkmini_stage_td3_t97_native4_"
    "20261028_S9JODK/evaluate.py"
)

SOURCE_SHA = (
    "053dd18d315c06fa6dac72c0c918524915d187f00bd489b85cbdfbc70c86cae2"
)

CORE_SHA = (
    "59b3b27bce8c3eeefc4c1b12989139a8fbe00474cc2784df0d8f5a5ec4353a64"
)

SNAPSHOTS = [
    (
        26,
        178926,
        "ce5400573123025140337656b33637104942d8befd2d1870fc397fcce2e7e6e0",
    ),
    (
        27,
        189533,
        "f16169414c8a51aec84834e4c37836723de7a8dc706df60a02a276d8e8962646",
    ),
]

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def top(source, name):
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
    node = top(source, name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]
    result = "".join(lines)
    ast.parse(result)
    return result

try:
    print("=" * 72)
    print("T98A STAGE TD3 INTERMEDIATE CHECKPOINT EVALUATION")
    print("=" * 72)

    check(
        SOURCE.is_file() and sha(SOURCE) == SOURCE_SHA,
        "VERIFIED_EVALUATOR_CHANGED"
    )

    source = SOURCE.read_text()

    check(
        ast.literal_eval(top(source, "TASKS").value) == TASKS,
        "TASKS_CHANGED"
    )

    model_node = top(source, "MODELS")
    check(
        isinstance(model_node.value, ast.Dict),
        "MODELS_INVALID"
    )

    models = {}
    for key, value in zip(
        model_node.value.keys,
        model_node.value.values,
    ):
        name = ast.literal_eval(key)
        models[name] = ast.get_source_segment(source, value)

    check(
        set(models) == {"T53", "T96M"},
        "REFERENCE_MODELS_CHANGED"
    )

    cases = []

    for episode, frames, expected_sha in SNAPSHOTS:

        checkpoint = (
            T96 / "checkpoints"
            / f"episode_{episode:04d}.pt"
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint) == expected_sha,
            f"CHECKPOINT_IDENTITY_FAILED_EP{episode}"
        )

        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        check(
            int(saved["frames"]) == frames,
            "CHECKPOINT_FRAMES_CHANGED"
        )

        check(
            saved["manifest"]["experiment"]
            == "STAGE_TD3_T96M",
            "CHECKPOINT_EXPERIMENT_CHANGED"
        )

        check(
            saved["manifest"]["source_core"] == CORE_SHA,
            "CHECKPOINT_CORE_CHANGED"
        )

        del saved

        case = ROOT / "cases" / f"episode_{episode:04d}"

        case.mkdir(
            parents=True,
            exist_ok=False,
        )

        candidate = (
            "{\n"
            " 'T53': " + models["T53"] + ",\n"
            " 'T96M': {\n"
            f"  'path': Path({str(checkpoint)!r}),\n"
            f"  'sha': {expected_sha!r},\n"
            f"  'frames': {frames},\n"
            f"  'core_sha': {CORE_SHA!r},\n"
            " },\n"
            "}"
        )

        generated = replace_top(
            source, "MODELS", candidate
        )

        generated = replace_top(
            generated,
            "JOB",
            f"Path({str(case)!r})"
        )

        check(
            ast.literal_eval(
                top(generated, "TASKS").value
            ) == TASKS,
            "GENERATED_TASKS_CHANGED"
        )

        ast.parse(generated)

        script = case / "evaluate.py"

        check(
            not script.exists(),
            "EVALUATOR_ALREADY_EXISTS"
        )

        script.write_text(generated)

        identity = {
            "episode": episode,
            "frames": frames,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_sha,
            "evaluator_sha256": sha(script),
            "training": False,
            "automatic_promotion": False,
        }

        (case / "identity.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )

        cases.append(identity)

        print(
            "T98_SNAPSHOT_READY=PASS",
            "EPISODE=", episode,
            "FRAMES=", frames,
            "SHA=", expected_sha,
        )

    check(
        len(cases) == 2,
        "INCOMPLETE_SNAPSHOT_SET"
    )

    print("T98_PREFLIGHT=PASS")

except Exception as e:
    print("T98_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()

finally:
    print("TRAINING=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
