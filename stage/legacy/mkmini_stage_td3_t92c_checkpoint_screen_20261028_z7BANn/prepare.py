from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T92C_JOB"])

EXPERIMENTS = {
    "T90S": {
        "root": H / (
            "mkmini_stage_td3_t90s_"
            "steer_regularization_20261028_kPT54Q"
        ),
        "evaluator": H / (
            "mkmini_stage_td3_t90s_"
            "steer_regularization_20261028_kPT54Q/"
            "native_eval_EaguFO/evaluate.py"
        ),
        "eval_sha": (
            "085f159ed672a843a49960f1f03f918c6e17d49"
            "c951f83d112efbeda3d40a4d5"
        ),
        "episodes": [23, 24, 25, 26, 27],
    },
    "T87": {
        "root": H / (
            "mkmini_stage_td3_t87_"
            "direction_learning_20261028_Mnr36j"
        ),
        "evaluator": H / (
            "mkmini_stage_td3_t87_"
            "direction_learning_20261028_Mnr36j/"
            "pilot4_verified_UhtIxl/evaluate.py"
        ),
        "eval_sha": (
            "3d23dad58d7c84cc8ecb47a4df09fc8005ca53"
            "a0a9718ef26ae7222621fae276"
        ),
        "episodes": [23, 25, 27],
    },
}

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def top_node(source, name):
    tree = ast.parse(source)

    matches = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == name
            for t in n.targets
        )
    ]

    check(
        len(matches) == 1,
        "BINDING_INVALID:" + name
    )

    return matches[0]

def replace_top(source, name, expression):
    node = top_node(source, name)

    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        name + " = " + expression + "\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

print("=" * 78)
print("T92C — FROZEN STAGE TD3 CHECKPOINT SCREENING")
print("=" * 78)

print("STAGE_ONLY=TRUE")
print("TRAINING=FALSE")
print("MODEL_WRITE=FALSE")
print("PLANNER=FALSE")
print("WAYPOINTS=FALSE")
print("ACTION_OVERRIDE=FALSE")
print("VIDEO_RECORDING=FALSE")

prepared = []

for name, spec in EXPERIMENTS.items():

    try:
        print("\nSOURCE_MODEL=", name)

        root = spec["root"]
        evaluator = spec["evaluator"]

        check(
            evaluator.is_file()
            and sha(evaluator) == spec["eval_sha"],
            "EVALUATOR_IDENTITY_FAILED"
        )

        core_path = (
            root / "td3_core_goal_contrast.py"
        )

        manifest_path = root / "experiment.json"

        check(
            core_path.is_file()
            and manifest_path.is_file(),
            "EXPERIMENT_SOURCE_MISSING"
        )

        experiment = json.loads(
            manifest_path.read_text()
        )

        core_sha = sha(core_path)

        check(
            core_sha == experiment["core_sha256"],
            "CORE_SHA_MISMATCH"
        )

        source = evaluator.read_text()

        check(
            ast.literal_eval(
                top_node(source, "TASKS").value
            ) == TASKS,
            "TASKS_CHANGED"
        )

        model_node = top_node(
            source, "MODELS"
        )

        check(
            isinstance(model_node.value, ast.Dict),
            "MODELS_SCHEMA_INVALID"
        )

        entries = {}

        for key, value in zip(
            model_node.value.keys,
            model_node.value.values
        ):
            label = ast.literal_eval(key)
            entries[label] = ast.get_source_segment(
                source, value
            )

        check(
            set(entries) == {"T53", name}
            and entries["T53"] is not None,
            "FROZEN_REFERENCE_INVALID"
        )

        for episode in spec["episodes"]:

            checkpoint = (
                root / "checkpoints"
                / f"episode_{episode:04d}.pt"
            )

            if not checkpoint.is_file():
                print(
                    "SNAPSHOT_MISSING=",
                    name, episode
                )
                continue

            saved = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False
            )

            frames = int(saved["frames"])

            check(
                134491 < frames <= 190491,
                "SNAPSHOT_FRAMES_INVALID"
            )

            check(
                saved["manifest"]["source_core"]
                == core_sha,
                "SNAPSHOT_CORE_MISMATCH"
            )

            check(
                saved["manifest"]["experiment"]
                == f"STAGE_TD3_{name}",
                "SNAPSHOT_EXPERIMENT_MISMATCH"
            )

            checkpoint_sha = sha(checkpoint)

            case = (
                JOB / "cases"
                / f"{name}_ep{episode:02d}"
            )

            case.mkdir(
                parents=True,
                exist_ok=False
            )

            model_entry = (
                "{\n"
                f" 'path': Path({str(checkpoint)!r}),\n"
                f" 'sha': {checkpoint_sha!r},\n"
                f" 'frames': {frames},\n"
                f" 'core_sha': {core_sha!r},\n"
                "}"
            )

            models = (
                "{\n"
                f" 'T53': {entries['T53']},\n"
                f" {name!r}: {model_entry},\n"
                "}"
            )

            generated = replace_top(
                source,
                "JOB",
                f"Path({str(case)!r})"
            )

            generated = replace_top(
                generated,
                "MODELS",
                models
            )

            ast.parse(generated)

            check(
                ast.literal_eval(
                    top_node(
                        generated, "TASKS"
                    ).value
                ) == TASKS,
                "GENERATED_TASKS_CHANGED"
            )

            script = case / "evaluate.py"

            script.write_text(generated)

            identity = {
                "name": name,
                "episode": episode,
                "frames": frames,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256":
                    checkpoint_sha,
                "core_sha256": core_sha,
                "evaluator_sha256": sha(script),
                "training": False,
                "automatic_promotion": False,
            }

            (case / "identity.json").write_text(
                json.dumps(
                    identity,
                    indent=2
                ) + "\n"
            )

            prepared.append(identity)

            print(
                "T92C_SNAPSHOT_READY",
                name,
                "EPISODE=", episode,
                "FRAMES=", frames,
                "SHA=", checkpoint_sha[:16],
                flush=True
            )

    except Exception as error:
        print(
            "T92C_SOURCE_BLOCKED",
            name,
            repr(error)
        )
        traceback.print_exc()

(JOB / "prepared.json").write_text(
    json.dumps(
        prepared,
        indent=2
    ) + "\n"
)

print("\nT92C_PREPARED=", len(prepared))

if prepared:
    print("T92C_PREFLIGHT=PASS")
else:
    print("T92C_PREFLIGHT=FAIL")

print("MODEL_PROMOTED=FALSE")
print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
