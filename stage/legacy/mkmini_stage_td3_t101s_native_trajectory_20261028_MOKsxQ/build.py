from pathlib import Path
import ast
import hashlib
import json
import os
import traceback
import torch

H = Path.home()
ROOT = Path(os.environ["T101S_JOB"])

SOURCE = H / (
    "mkmini_stage_td3_t91b_manifest_fix_"
    "20261028_EYmRyJ/evaluate.py"
)

SOURCE_SHA = (
    "5954c2b1714ab0eca45d6d29bb6373bfef81f"
    "8808ea036b6a55108fbb86359dd"
)

CORE_SHA = (
    "59b3b27bce8c3eeefc4c1b12989139a8fbe004"
    "74cc2784df0d8f5a5ec4353a64"
)

MODELS = {
    "T87A": {
        "path": H / (
            "mkmini_stage_td3_t87_direction_learning_"
            "20261028_Mnr36j/checkpoints/episode_0025.pt"
        ),
        "sha": (
            "0378e4af4296683d5d612a2d71607149e8fb"
            "3b0ff5235e35643766059ffaf94e"
        ),
        "frames": 171387,
        "experiment": "STAGE_TD3_T87",
    },
    "T96A": {
        "path": H / (
            "mkmini_stage_td3_t96m_balanced_sampling_"
            "20261028_zgWvJE/checkpoints/episode_0026.pt"
        ),
        "sha": (
            "ce5400573123025140337656b33637104942d"
            "8befd2d1870fc397fcce2e7e6e0"
        ),
        "frames": 178926,
        "experiment": "STAGE_TD3_T96M",
    },
}

ORIGINAL_TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

TRACE_SEEDS = {2029653601, 2029654802}

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()

def top(source, name):
    nodes = [
        n for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in n.targets
        )
    ]

    check(
        len(nodes) == 1,
        "TOP_BINDING_INVALID:" + name
    )

    return nodes[0]

def replace_top(source, name, expression):
    node = top(source, name)
    lines = source.splitlines(keepends=True)

    lines[node.lineno-1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    output = "".join(lines)
    ast.parse(output)
    return output

try:
    print("=" * 76)
    print("T101S — STAGE NATIVE TRAJECTORY COMPARISON")
    print("=" * 76)

    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file()
        and sha(SOURCE) == SOURCE_SHA,
        "T91B_EVALUATOR_IDENTITY_INVALID"
    )

    original = SOURCE.read_text()

    check(
        ast.literal_eval(
            top(original, "TASKS").value
        ) == ORIGINAL_TASKS,
        "FOUR_TASKS_CHANGED"
    )

    check(
        ast.literal_eval(
            top(original, "_T91_SEEDS").value
        ) == {2029654801, 2029654802},
        "TRACE_SEED_SOURCE_CHANGED"
    )

    check(
        'context.get("name") != "T90S"'
        in original,
        "TRACE_MODEL_FILTER_CHANGED"
    )

    check(
        'f"T90S_seed{seed}.csv"'
        in original,
        "TRACE_FILE_NAMING_CHANGED"
    )

    check(
        "_t91_step(N.step_checked, env, action, locals())"
        in original,
        "ORIGINAL_NATIVE_STEP_OBSERVER_CHANGED"
    )

    check(
        'result = native_step(env, action)'
        in original,
        "NATIVE_ACTION_PASSTHROUGH_CHANGED"
    )

    model_node = top(original, "MODELS")

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
        entries[name] = ast.get_source_segment(
            original, value
        )

    check(
        set(entries) == {"T53", "T90S"},
        "MODEL_REFERENCE_TABLE_INVALID"
    )

    check(
        entries["T53"] is not None,
        "T53_REFERENCE_MISSING"
    )

    for label, model in MODELS.items():

        path = model["path"]

        check(
            path.is_file()
            and sha(path) == model["sha"],
            "CHECKPOINT_SHA_INVALID:" + label
        )

        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            int(saved["frames"]) == model["frames"],
            "CHECKPOINT_FRAMES_INVALID:" + label
        )

        check(
            saved["manifest"]["experiment"]
            == model["experiment"],
            "CHECKPOINT_EXPERIMENT_INVALID:" + label
        )

        check(
            saved["manifest"]["source_core"]
            == CORE_SHA,
            "CHECKPOINT_CORE_INVALID:" + label
        )

        del saved

        case = ROOT / label
        case.mkdir(parents=True, exist_ok=False)

        models_expression = (
            "{\n"
            " 'T53': " + entries["T53"] + ",\n"
            f" {label!r}: {{\n"
            f"  'path': Path({str(path)!r}),\n"
            f"  'sha': {model['sha']!r},\n"
            f"  'frames': {model['frames']},\n"
            f"  'core_sha': {CORE_SHA!r},\n"
            " },\n"
            "}"
        )

        generated = replace_top(
            original,
            "MODELS",
            models_expression
        )

        generated = replace_top(
            generated,
            "JOB",
            f"Path({str(case)!r})"
        )

        generated = replace_top(
            generated,
            "_T91_SEEDS",
            "{2029653601, 2029654802}"
        )

        # 同步替换评估器中的模型标签、
        # 轨迹记录条件及CSV文件名。
        generated = generated.replace(
            "T90S", label
        )

        ast.parse(generated)

        check(
            ast.literal_eval(
                top(generated, "TASKS").value
            ) == ORIGINAL_TASKS,
            "GENERATED_TASKS_CHANGED"
        )

        check(
            ast.literal_eval(
                top(generated, "_T91_SEEDS").value
            ) == TRACE_SEEDS,
            "GENERATED_TRACE_SEEDS_INVALID"
        )

        check(
            f'context.get("name") != "{label}"'
            in generated,
            "GENERATED_MODEL_FILTER_INVALID"
        )

        check(
            f'f"{label}_seed{{seed}}.csv"'
            in generated,
            "GENERATED_TRACE_FILENAME_INVALID"
        )

        check(
            "result = native_step(env, action)"
            in generated,
            "NATIVE_ACTION_CHANGED"
        )

        script = case / "evaluate.py"

        check(
            not script.exists(),
            "EVALUATOR_ALREADY_EXISTS"
        )

        script.write_text(generated)

        identity = {
            "model": label,
            "checkpoint": str(path),
            "checkpoint_sha256": model["sha"],
            "frames": model["frames"],
            "evaluator_sha256": sha(script),
            "source_sha256": SOURCE_SHA,
            "trace_seeds": sorted(TRACE_SEEDS),
            "action_override": False,
            "training": False,
        }

        (case / "identity.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )

        print(
            "T101S_CASE_READY=PASS",
            "MODEL=", label,
            "FRAMES=", model["frames"],
            "SHA=", model["sha"]
        )

    print("T101S_ORIGINAL_FOUR_TASKS=PASS")
    print("T101S_NATIVE_ACTION_UNCHANGED=PASS")
    print("T101S_TRACE_FILTER=PASS")
    print("T101S_PREFLIGHT=PASS")

except Exception as error:
    print(
        "T101S_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("TRAINING_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("HISTORICAL_MODEL_MODIFIED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
