from pathlib import Path
import ast
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T101_JOB"])

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
    "T87A": (
        H / (
            "mkmini_stage_td3_t87_direction_learning_"
            "20261028_Mnr36j/checkpoints/episode_0025.pt"
        ),
        "0378e4af4296683d5d612a2d71607149e8fb3b"
        "0ff5235e35643766059ffaf94e",
        171387,
        "STAGE_TD3_T87",
    ),
    "T96A": (
        H / (
            "mkmini_stage_td3_t96m_balanced_sampling_"
            "20261028_zgWvJE/checkpoints/episode_0026.pt"
        ),
        "ce5400573123025140337656b33637104942d8befd2"
        "d1870fc397fcce2e7e6e0",
        178926,
        "STAGE_TD3_T96M",
    ),
}

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
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def top(source, name):
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in node.targets
        )
    ]
    check(
        len(matches) == 1,
        "TOP_BINDING_INVALID:" + name
    )
    return matches[0]

def replace_top(source, name, expression):
    node = top(source, name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [
        f"{name} = {expression}\n"
    ]
    result = "".join(lines)
    ast.parse(result)
    return result

class TraceSeedPatch(ast.NodeTransformer):
    """
    只修改T91诊断函数中的记录目标种子。
    TASKS及原生MuJoCo执行逻辑不变。
    """
    def __init__(self):
        self.inside_trace = False
        self.changed = 0

    def visit_FunctionDef(self, node):
        previous = self.inside_trace
        self.inside_trace = (
            previous or node.name.startswith("_t91")
        )
        node = self.generic_visit(node)
        self.inside_trace = previous
        return node

    def visit_Constant(self, node):
        if (
            self.inside_trace
            and type(node.value) is int
            and node.value == 2029654801
        ):
            self.changed += 1
            return ast.copy_location(
                ast.Constant(value=2029653601),
                node
            )
        return node

try:
    print("=" * 76)
    print("T101 STAGE TD3 NATIVE TRAJECTORY COMPARISON")
    print("=" * 76)

    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    check(
        SOURCE.is_file()
        and sha(SOURCE) == SOURCE_SHA,
        "VERIFIED_TRACE_SOURCE_CHANGED"
    )

    original = SOURCE.read_text()

    check(
        ast.literal_eval(
            top(original, "TASKS").value
        ) == TASKS,
        "ORIGINAL_TASKS_INVALID"
    )

    check(
        "T91B_JOB" in original
        and "_t91_step" in original,
        "TRACE_INSTRUMENTATION_MISSING"
    )

    model_node = top(original, "MODELS")

    check(
        isinstance(model_node.value, ast.Dict),
        "MODEL_TABLE_INVALID"
    )

    entries = {}

    for k, v in zip(
        model_node.value.keys,
        model_node.value.values
    ):
        name = ast.literal_eval(k)
        entries[name] = ast.get_source_segment(
            original, v
        )

    check(
        set(entries) == {"T53", "T90S"},
        "REFERENCE_PAIR_INVALID"
    )

    for label, (path, expected_sha, frames, exp) in MODELS.items():

        check(
            path.is_file() and sha(path) == expected_sha,
            "MODEL_IDENTITY_INVALID:" + label
        )

        saved = torch.load(
            path, map_location="cpu",
            weights_only=False
        )

        check(
            int(saved["frames"]) == frames
            and saved["manifest"]["experiment"] == exp
            and saved["manifest"]["source_core"] == CORE_SHA,
            "CHECKPOINT_CONTENT_INVALID:" + label
        )

        del saved

        case = JOB / label
        case.mkdir(parents=True, exist_ok=False)

        model_table = (
            "{\n"
            " 'T53': " + entries["T53"] + ",\n"
            f" {label!r}: {{\n"
            f"   'path': Path({str(path)!r}),\n"
            f"   'sha': {expected_sha!r},\n"
            f"   'frames': {frames},\n"
            f"   'core_sha': {CORE_SHA!r},\n"
            " },\n"
            "}"
        )

        generated = replace_top(
            original, "MODELS", model_table
        )

        generated = replace_top(
            generated, "JOB",
            f"Path({str(case)!r})"
        )

        generated = generated.replace("T90S", label)

        tree = ast.parse(generated)

        patch = TraceSeedPatch()
        tree = patch.visit(tree)
        ast.fix_missing_locations(tree)

        check(
            patch.changed >= 1,
            "TRACE_SEED_FILTER_NOT_FOUND"
        )

        generated = ast.unparse(tree) + "\n"

        check(
            ast.literal_eval(
                top(generated, "TASKS").value
            ) == TASKS,
            "NATIVE_TASKS_CHANGED"
        )

        check(
            "N.step_checked" in generated
            and "_t91_step(" in generated,
            "NATIVE_STEP_OBSERVER_CHANGED"
        )

        check(
            set(
                ast.literal_eval(
                    top(generated, "MODELS").value.keys[0]
                )
                for _ in [0]
            ) == {"T53"},
            "REFERENCE_LABEL_INVALID"
        )

        script = case / "evaluate.py"

        script.write_text(generated)

        (case / "identity.json").write_text(
            json.dumps({
                "label": label,
                "model_sha256": expected_sha,
                "frames": frames,
                "source_evaluator_sha256": SOURCE_SHA,
                "generated_sha256": sha(script),
                "trace_seed_patch_count": patch.changed,
                "training": False,
                "action_override": False,
            }, indent=2) + "\n"
        )

        print(
            "T101_CASE_READY=PASS",
            "MODEL=", label,
            "TRACE_PATCH_COUNT=", patch.changed,
            "EVALUATOR_SHA=", sha(script)
        )

    print("T101_PREFLIGHT=PASS")

except Exception as error:
    print("T101_PREFLIGHT_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("MODEL_PROMOTED=FALSE")
    print("HISTORICAL_MODEL_MODIFIED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
