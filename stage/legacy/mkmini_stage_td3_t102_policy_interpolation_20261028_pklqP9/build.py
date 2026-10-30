from pathlib import Path
import os
import ast
import json
import hashlib
import copy
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T102_JOB"])

BASE = H / (
    "mkmini_stage_td3_t87_direction_learning_"
    "20261028_Mnr36j/checkpoints/episode_0025.pt"
)

UPDATED = H / (
    "mkmini_stage_td3_t96m_balanced_sampling_"
    "20261028_zgWvJE/checkpoints/episode_0026.pt"
)

EVALUATOR = H / (
    "mkmini_stage_td3_t97_native4_"
    "20261028_S9JODK/evaluate.py"
)

EXPECTED = {
    "base":
        "0378e4af4296683d5d612a2d71607149e8fb3b0ff5235e35643766059ffaf94e",
    "updated":
        "ce5400573123025140337656b33637104942d8befd2d1870fc397fcce2e7e6e0",
    "evaluator":
        "053dd18d315c06fa6dac72c0c918524915d187f00bd489b85cbdfbc70c86cae2",
}

CORE_SHA = (
    "59b3b27bce8c3eeefc4c1b12989139a8fbe00474cc2784df0d8f5a5ec4353a64"
)

ALPHAS = [
    ("T102_B15", 0.15),
    ("T102_B35", 0.35),
    ("T102_B55", 0.55),
]

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
    (2029654801, -12.5, 3.0),
    (2029654802, 18.8, 3.0),
]

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
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
    check(len(nodes) == 1, "BINDING_INVALID:" + name)
    return nodes[0]

def replace_top(source, name, expression):
    n = top(source, name)
    lines = source.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]
    result = "".join(lines)
    ast.parse(result)
    return result

def blend_state(a, b, alpha):
    check(
        set(a.keys()) == set(b.keys()),
        "ACTOR_KEYS_DIFFER"
    )
    mixed = {}

    for key in a:
        x, y = a[key], b[key]
        check(
            x.shape == y.shape and x.dtype == y.dtype,
            "ACTOR_TENSOR_MISMATCH:" + key
        )
        if torch.is_floating_point(x):
            z = x.detach().cpu().clone()
            z = z * (1.0 - alpha) + y.detach().cpu() * alpha
            check(
                torch.isfinite(z).all().item(),
                "BLENDED_TENSOR_NONFINITE:" + key
            )
            mixed[key] = z
        else:
            check(
                torch.equal(x, y),
                "ACTOR_BUFFER_MISMATCH:" + key
            )
            mixed[key] = x.detach().cpu().clone()
    return mixed

try:
    print("=" * 76)
    print("T102 — STAGE TD3 SINGLE-ACTOR POLICY INTERPOLATION")
    print("=" * 76)

    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("HISTORICAL_MODEL_WRITE=FALSE")

    for name, path in [
        ("base", BASE),
        ("updated", UPDATED),
        ("evaluator", EVALUATOR),
    ]:
        check(path.is_file(), "SOURCE_MISSING:" + name)
        check(
            sha(path) == EXPECTED[name],
            "SOURCE_HASH_MISMATCH:" + name
        )
        print("T102_SOURCE_VERIFIED=", name)

    base = torch.load(
        BASE, map_location="cpu", weights_only=False
    )
    updated = torch.load(
        UPDATED, map_location="cpu", weights_only=False
    )

    check(
        int(base["frames"]) == 171387
        and int(updated["frames"]) == 178926,
        "SOURCE_FRAMES_INVALID"
    )

    check(
        base["manifest"]["source_core"] == CORE_SHA
        and updated["manifest"]["source_core"] == CORE_SHA,
        "SOURCE_CORE_CHANGED"
    )

    check(
        base["manifest"]["experiment"] == "STAGE_TD3_T87"
        and updated["manifest"]["experiment"] == "STAGE_TD3_T96M",
        "SOURCE_EXPERIMENT_INVALID"
    )

    check(
        set(base["actor"]) == set(updated["actor"]),
        "ACTOR_ARCHITECTURE_CHANGED"
    )

    source = EVALUATOR.read_text()

    check(
        ast.literal_eval(top(source, "TASKS").value) == TASKS,
        "ORIGINAL_FOUR_TASKS_CHANGED"
    )

    model_node = top(source, "MODELS")
    check(
        isinstance(model_node.value, ast.Dict),
        "EVALUATOR_MODEL_TABLE_INVALID"
    )

    entries = {}
    for key, value in zip(
        model_node.value.keys,
        model_node.value.values
    ):
        name = ast.literal_eval(key)
        entries[name] = ast.get_source_segment(source, value)

    check(
        set(entries) == {"T53", "T96M"},
        "EVALUATOR_MODEL_LABELS_CHANGED"
    )

    for label, alpha in ALPHAS:
        case = JOB / label
        case.mkdir(parents=True, exist_ok=False)

        checkpoint = case / "candidate.pt"

        child = copy.deepcopy(base)

        # 只对Actor网络进行参数插值。
        child["actor"] = blend_state(
            base["actor"], updated["actor"], alpha
        )

        # 同步记录对应的目标Actor；
        # 实际推理始终只使用actor。
        child["actor_target"] = blend_state(
            base["actor_target"],
            updated["actor_target"],
            alpha
        )

        # 此模型没有经过额外TD3训练。
        child["manifest"] = dict(base["manifest"])
        child["manifest"]["experiment"] = (
            "STAGE_TD3_T102_INTERPOLATION"
        )
        child["manifest"]["policy_interpolation"] = {
            "base": str(BASE),
            "base_sha256": EXPECTED["base"],
            "updated": str(UPDATED),
            "updated_sha256": EXPECTED["updated"],
            "alpha": alpha,
            "additional_training": False,
        }

        child["automatic_promotion"] = False

        torch.save(child, checkpoint)
        candidate_sha = sha(checkpoint)

        # 重新加载验证生成的候选。
        check_copy = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False
        )

        check(
            int(check_copy["frames"]) == 171387
            and set(check_copy["actor"]) == set(base["actor"]),
            "CANDIDATE_RELOAD_INVALID"
        )

        del check_copy

        model_expression = (
            "{\n"
            " 'T53': " + entries["T53"] + ",\n"
            f" {label!r}: {{\n"
            f"  'path': Path({str(checkpoint)!r}),\n"
            f"  'sha': {candidate_sha!r},\n"
            "  'frames': 171387,\n"
            f"  'core_sha': {CORE_SHA!r},\n"
            " },\n"
            "}"
        )

        generated = source.replace(
            "T96M", label
        )

        generated = replace_top(
            generated, "MODELS", model_expression
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
        script.write_text(generated)

        manifest = {
            "candidate": label,
            "alpha": alpha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": candidate_sha,
            "evaluator_sha256": sha(script),
            "base_sha256": EXPECTED["base"],
            "updated_sha256": EXPECTED["updated"],
            "additional_training": False,
            "automatic_promotion": False,
        }

        (case / "identity.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        print(
            "T102_CANDIDATE_READY=PASS",
            "LABEL=", label,
            "ALPHA=", alpha,
            "SHA256=", candidate_sha
        )

        del child

    print("T102_FOUR_TASKS_UNCHANGED=PASS")
    print("T102_SINGLE_ACTOR=PASS")
    print("T102_PREFLIGHT=PASS")

except Exception as error:
    print("T102_PREFLIGHT_BLOCKED=", repr(error))
    traceback.print_exc()

finally:
    print("TRAINING_LAUNCHED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
