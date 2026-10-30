from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import traceback

import numpy as np
import torch

H = Path.home()

JOB = (
    H / "mkmini_stage_td3_t44_retention_gate_20261027_v1"
)

T16 = (
    H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
)

T40 = (
    H / "mkmini_stage_td3_t40_forgetting_regression_20261027_v1"
)

T43 = (
    H / "mkmini_stage_td3_t43_mixed_replay_pilot_20261027_v1"
)

T38 = (
    H / "mkmini_stage_td3_t38_policy_retention_20261027_v1"
)

SOURCE = T40 / "evaluate.py"

CORE = T16 / "td3_core_goal_contrast.py"

BANK = T38 / "t16_reference_bank.npz"

CORE_SHA = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

BANK_SHA = (
    "37d100a34140bbb3ab986297c1f5db786f8b2f1a5862b91efea8bc761efa1732"
)

MODELS = {
    "T16": {
        "path": T16 / "checkpoints/episode_0008.pt",
        "sha": (
            "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
        ),
        "frames": 13491,
        "core_sha": CORE_SHA,
    },
    "T43R": {
        "path": T43 / "checkpoints/episode_0011.pt",
        "sha": (
            "45102b3257bf21e1a5809555420d0ef167a1710f545d55f3508b7aba14acb2c6"
        ),
        "frames": 25491,
        "core_sha": CORE_SHA,
    },
}

TASKS = [
    (2029653601, -17.3, 3.0),
    (2029653602, 13.6, 3.0),
]


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def patch_assignments(source, changes):
    tree = ast.parse(source)
    nodes = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in changes
            ):
                check(
                    target.id not in nodes,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )

                nodes[target.id] = node

    check(
        set(nodes) == set(changes),
        "MISSING_ASSIGNMENTS:"
        + repr(set(changes) - set(nodes))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        nodes.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)

    ast.parse(result)

    return result


def offline_actor_audit(checkpoints):
    print()
    print("========== T44相同观测Actor检查 ==========")

    check(
        BANK.is_file() and sha(BANK) == BANK_SHA,
        "T38_REFERENCE_CHANGED"
    )

    check(
        CORE.is_file() and sha(CORE) == CORE_SHA,
        "ACTOR_SOURCE_CHANGED"
    )

    spec = importlib.util.spec_from_file_location(
        "t44_stage_core",
        CORE
    )

    check(
        spec is not None
        and spec.loader is not None,
        "CORE_IMPORT_INVALID"
    )

    core = importlib.util.module_from_spec(spec)
    sys.modules["t44_stage_core"] = core
    spec.loader.exec_module(core)

    with np.load(BANK) as data:
        observations = np.asarray(
            data["observations"],
            dtype=np.float32
        )

        teacher = np.asarray(
            data["teacher_raw_actions"],
            dtype=np.float32
        )

    check(
        observations.shape == (8192, 112)
        and teacher.shape == (8192, 2),
        "REFERENCE_SHAPE_INVALID"
    )

    torch.set_num_threads(2)

    results = {}

    for name, checkpoint in checkpoints.items():
        actor = core.Actor(112, 2)

        actor.load_state_dict(
            checkpoint["actor"],
            strict=True
        )

        actor.eval()

        outputs = []

        with torch.no_grad():
            for i in range(0, len(observations), 1024):
                x = torch.from_numpy(
                    observations[i:i + 1024]
                )

                outputs.append(
                    actor(x).cpu().numpy()
                )

        results[name] = np.concatenate(outputs)

    t16 = results["T16"]
    t43 = results["T43R"]

    check(
        np.allclose(
            t16, teacher,
            atol=1e-6,
            rtol=0
        ),
        "T16_REFERENCE_NOT_REPRODUCED"
    )

    masks = {
        "ALL": np.ones(
            len(teacher), dtype=bool
        ),
        "T16_LEFT_TURN": teacher[:, 1] < -0.20,
        "T16_RIGHT_TURN": teacher[:, 1] > 0.20,
        "NEAR_GOAL_REFERENCE": (
            observations[:, 48] * 10.0 < 0.6
        ),
    }

    for label, mask in masks.items():
        if not mask.any():
            print(
                "T44_OFFLINE",
                label,
                "COUNT=0"
            )
            continue

        old = t16[mask]
        new = t43[mask]

        same_sign = np.mean(
            np.sign(old[:, 1])
            == np.sign(new[:, 1])
        )

        print(
            "T44_OFFLINE",
            label,
            "COUNT=", int(mask.sum()),
            "T16_THROTTLE=",
            round(
                float(((old[:, 0] + 1) / 2).mean()),
                5
            ),
            "T43_THROTTLE=",
            round(
                float(((new[:, 0] + 1) / 2).mean()),
                5
            ),
            "T16_STEER=",
            round(float(old[:, 1].mean()), 5),
            "T43_STEER=",
            round(float(new[:, 1].mean()), 5),
            "STEER_SIGN_AGREEMENT=",
            round(float(same_sign), 5),
            "MEAN_ABS_STEER_DRIFT=",
            round(
                float(
                    np.abs(
                        old[:, 1] - new[:, 1]
                    ).mean()
                ),
                5
            )
        )

    print("T44_OFFLINE_AUDIT=PASS")
    print(
        "T44_OFFLINE_PASS_DOES_NOT_PROVE_NAVIGATION=TRUE"
    )


def main():
    print("T44_STAGE_ONLY=TRUE")
    print("T44_TRAINING=FALSE")
    print("T44_ACTION_OVERRIDE=FALSE")
    print("T44_PLANNER=FALSE")

    provenance_path = T40 / "preflight.json"

    check(
        SOURCE.is_file()
        and provenance_path.is_file(),
        "T40_VERIFIED_EVALUATOR_MISSING"
    )

    provenance = json.loads(
        provenance_path.read_text()
    )

    check(
        sha(SOURCE) == provenance["evaluator_sha256"],
        "T40_EVALUATOR_CHANGED"
    )

    checkpoints = {}

    for name, cfg in MODELS.items():
        path = cfg["path"]

        check(
            path.is_file()
            and sha(path) == cfg["sha"],
            "CHECKPOINT_CHANGED:" + name
        )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3"
            and int(state["frames"]) == cfg["frames"]
            and state["manifest"]["source_core"]
                == cfg["core_sha"],
            "CHECKPOINT_PROVENANCE_INVALID:" + name
        )

        checkpoints[name] = state

        print("T44_MODEL_VERIFIED=", name)

    offline_actor_audit(checkpoints)

    model_lines = ["MODELS = {"]

    for name, cfg in MODELS.items():
        model_lines.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(cfg['path'])!r}),",
            f"        'sha': {cfg['sha']!r},",
            f"        'frames': {cfg['frames']},",
            f"        'core_sha': {cfg['core_sha']!r},",
            "    },",
        ])

    model_lines.append("}")

    generated = patch_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_lines),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 5400",
        }
    )

    generated = generated.replace("T40", "T44")
    ast.parse(generated)

    target = JOB / "evaluate.py"

    expected = hashlib.sha256(
        generated.encode()
    ).hexdigest()

    if target.exists():
        check(
            sha(target) == expected,
            "EXISTING_T44_EVALUATOR_CHANGED"
        )
    else:
        target.write_text(generated)

    info = {
        "experiment": "STAGE_TD3_T44",
        "evaluator_sha256": expected,
        "source_evaluator_sha256": sha(SOURCE),
        "models": {
            name: cfg["sha"]
            for name, cfg in MODELS.items()
        },
        "tasks": TASKS,
        "training": False,
        "automatic_promotion": False,
    }

    manifest = JOB / "provenance.json"

    if manifest.exists():
        check(
            json.loads(manifest.read_text()) == info,
            "T44_PROVENANCE_CHANGED"
        )
    else:
        manifest.write_text(
            json.dumps(
                info,
                indent=2
            ) + "\n"
        )

    print("T44_OLD_SUCCESS_CASES=2")
    print("T44_REAL_MUJOCO_RUNS=4")
    print("T44_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T44_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
