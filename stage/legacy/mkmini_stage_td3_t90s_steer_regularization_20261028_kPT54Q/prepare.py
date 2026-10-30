from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import shutil
import traceback

import torch

H = Path.home()
JOB = Path(os.environ["T90S_JOB"])

BASE = H / (
    "mkmini_stage_td3_t71_actor_ablation_"
    "20261028_dE1arY/T71B"
)

T77 = H / (
    "mkmini_stage_td3_t77_goal_recovery_"
    "20261028_AMyCJl"
)

PARENT = H / (
    "mkmini_stage_td3_t67_3m_nightrelay_"
    "20261027_kEPAW0/T67C022"
)

TRAIN_SHA = (
    "19a03f58bba9b7122b1eb17c1f9806f9e00b75"
    "a0acc058a0a7af3eaaac410763"
)

CORE_SHA = (
    "c7e06605579a8dec7d0bd95033ae437997583"
    "3968fe69968e7294396e9b21cdc"
)

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()

def replace_top(source, name, expression):
    tree = ast.parse(source)
    nodes = [
        n for n in tree.body
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

    n = nodes[0]
    lines = source.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    updated = "".join(lines)
    ast.parse(updated)
    return updated

try:
    print("=" * 76)
    print("T90S STAGE TD3 STEERING REGULARIZATION ABLATION")
    print("=" * 76)

    train_path = BASE / "train.py"
    core_path = BASE / "td3_core_goal_contrast.py"

    check(
        sha(train_path) == TRAIN_SHA,
        "BASE_TRAIN_SOURCE_CHANGED"
    )

    check(
        sha(core_path) == CORE_SHA,
        "BASE_CORE_SOURCE_CHANGED"
    )

    initial_path = (
        T77 / "checkpoints/episode_0022.pt"
    )

    initial = torch.load(
        initial_path,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(initial["frames"]) == 134491
        and initial["manifest"]["source_core"]
            == CORE_SHA,
        "INITIAL_CHECKPOINT_INVALID"
    )

    parent_status = json.loads(
        (PARENT / "status.json").read_text()
    )

    parent_path = (
        PARENT / "checkpoints"
        / parent_status["checkpoint"]
    )

    check(
        sha(parent_path)
        == parent_status["checkpoint_sha256"],
        "FROZEN_PARENT_HASH_MISMATCH"
    )

    parent = torch.load(
        parent_path,
        map_location="cpu",
        weights_only=False
    )

    for key in (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
    ):
        a = initial[key]
        b = parent[key]

        check(
            set(a) == set(b)
            and all(
                torch.equal(a[k], b[k])
                for k in a
            ),
            "INITIAL_NETWORK_MISMATCH:" + key
        )

    core = core_path.read_text()

    old = (
        "td3_t9_actions[:, 1].square().mean()"
    )

    new = (
        "0.10 * td3_t9_actions[:, 1].square().mean()"
    )

    check(
        core.count(old) == 1,
        "STEERING_REGULARIZER_NOT_UNIQUE"
    )

    core = core.replace(old, new, 1)
    ast.parse(core)

    new_core_sha = hashlib.sha256(
        core.encode("utf-8")
    ).hexdigest()

    train = train_path.read_text()

    train = replace_top(
        train,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    check(
        train.count("STAGE_TD3_T71B") == 1,
        "EXPERIMENT_ID_NOT_UNIQUE"
    )

    train = train.replace(
        "STAGE_TD3_T71B",
        "STAGE_TD3_T90S",
        1
    )

    # 训练器记录的Core标识与新源码同步。
    train = train.replace(
        CORE_SHA,
        new_core_sha
    )

    ast.parse(train)

    check(
        train.count("T66_PROGRESS_REWARD_ACTIVE") == 1,
        "DISTANCE_REWARD_UNEXPECTED"
    )

    check(
        "0.5 * td3_t9_actions[:, 0].square().mean()"
        in core,
        "THROTTLE_PENALTY_CHANGED"
    )

    check(
        "(0.8 * _t71_factor) * _t53_ref_loss"
        in core,
        "SUCCESS_REFERENCE_CHANGED"
    )

    check(
        not (JOB / "train.py").exists(),
        "T90S_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(train)

    (
        JOB / "td3_core_goal_contrast.py"
    ).write_text(core)

    shutil.copy2(
        BASE / "goal_pairs.npz",
        JOB / "goal_pairs.npz"
    )

    (JOB / "checkpoints").mkdir(
        exist_ok=True
    )

    child = copy.deepcopy(initial)

    child["manifest"]["experiment"] = (
        "STAGE_TD3_T90S"
    )

    child["manifest"]["source_core"] = (
        new_core_sha
    )

    child["manifest"]["target_frames"] = (
        190491
    )

    checkpoint = (
        JOB / "checkpoints/episode_0022.pt"
    )

    torch.save(child, checkpoint)

    status = copy.deepcopy(parent_status)

    status.update({
        "state": "RUNNING",
        "frames": 134491,
        "target_frames": 190491,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha(checkpoint),
        "automatic_promotion": False,
    })

    (JOB / "status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )

    report = {
        "experiment": "STAGE_TD3_T90S",
        "baseline": "T71B",
        "initial_frames": 134491,
        "target_frames": 190491,
        "single_functional_change":
            "Steering L2 coefficient 0.03 to 0.003",
        "steering_l2_before": 0.03,
        "steering_l2_after": 0.003,
        "throttle_l2_unchanged": 0.015,
        "reward_changed": False,
        "replay_changed": False,
        "actor_lr_changed": False,
        "teacher_bc_changed": False,
        "t53_anchor_changed": False,
        "environment_changed": False,
        "automatic_promotion": False,
        "core_sha256": new_core_sha,
        "train_sha256": sha(
            JOB / "train.py"
        ),
        "initial_checkpoint_sha256": sha(
            checkpoint
        ),
    }

    (JOB / "experiment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("T90S_INITIAL_NETWORK=PASS")
    print("T90S_SINGLE_CHANGE=PASS")
    print("T90S_ORIGINAL_THROTTLE_PENALTY=PASS")
    print("T90S_SUCCESS_POLICY_PROTECTION=PASS")
    print("T90S_CORE_SHA256=", new_core_sha)
    print("T90S_PREFLIGHT=PASS")

except Exception as e:
    print("T90S_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()

finally:
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
