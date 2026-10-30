from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path(os.environ["T87_JOB"])

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
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def replace_top(source, name, expression):
    tree = ast.parse(source)

    found = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == name
            for t in n.targets
        )
    ]

    check(
        len(found) == 1,
        "TOP_BINDING_INVALID:" + name
    )

    n = found[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

def replace_once(source, old, new, label):
    check(
        source.count(old) == 1,
        "PATCH_ANCHOR_INVALID:" + label
    )

    result = source.replace(old, new, 1)
    ast.parse(result)
    return result

def main():
    print("=" * 78)
    print("T87 — STAGE TD3 REAL-STATE DIRECTION LEARNING")
    print("=" * 78)

    print("TRAINING_STARTED=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    train_path = BASE / "train.py"
    core_path = BASE / "td3_core_goal_contrast.py"

    check(
        train_path.is_file()
        and sha(train_path) == TRAIN_SHA,
        "BASE_TRAIN_CHANGED"
    )

    check(
        core_path.is_file()
        and sha(core_path) == CORE_SHA,
        "BASE_CORE_CHANGED"
    )

    initial_path = (
        T77 / "checkpoints/episode_0022.pt"
    )

    check(
        initial_path.is_file(),
        "INITIAL_CHECKPOINT_MISSING"
    )

    initial = torch.load(
        initial_path,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(initial["frames"]) == 134491
        and initial["manifest"]["source_core"] == CORE_SHA,
        "INITIAL_STATE_INVALID"
    )

    parent_status = json.loads(
        (PARENT / "status.json").read_text()
    )

    parent_ck = (
        PARENT / "checkpoints"
        / parent_status["checkpoint"]
    )

    check(
        parent_ck.is_file()
        and sha(parent_ck)
            == parent_status["checkpoint_sha256"],
        "FROZEN_PARENT_INVALID"
    )

    parent = torch.load(
        parent_ck,
        map_location="cpu",
        weights_only=False
    )

    for key in (
        "actor",
        "critic",
        "actor_target",
        "critic_target"
    ):
        a = initial[key]
        b = parent[key]

        check(
            set(a) == set(b)
            and all(
                torch.equal(a[k], b[k])
                for k in a
            ),
            "PARENT_NETWORK_MISMATCH:" + key
        )

    print("T87_INITIAL_NETWORK_IDENTICAL=PASS")

    # 对恢复检查点进行观测语义核对。
    # 避免把未知的112维特征错当成目标角度。
    replay = initial["replay"]

    check(
        len(replay) >= 500,
        "REPLAY_SAMPLE_INSUFFICIENT"
    )

    observations = np.stack([
        np.asarray(row[0], dtype=np.float32)
        for row in replay[-500:]
    ])

    check(
        observations.shape == (500, 112),
        "OBSERVATION_LAYOUT_INVALID"
    )

    lidar = observations[:, :48]
    goal_sin = observations[:, 49]
    goal_cos = observations[:, 50]

    norm = goal_sin**2 + goal_cos**2

    norm_valid = float(
        np.mean(np.abs(norm - 1.0) < 0.04)
    )

    lidar_valid = bool(
        np.isfinite(lidar).all()
        and lidar.min() >= -0.001
        and lidar.max() <= 1.01
    )

    check(
        norm_valid >= 0.95,
        "GOAL_SIN_COS_LAYOUT_NOT_VERIFIED"
    )

    check(
        lidar_valid,
        "LIDAR_NORMALIZATION_NOT_VERIFIED"
    )

    print(
        "T87_GOAL_SIN_COS_VALID_RATIO=",
        round(norm_valid, 4)
    )

    print(
        "T87_LIDAR_RANGE=",
        round(float(lidar.min()), 5),
        round(float(lidar.max()), 5)
    )

    print("T87_OBSERVATION_SEMANTICS=PASS")

    # 仅在目标位于前半平面、
    # 偏角达到一定幅度且障碍净空
    # 满足筛选条件时施加辅助损失。
    #
    # 本项只影响训练Actor损失，
    # 不替代推理动作，不更改MuJoCo控制器。

    core = core_path.read_text()

    old = "            actor_loss.backward()"

    new = '''
            # T87: goal-direction consistency on real replay states.
            # Training-only auxiliary loss, not an action override.
            _t87_sin = state[:, 49].detach()
            _t87_cos = state[:, 50].detach()

            _t87_clearance = (
                state[:, :48].detach()
                .amin(dim=1)
            )

            _t87_mask = (
                (_t87_sin.abs() >= 0.12)
                & (_t87_sin.abs() <= 0.80)
                & (_t87_cos >= 0.55)
                & (_t87_clearance >= 0.14)
            )

            _t87_actions = self.actor(state)
            _t87_steer = _t87_actions[:, 1]

            _t87_margin = (
                0.10
                - torch.sign(_t87_sin) * _t87_steer
            )

            _t87_penalty = (
                torch.relu(_t87_margin).square()
            )

            _t87_count = _t87_mask.sum()

            _t87_loss = (
                (_t87_penalty * _t87_mask.float()).sum()
                / _t87_count.clamp(min=1)
            )

            actor_loss = (
                actor_loss + 0.60 * _t87_loss
            )

            if self.total_updates % 2000 == 0:
                print(
                    "T87_DIRECTION_LOSS_ACTIVE=PASS",
                    "UPDATES=", self.total_updates,
                    "MASKED_SAMPLES=",
                    int(_t87_count.detach().cpu()),
                    "LOSS=",
                    round(
                        float(_t87_loss.detach().cpu()),
                        7
                    ),
                    flush=True
                )

            actor_loss.backward()
'''

    core = replace_once(
        core,
        old,
        new,
        "ACTOR_LOSS"
    )

    new_core_sha = hashlib.sha256(
        core.encode()
    ).hexdigest()

    train = train_path.read_text()

    train = replace_top(
        train,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    train = replace_once(
        train,
        "STAGE_TD3_T71B",
        "STAGE_TD3_T87",
        "EXPERIMENT_ID"
    )

    train = train.replace(
        CORE_SHA,
        new_core_sha
    )

    ast.parse(train)
    ast.parse(core)

    check(
        not (JOB / "train.py").exists(),
        "T87_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(train)

    (JOB / "td3_core_goal_contrast.py").write_text(
        core
    )

    (JOB / "goal_pairs.npz").write_bytes(
        (BASE / "goal_pairs.npz").read_bytes()
    )

    (JOB / "checkpoints").mkdir(
        exist_ok=True
    )

    child = copy.deepcopy(initial)

    child["manifest"]["experiment"] = (
        "STAGE_TD3_T87"
    )

    child["manifest"]["source_core"] = (
        new_core_sha
    )

    child["manifest"]["target_frames"] = 190491

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
        "automatic_promotion": False
    })

    (JOB / "status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )

    report = {
        "experiment": "STAGE_TD3_T87",
        "base": "T71B",
        "initial_frames": 134491,
        "target_frames": 190491,
        "single_functional_change":
            "Replay-state actor goal-direction auxiliary loss",
        "direction_loss_weight": 0.60,
        "goal_sin_index": 49,
        "goal_cos_index": 50,
        "lidar_indices": [0, 47],
        "lidar_min_normalized": 0.14,
        "actor_lr_unchanged": True,
        "td3_rewards_unchanged": True,
        "replay_unchanged": True,
        "environment_unchanged": True,
        "core_sha256": new_core_sha,
        "train_sha256": sha(JOB / "train.py"),
        "checkpoint_sha256": sha(checkpoint),
        "automatic_promotion": False
    }

    (JOB / "experiment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("T87_NEW_CORE_SHA256=", new_core_sha)
    print("T87_TRAIN_SOURCE=PASS")
    print("T87_INITIAL_CHECKPOINT=PASS")
    print("T87_PREFLIGHT=PASS")

try:
    main()

except Exception as error:
    print(
        "T87_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
