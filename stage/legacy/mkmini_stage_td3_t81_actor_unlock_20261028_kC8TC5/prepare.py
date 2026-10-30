from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import shutil
import traceback

import torch

HOME = Path.home()
JOB = Path(os.environ["T81_JOB"])

BASE = HOME / (
    "mkmini_stage_td3_t71_actor_ablation_"
    "20261028_dE1arY/T71B"
)

T77 = HOME / (
    "mkmini_stage_td3_t77_goal_recovery_"
    "20261028_AMyCJl"
)

PARENT = HOME / (
    "mkmini_stage_td3_t67_3m_nightrelay_"
    "20261027_kEPAW0/T67C022"
)

OLD_TRAIN_SHA = (
    "19a03f58bba9b7122b1eb17c1f9806f9e00b75a0acc"
    "058a0a7af3eaaac410763"
)

OLD_CORE_SHA = (
    "c7e06605579a8dec7d0bd95033ae4379975833968fe"
    "69968e7294396e9b21cdc"
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)

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
        "TOP_ASSIGNMENT_INVALID:" + name
    )

    n = found[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno - 1:n.end_lineno] = [
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

    source = source.replace(old, new, 1)
    ast.parse(source)
    return source

def main():
    print("=" * 78)
    print("T81 — TD3 ACTOR UNLOCK TRAINING BRANCH")
    print("=" * 78)

    print("STAGE_ONLY=TRUE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("TRAINING_STARTED=FALSE")

    train_path = BASE / "train.py"
    core_path = BASE / "td3_core_goal_contrast.py"

    check(
        train_path.is_file()
        and sha(train_path) == OLD_TRAIN_SHA,
        "BASE_TRAIN_SHA_MISMATCH"
    )

    check(
        core_path.is_file()
        and sha(core_path) == OLD_CORE_SHA,
        "BASE_CORE_SHA_MISMATCH"
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
        int(initial["frames"]) == 134491,
        "INITIAL_FRAME_MISMATCH"
    )

    check(
        initial["manifest"]["experiment"]
            == "STAGE_TD3_T77",
        "INITIAL_EXPERIMENT_MISMATCH"
    )

    check(
        initial["manifest"]["source_core"]
            == OLD_CORE_SHA,
        "INITIAL_CORE_MISMATCH"
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
        "FROZEN_PARENT_SHA_MISMATCH"
    )

    parent = torch.load(
        parent_ck,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(parent["frames"]) == 134491,
        "PARENT_FRAME_MISMATCH"
    )

    for name in (
        "actor",
        "critic",
        "actor_target",
        "critic_target"
    ):
        a = initial[name]
        b = parent[name]

        check(
            set(a) == set(b)
            and all(
                torch.equal(a[k], b[k])
                for k in a
            ),
            "INITIAL_NETWORK_NOT_IDENTICAL:" + name
        )

    print("T81_IDENTICAL_INITIAL_NETWORK=PASS")

    # ============================================
    # A. 修改TD3的Actor损失
    # ============================================

    core = core_path.read_text()

    # T80审计证实存在这一行。
    old_lr = (
        "_t53_group['lr'] = "
        "min(float(_t53_group['lr']), 1e-5)"
    )

    new_lr = (
        "_t53_group['lr'] = 3e-5"
    )

    core = replace_once(
        core,
        old_lr,
        new_lr,
        "ACTOR_LR_CAP"
    )

    # 保留原有TD3优化器、Critic、目标网络更新。
    # 只在Actor反向传播前重构最终损失。
    #
    # 原T53锚定仍会进行数值计算，
    # 但不会进入新的Actor反向传播损失。
    # 保留少量已验证成功轨迹BC，
    # 避免直接丢弃已有成功经验。

    anchor = "            actor_loss.backward()"

    replacement = '''
            # T81: train-only actor constraint release.
            # No inference-time action override.
            _t81_actions = self.actor(state)

            _t81_policy = -self.critic.q1(
                torch.cat(
                    (state, _t81_actions),
                    dim=-1
                )
            ).mean()

            _t81_regularization = 0.03 * (
                _t81_actions[:, 1].square().mean()
                + 0.5
                * _t81_actions[:, 0].square().mean()
            )

            _t81_goal_pair = (
                0.05 * t12_goal_loss
            )

            _t81_success_bc = (
                0.10 * _bc_loss
            )

            actor_loss = (
                _t81_policy
                + _t81_regularization
                + _t81_goal_pair
                + _t81_success_bc
            )

            if self.total_updates % 2000 == 0:
                print(
                    "T81_ACTOR_UNLOCK_ACTIVE=PASS",
                    "UPDATES=", self.total_updates,
                    "ACTOR_LR_TARGET=", 3e-5,
                    "SUCCESS_BC_WEIGHT=", 0.10,
                    "T53_ANCHOR_GRAD_WEIGHT=", 0.0,
                    "POLICY_LOSS=",
                    round(
                        float(_t81_policy.detach()),
                        6
                    ),
                    flush=True
                )

            actor_loss.backward()
'''

    core = replace_once(
        core,
        anchor,
        replacement,
        "ACTOR_BACKWARD"
    )

    new_core_sha = hashlib.sha256(
        core.encode()
    ).hexdigest()

    print("T81_OLD_CORE_SHA256=", OLD_CORE_SHA)
    print("T81_NEW_CORE_SHA256=", new_core_sha)

    # ============================================
    # B. 复制原始T71B训练程序
    # ============================================

    train = train_path.read_text()

    train = replace_top(
        train,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    train = replace_once(
        train,
        "STAGE_TD3_T71B",
        "STAGE_TD3_T81",
        "EXPERIMENT_ID"
    )

    # 若训练源码固定引用原始核心哈希，
    # 同步更新为新核心源码哈希。
    train = train.replace(
        OLD_CORE_SHA,
        new_core_sha
    )

    ast.parse(train)
    ast.parse(core)

    # ============================================
    # C. 独立保存训练文件
    # ============================================

    check(
        not (JOB / "train.py").exists(),
        "T81_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(train)

    (JOB / "td3_core_goal_contrast.py").write_text(
        core
    )

    shutil.copy2(
        BASE / "goal_pairs.npz",
        JOB / "goal_pairs.npz"
    )

    (JOB / "checkpoints").mkdir(
        exist_ok=True
    )

    # ============================================
    # D. 恢复相同的134491帧起点
    # ============================================

    child = copy.deepcopy(initial)

    child["manifest"]["experiment"] = (
        "STAGE_TD3_T81"
    )

    child["manifest"]["source_core"] = (
        new_core_sha
    )

    child["manifest"]["target_frames"] = 190491

    new_checkpoint = (
        JOB / "checkpoints/episode_0022.pt"
    )

    torch.save(child, new_checkpoint)

    new_status = copy.deepcopy(parent_status)

    new_status.update({
        "state": "RUNNING",
        "frames": 134491,
        "target_frames": 190491,
        "checkpoint": new_checkpoint.name,
        "checkpoint_sha256": sha(new_checkpoint),
        "automatic_promotion": False
    })

    (JOB / "status.json").write_text(
        json.dumps(
            new_status,
            indent=2
        ) + "\n"
    )

    # ============================================
    # E. 审计证据
    # ============================================

    report = {
        "experiment": "STAGE_TD3_T81",
        "baseline": "T71B",
        "frozen_parent": "T67C022",
        "initial_frames": 134491,
        "target_frames": 190491,
        "changes": {
            "actor_lr": 3e-5,
            "success_bc_weight": 0.10,
            "t53_actor_anchor_gradient_weight": 0.0
        },
        "unchanged": [
            "TD3 network",
            "112D observations",
            "native lidar",
            "vehicle model",
            "environment",
            "training tasks",
            "reward",
            "replay capacity",
            "physics stepping"
        ],
        "parent_checkpoint_sha256":
            parent_status["checkpoint_sha256"],
        "new_initial_checkpoint_sha256":
            sha(new_checkpoint),
        "train_sha256":
            sha(JOB / "train.py"),
        "core_sha256": new_core_sha,
        "automatic_promotion": False
    }

    (JOB / "experiment.json").write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print("T81_INITIAL_CHECKPOINT=PASS")
    print("T81_TRAIN_SOURCE=PASS")
    print("T81_ACTOR_CORE=PASS")
    print("T81_TRAINING_BUDGET=56000")
    print("T81_PREFLIGHT=PASS")

try:
    main()

except Exception as error:
    print(
        "T81_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "FROZEN_MODEL_MODIFIED=FALSE"
    )
    print(
        "MODEL_PROMOTED=FALSE"
    )
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
