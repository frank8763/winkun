from pathlib import Path
import ast
import copy
import hashlib
import json
import math
import os
import shutil
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path(os.environ["T77_JOB"])

B = H / (
    "mkmini_stage_td3_t71_actor_ablation_"
    "20261028_dE1arY/T71B"
)

PARENT = H / (
    "mkmini_stage_td3_t67_3m_nightrelay_"
    "20261027_kEPAW0/T67C022"
)

TRACE = H / (
    "mkmini_stage_td3_t75_observation_contract_"
    "20261028_oWw8e2/trajectory.jsonl"
)

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

def replace_once(source, old, new):
    check(
        source.count(old) == 1,
        "PATCH_ANCHOR_INVALID:" + old
    )
    result = source.replace(old, new, 1)
    ast.parse(result)
    return result

def replace_top(source, name, expression):
    tree = ast.parse(source)

    nodes = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == name
            for t in n.targets
        )
    ]

    check(
        len(nodes) == 1,
        "TOP_ASSIGNMENT_INVALID:" + name
    )

    n = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

def main():
    print("T77_STAGE_ONLY=TRUE")
    print("T77_TRAINING=FALSE")
    print("T77_PLANNER=FALSE")
    print("T77_ACTION_OVERRIDE=FALSE")

    # 检查原有T71B源码。
    meta = json.loads(
        (B / "prepare.json").read_text()
    )

    for filename, expected in meta["files"].items():
        path = B / filename

        check(
            path.is_file()
            and sha(path) == expected,
            "T71B_SOURCE_CHANGED:" + filename
        )

    b_status = json.loads(
        (B / "status.json").read_text()
    )

    check(
        int(b_status["frames"]) == 190491,
        "T71B_FINAL_STATE_INVALID"
    )

    # 使用T71B最初的134491帧检查点，
    # 而不是最终190491帧检查点。
    initial_path = (
        B / "checkpoints/episode_0022.pt"
    )

    check(
        initial_path.is_file(),
        "T71B_INITIAL_CHECKPOINT_MISSING"
    )

    initial = torch.load(
        initial_path,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(initial["frames"]) == 134491
        and initial["manifest"]["experiment"]
            == "STAGE_TD3_T71B",
        "T71B_INITIAL_STATE_INVALID"
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
        "PARENT_CHECKPOINT_INVALID"
    )

    parent = torch.load(
        parent_ck,
        map_location="cpu",
        weights_only=False
    )

    check(
        int(parent["frames"]) == 134491,
        "PARENT_FRAME_COUNT_INVALID"
    )

    # 网络参数与原始T67C022必须相同。
    for name in (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
    ):
        a = initial[name]
        b = parent[name]

        check(
            set(a) == set(b)
            and all(
                torch.equal(a[k], b[k])
                for k in a
            ),
            "INITIAL_NETWORK_MISMATCH:" + name
        )

    print("T77_IDENTICAL_INITIAL_NETWORK=PASS")

    # 验证目标角度的实际归一化公式。
    check(
        TRACE.is_file(),
        "T75_TRACE_MISSING"
    )

    errors = []
    recovery_samples = 0

    for line in TRACE.read_text().splitlines():
        row = json.loads(line)

        if row.get("model") != "T71B":
            continue

        obs = row["t75_obs_112"]

        bearing = math.radians(
            row["goal_error_body_x_deg"]
        )

        errors.append(
            abs(obs[109] - bearing / math.pi)
        )

        if (
            row["angle_deg"] == -12.5
            and abs(bearing) >= 0.45
            and min(row["t75_native_lidar48"])
                * 6.0 >= 1.2
        ):
            recovery_samples += 1

    check(
        errors and max(errors) <= 0.005,
        "GOAL_BEARING_NORMALIZATION_NOT_VERIFIED"
    )

    check(
        recovery_samples >= 5,
        "RECOVERY_STATE_COVERAGE_INSUFFICIENT"
    )

    print(
        "T77_GOAL_BEARING_DIV_PI=PASS",
        "MAX_ERROR=", round(max(errors), 7)
    )

    print(
        "T77_LEFT_RECOVERY_SAMPLES=",
        recovery_samples
    )

    # 复制原有T71B训练代码，只修改奖励。
    train_source = (B / "train.py").read_text()
    core_bytes = (
        B / "td3_core_goal_contrast.py"
    ).read_bytes()

    core_sha = hashlib.sha256(
        core_bytes
    ).hexdigest()

    train_source = replace_top(
        train_source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    train_source = replace_once(
        train_source,
        "STAGE_TD3_T71B",
        "STAGE_TD3_T77"
    )

    old_before = (
        "                _t66_before = "
        "float(env._goal_state()[1])"
    )

    new_before = (
        old_before
        + "\n"
        + "                _t77_angle_before = "
        + "abs(float(env._goal_state()[2]))"
    )

    train_source = replace_once(
        train_source,
        old_before,
        new_before
    )

    old_after = (
        "                reward = "
        "float(reward) + _t66_bonus"
    )

    # 新增的是训练奖励，不修改控制动作。
    new_after = old_after + '''
                _t77_angle_after = abs(
                    float(env._goal_state()[2])
                )

                _t77_scan = np.asarray(
                    env._lidar48(),
                    dtype=np.float32
                ).reshape(-1)

                check(
                    _t77_scan.shape == (48,)
                    and np.isfinite(_t77_scan).all(),
                    "T77_LIDAR_INVALID"
                )

                _t77_clearance_m = (
                    float(np.min(_t77_scan))
                    * float(env.lidar_max_range)
                )

                _t77_gate = (
                    _t77_angle_before >= 0.45
                    and _t77_clearance_m >= 1.2
                )

                _t77_delta = max(
                    -0.01,
                    min(
                        0.01,
                        _t77_angle_before
                        - _t77_angle_after
                    )
                )

                _t77_bonus = (
                    25.0 * _t77_delta
                    if _t77_gate
                    else 0.0
                )

                reward = (
                    float(reward) + _t77_bonus
                )

                _t77_count = (
                    globals().get(
                        "_T77_REWARD_COUNT", 0
                    ) + 1
                )

                globals()[
                    "_T77_REWARD_COUNT"
                ] = _t77_count

                if _t77_gate:
                    globals()[
                        "_T77_ACTIVE_COUNT"
                    ] = (
                        globals().get(
                            "_T77_ACTIVE_COUNT", 0
                        ) + 1
                    )

                if _t77_count % 2000 == 0:
                    print(
                        "T77_RECOVERY_AUDIT",
                        "COUNT=", _t77_count,
                        "ACTIVE=",
                        globals().get(
                            "_T77_ACTIVE_COUNT", 0
                        ),
                        "LAST_BONUS=",
                        round(_t77_bonus, 7),
                        flush=True
                    )
'''

    train_source = replace_once(
        train_source,
        old_after,
        new_after
    )

    ast.parse(train_source)

    # 初始化新的独立目录。
    check(
        not (JOB / "train.py").exists(),
        "T77_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(
        train_source
    )

    (JOB / "td3_core_goal_contrast.py").write_bytes(
        core_bytes
    )

    shutil.copy2(
        B / "goal_pairs.npz",
        JOB / "goal_pairs.npz"
    )

    (JOB / "checkpoints").mkdir(
        exist_ok=True
    )

    child = copy.deepcopy(initial)

    child["manifest"]["experiment"] = (
        "STAGE_TD3_T77"
    )

    child["manifest"]["target_frames"] = 190491

    check(
        child["manifest"]["source_core"]
            == core_sha,
        "CORE_MANIFEST_MISMATCH"
    )

    destination = (
        JOB / "checkpoints/episode_0022.pt"
    )

    torch.save(child, destination)

    new_status = copy.deepcopy(parent_status)

    new_status.update({
        "state": "RUNNING",
        "frames": 134491,
        "target_frames": 190491,
        "checkpoint": destination.name,
        "checkpoint_sha256": sha(destination),
        "automatic_promotion": False
    })

    (JOB / "status.json").write_text(
        json.dumps(
            new_status,
            indent=2
        ) + "\n"
    )

    report = {
        "experiment": "T77",
        "parent": "T67C022",
        "control": "T71B",
        "initial_frames": 134491,
        "target_frames": 190491,
        "single_changed_variable":
            "conditional_goal_heading_recovery_reward",
        "reward_coefficient": 25.0,
        "heading_gate_rad": 0.45,
        "clearance_gate_m": 1.2,
        "parent_checkpoint_sha256":
            parent_status["checkpoint_sha256"],
        "initial_checkpoint_sha256":
            sha(destination),
        "train_source_sha256":
            sha(JOB / "train.py"),
        "core_sha256": core_sha,
        "automatic_promotion": False
    }

    (JOB / "experiment.json").write_text(
        json.dumps(
            report, indent=2
        ) + "\n"
    )

    print("T77_TRAIN_SOURCE=PASS")
    print("T77_INITIAL_CHECKPOINT=PASS")
    print("T77_SINGLE_VARIABLE=HEADING_RECOVERY_REWARD")
    print("T77_PREFLIGHT=PASS")

try:
    main()

except Exception as error:
    print(
        "T77_PREFLIGHT_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
