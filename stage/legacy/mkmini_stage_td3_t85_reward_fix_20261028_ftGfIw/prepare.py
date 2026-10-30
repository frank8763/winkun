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
JOB = Path(os.environ["T85_JOB"])

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

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()

def check(value, message):
    if not value:
        raise RuntimeError(message)

def replace_once(source, old, new, name):
    check(
        source.count(old) == 1,
        "PATCH_NOT_UNIQUE:" + name
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
        "TOP_BINDING_INVALID:" + name
    )

    n = nodes[0]
    lines = source.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        f"{name} = {expression}\n"
    ]

    result = "".join(lines)
    ast.parse(result)
    return result

try:
    print("=" * 78)
    print("T85 — TD3 GOAL DISTANCE REWARD CORRECTION")
    print("=" * 78)

    print("STAGE_ONLY=TRUE")
    print("TRAINING_STARTED=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")

    train_path = BASE / "train.py"
    core_path = BASE / "td3_core_goal_contrast.py"

    check(
        train_path.is_file()
        and sha(train_path) == TRAIN_SHA,
        "ORIGINAL_TRAIN_CHANGED"
    )

    check(
        core_path.is_file()
        and sha(core_path) == CORE_SHA,
        "ORIGINAL_CORE_CHANGED"
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
        and initial["manifest"]["source_core"]
            == CORE_SHA
        and initial["manifest"]["experiment"]
            == "STAGE_TD3_T77",
        "INITIAL_IDENTITY_INVALID"
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

    check(
        int(parent["frames"]) == 134491,
        "PARENT_FRAME_MISMATCH"
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
            "INITIAL_NETWORK_MISMATCH:" + key
        )

    print("T85_IDENTICAL_INITIAL_NETWORK=PASS")

    source = train_path.read_text()

    check(
        source.count("STAGE_TD3_T71B") == 1,
        "EXPERIMENT_ID_UNEXPECTED"
    )

    # T85：只修改T66进度奖励的距离来源。
    # obs[48]与next_obs[48]已在此前112维观测
    # 审计中确认对应目标距离/10。
    old_before = (
        "_t66_before = "
        "float(env._goal_state()[1])"
    )

    old_after = (
        "_t66_after = "
        "float(env._goal_state()[1])"
    )

    new_before = (
        "_t66_before = "
        "10.0 * float(obs[48])"
    )

    new_after = (
        "_t66_after = "
        "10.0 * float(next_obs[48])"
    )

    source = replace_once(
        source,
        old_before,
        new_before,
        "DISTANCE_BEFORE"
    )

    source = replace_once(
        source,
        old_after,
        new_after,
        "DISTANCE_AFTER"
    )

    # 增加只读语义检查。
    # 发现观测与环境内部目标距离不一致就停止。
    audit_anchor = (
        "                _t66_delta = "
        "max(-0.02, min(0.02, "
        "_t66_before - _t66_after))"
    )

    audit_code = '''
                if not globals().get(
                    "_T85_DISTANCE_GATE_DONE", False
                ):
                    check(
                        len(obs) == 112
                        and len(next_obs) == 112,
                        "T85_OBSERVATION_DIM_INVALID"
                    )

                    _t85_native_distance = (
                        10.0 * float(
                            env._goal_state()[0]
                        )
                    )

                    check(
                        abs(
                            _t66_after
                            - _t85_native_distance
                        ) < 0.005,
                        "T85_DISTANCE_SEMANTICS_INVALID"
                    )

                    check(
                        0.0 <= _t66_before <= 10.0
                        and 0.0 <= _t66_after <= 10.0,
                        "T85_DISTANCE_RANGE_INVALID"
                    )

                    print(
                        "T85_DISTANCE_SEMANTICS=PASS",
                        "BEFORE_M=",
                        round(_t66_before, 6),
                        "AFTER_M=",
                        round(_t66_after, 6),
                        flush=True
                    )

                    globals()[
                        "_T85_DISTANCE_GATE_DONE"
                    ] = True

'''

    source = replace_once(
        source,
        audit_anchor,
        audit_code + audit_anchor,
        "DISTANCE_SEMANTIC_AUDIT"
    )

    source = replace_top(
        source,
        "JOB",
        f"Path({str(JOB)!r})"
    )

    source = replace_once(
        source,
        "STAGE_TD3_T71B",
        "STAGE_TD3_T85",
        "EXPERIMENT_ID"
    )

    ast.parse(source)

    check(
        "16.0 * _t66_delta" in source
        and "max(-0.02, min(0.02" in source,
        "T66_REWARD_FORMULA_CHANGED"
    )

    check(
        source.count(
            "float(env._goal_state()[1])"
        ) == 0,
        "OLD_T66_VARIABLE_REMAINS"
    )

    check(
        not (JOB / "train.py").exists(),
        "T85_ALREADY_PREPARED"
    )

    (JOB / "train.py").write_text(source)

    shutil.copy2(
        core_path,
        JOB / "td3_core_goal_contrast.py"
    )

    shutil.copy2(
        BASE / "goal_pairs.npz",
        JOB / "goal_pairs.npz"
    )

    (JOB / "checkpoints").mkdir(
        exist_ok=True
    )

    child = copy.deepcopy(initial)

    child["manifest"]["experiment"] = (
        "STAGE_TD3_T85"
    )

    child["manifest"]["target_frames"] = 190491

    destination = (
        JOB / "checkpoints/episode_0022.pt"
    )

    torch.save(child, destination)

    status = copy.deepcopy(parent_status)

    status.update({
        "state": "RUNNING",
        "frames": 134491,
        "target_frames": 190491,
        "checkpoint": destination.name,
        "checkpoint_sha256": sha(destination),
        "automatic_promotion": False
    })

    (JOB / "status.json").write_text(
        json.dumps(
            status,
            indent=2
        ) + "\n"
    )

    report = {
        "experiment": "STAGE_TD3_T85",
        "baseline": "T71B",
        "frozen_parent": "T67C022",
        "initial_frames": 134491,
        "target_frames": 190491,
        "single_functional_change":
            "T66 reward uses distance progress in meters",
        "distance_before": "10 * obs[48]",
        "distance_after": "10 * next_obs[48]",
        "reward_coefficient": 16.0,
        "reward_delta_clip_m": 0.02,
        "actor_changes": False,
        "replay_changes": False,
        "environment_changes": False,
        "automatic_promotion": False,
        "source_sha256":
            sha(JOB / "train.py"),
        "core_sha256":
            sha(JOB / "td3_core_goal_contrast.py"),
        "initial_checkpoint_sha256":
            sha(destination)
    }

    (JOB / "experiment.json").write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print("T85_SOURCE_SHA256=", sha(JOB / "train.py"))
    print("T85_CORE_SHA256=", sha(JOB / "td3_core_goal_contrast.py"))
    print("T85_INITIAL_CHECKPOINT=PASS")
    print("T85_REWARD_PATCH=PASS")
    print("T85_PREFLIGHT=PASS")

except Exception as e:
    print("T85_PREFLIGHT_BLOCKED=", repr(e))
    traceback.print_exc()

finally:
    print("FROZEN_MODEL_MODIFIED=FALSE")
    print("MODEL_PROMOTED=FALSE")
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
