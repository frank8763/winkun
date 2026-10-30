from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t19_neargoal_learning_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

SOURCE = T16 / "train.py"
CORE = T16 / "td3_core_goal_contrast.py"
PAIRS = T16 / "goal_pairs.npz"
PARENT = T16 / "checkpoints/episode_0008.pt"

HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

PARENT_SHA = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

FROZEN_SHA = (
    "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8"
)

HELPER_SHA = (
    "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d"
)

STAGE_SHA = (
    "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556"
)

INITIAL = 13491
TARGET = 17491
BETA = 0.04

DISTANCES = np.asarray(
    [0.25, 0.35, 0.45, 0.65, 0.90],
    dtype=np.float64
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def patch_once(source, old, new):
    check(
        source.count(old) == 1,
        "PATCH_ANCHOR_INVALID:" + old[:90]
    )
    return source.replace(old, new, 1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )
    check(
        spec is not None and spec.loader is not None,
        "MODULE_LOAD_SPEC_INVALID"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path, data):
    temp = Path(str(path) + ".partial")
    temp.write_text(
        json.dumps(
            data, indent=2, ensure_ascii=False
        ) + "\n"
    )
    os.replace(temp, path)


def verify_parent():
    prep = T16 / "prepare.json"

    check(prep.is_file(), "T16_PREPARE_MISSING")

    info = json.loads(prep.read_text())

    for name, expected in info["files"].items():
        path = T16 / name
        check(
            path.is_file() and sha(path) == expected,
            "T16_SOURCE_CHANGED:" + name
        )

    for path, expected in (
        (PARENT, PARENT_SHA),
        (FROZEN, FROZEN_SHA),
        (HELPER, HELPER_SHA),
        (STAGE, STAGE_SHA)
    ):
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    state = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        state.get("algorithm") == "TD3",
        "PARENT_ALGORITHM_INVALID"
    )

    check(
        state["frames"] == INITIAL,
        "PARENT_FRAMES_INVALID"
    )

    check(
        state["updates"] == 6618,
        "PARENT_UPDATES_INVALID"
    )

    check(
        state["episode_index"] == 8,
        "PARENT_EPISODE_INVALID"
    )

    check(
        state["manifest"]["source_core"] == sha(CORE),
        "PARENT_CORE_PROVENANCE_INVALID"
    )

    check(
        state["manifest"]["goal_pair_dataset_sha256"]
        == sha(PAIRS),
        "PARENT_GOAL_PAIR_PROVENANCE_INVALID"
    )

    print("T19_PARENT_VERIFICATION=PASS")
    print("T19_INITIAL_FRAMES=", INITIAL)

    return state


def identify_goal_distance_channel():
    """
    保持车体和传感器状态不变。
    只修改最终目标距离，检查112维原生观测。
    不执行物理步，不覆盖动作。
    """

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER,
        "stage_t19_helper"
    )

    legacy = load_module(
        STAGE,
        "stage_t19_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "stage_t19_native"
    )

    _, v, native_mod, prep, hashes = runner.load_native()

    check(
        isinstance(hashes, dict) and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_FAILED"
    )

    N = prep[3]
    make_base = legacy.make_short_task(runner.make_task)

    groups = []

    for seed in (
        2029619201,
        2029619202,
        2029619203
    ):
        for angle in (-20, 20):

            env = None
            renderer = None

            try:
                env, renderer, _, actual_seed = (
                    helper.make_goal_task(
                        make_base,
                        v,
                        native_mod,
                        N,
                        seed,
                        angle,
                        0.8
                    )
                )

                start = np.asarray(
                    env.data.qpos[:2],
                    dtype=np.float64
                ).copy()

                yaw = float(env._pose()[2])
                bearing = yaw + math.radians(angle)

                direction = np.asarray([
                    math.cos(bearing),
                    math.sin(bearing)
                ])

                rows = []

                for distance in DISTANCES:

                    env.goal_xy = (
                        start + distance * direction
                    )

                    actual_distance = float(
                        env._goal_state()[1]
                    )

                    check(
                        abs(actual_distance - distance)
                        < 0.005,
                        "COUNTERFACTUAL_DISTANCE_INVALID"
                    )

                    env.prev_goal_dist = actual_distance

                    obs = np.asarray(
                        env._observation(
                            update_history=False
                        ),
                        dtype=np.float64
                    ).reshape(-1).copy()

                    check(
                        obs.shape == (112,)
                        and np.isfinite(obs).all(),
                        "COUNTERFACTUAL_OBSERVATION_INVALID"
                    )

                    rows.append(obs)

                groups.append(np.stack(rows))

            finally:
                if renderer is not None:
                    renderer.close()

                if env is not None:
                    env.close()

    check(
        len(groups) == 6,
        "GOAL_PROBE_GROUP_COUNT_INVALID"
    )

    data = np.stack(groups)

    candidates = []

    for index in range(48, 112):

        values = data[:, :, index]

        mean_values = values.mean(axis=0)

        span = float(np.ptp(mean_values))

        # 通道必须随距离明显变化。
        if span < 0.025:
            continue

        # 相同距离下，不同场景的编码必须一致。
        across_scene_error = float(
            np.max(np.ptp(values, axis=0))
        )

        if across_scene_error > 0.015:
            continue

        slope, intercept = np.polyfit(
            DISTANCES,
            mean_values,
            1
        )

        fitted = slope * DISTANCES + intercept

        residual = float(
            np.max(np.abs(
                fitted - mean_values
            ))
        )

        if residual > 0.015:
            continue

        candidates.append({
            "index": index,
            "slope": float(slope),
            "span": span,
            "residual": residual,
            "across_scene_error": across_scene_error,
            "value_at_025": float(mean_values[0]),
            "value_at_045": float(mean_values[2]),
            "values": mean_values.tolist()
        })

    print(
        "T19_GOAL_DISTANCE_CANDIDATES=",
        json.dumps(
            candidates,
            ensure_ascii=False
        )
    )

    # 第48号通道若符合全部实测条件，优先使用。
    # 否则只允许唯一候选。
    by_index = {
        item["index"]: item
        for item in candidates
    }

    if 48 in by_index:
        selected = by_index[48]
    else:
        check(
            len(candidates) == 1,
            "GOAL_DISTANCE_CHANNEL_AMBIGUOUS"
        )
        selected = candidates[0]

    low = selected["value_at_025"]
    high = selected["value_at_045"]

    check(
        abs(high - low) > 0.01,
        "GOAL_DISTANCE_ENCODING_TOO_SMALL"
    )

    # 确认该通道在所有6组中都可重复编码距离。
    normalized = (
        data[:, :, selected["index"]] - low
    ) / (high - low)

    expected = (
        DISTANCES - 0.25
    ) / 0.20

    normalized_error = float(
        np.max(np.abs(
            normalized - expected[None, :]
        ))
    )

    check(
        normalized_error < 0.35,
        "GOAL_DISTANCE_ENCODING_NONLINEAR_OR_UNSTABLE"
    )

    print(
        "T19_GOAL_DISTANCE_CHANNEL=PASS",
        json.dumps(
            selected,
            ensure_ascii=False
        )
    )

    return selected


def make_regularized_core(original, goal):
    tree = ast.parse(original)

    # T9已有的动作正则化结果应被T12/T16继承。
    actor_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == "td3_t9_actions"
            for t in node.targets
        )
    ]

    check(
        len(actor_assignments) == 1,
        "T9_ACTOR_ACTION_ANCHOR_INVALID"
    )

    check(
        "t12_goal_loss =" in original,
        "T12_GOAL_LOSS_MISSING"
    )

    lines = original.splitlines(keepends=True)

    indices = [
        i for i, line in enumerate(lines)
        if line.strip() == "actor_loss.backward()"
    ]

    check(
        len(indices) == 1,
        "ACTOR_BACKWARD_ANCHOR_INVALID"
    )

    index = indices[0]
    old = lines[index]

    indent = old[:len(old) - len(old.lstrip())]

    actor_states = actor_assignments[0].value

    check(
        isinstance(actor_states, ast.Call)
        and len(actor_states.args) == 1,
        "ACTOR_STATES_UNAVAILABLE"
    )

    state_expression = ast.unparse(
        actor_states.args[0]
    )

    channel = int(goal["index"])

    low = float(goal["value_at_025"])
    delta = (
        float(goal["value_at_045"]) - low
    )

    # 0.25米对应0，0.45米对应1。
    # 在约0.35米附近增加平滑油门惩罚。
    # 该项只作用于Actor优化，不修改环境动作。
    addition = (
        f"{indent}t19_distance_feature = "
        f"({state_expression})[:, {channel}]\n"

        f"{indent}t19_distance_scaled = "
        f"(t19_distance_feature - ({low:.12f}))"
        f" / ({delta:.12f})\n"

        f"{indent}t19_near_weight = "
        "torch.sigmoid("
        "8.0 * (0.5 - t19_distance_scaled)"
        ")\n"

        f"{indent}t19_policy_throttle = "
        "0.5 * (td3_t9_actions[:, 0] + 1.0)\n"

        f"{indent}t19_near_loss = "
        "(t19_near_weight * "
        "t19_policy_throttle.square()).mean()\n"

        f"{indent}actor_loss = actor_loss + "
        f"{BETA:.8f} * t19_near_loss\n"
    )

    lines.insert(index, addition)

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count("t19_near_loss =") == 1,
        "T19_ACTOR_PATCH_FAILED"
    )

    return result


def main():
    print("STAGE_ONLY=TRUE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("HISTORICAL_CHECKPOINT_WRITE=FALSE")
    print("T19_SINGLE_CHANGE=NEAR_GOAL_ACTOR_THROTTLE_LOSS")

    JOB.mkdir(parents=True, exist_ok=True)

    parent = verify_parent()

    prepared = JOB / "prepare_manifest.json"

    if prepared.exists():
        info = json.loads(prepared.read_text())

        for name, expected in info["files"].items():
            check(
                sha(JOB / name) == expected,
                "T19_PREPARED_FILE_CHANGED:" + name
            )

        status = json.loads(
            (JOB / "status.json").read_text()
        )

        checkpoint = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint)
            == status["checkpoint_sha256"],
            "T19_CHECKPOINT_CHANGED"
        )

        print("T19_PREPARE=EXISTING_VERIFIED")
        print("T19_READY=TRUE")
        return

    for name in (
        "train.py",
        "td3_core_goal_contrast.py",
        "goal_pairs.npz",
        "status.json",
        "checkpoints/episode_0008.pt"
    ):
        check(
            not (JOB / name).exists(),
            "T19_PARTIAL_DIRECTORY:" + name
        )

    goal = identify_goal_distance_channel()

    core_source = make_regularized_core(
        CORE.read_text(),
        goal
    )

    new_core_sha = hashlib.sha256(
        core_source.encode()
    ).hexdigest()

    old_core_sha = sha(CORE)

    script = SOURCE.read_text()

    script = patch_once(
        script,
        'JOB = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"',
        'JOB = H / "mkmini_stage_td3_t19_neargoal_learning_20261027_v1"'
    )

    script = patch_once(
        script,
        '"experiment": "STAGE_TD3_T16"',
        '"experiment": "STAGE_TD3_T19"'
    )

    script = patch_once(
        script,
        "TARGET = 20000",
        f"TARGET = {TARGET}"
    )

    script = patch_once(
        script,
        "WALL = 6300",
        "WALL = 2400"
    )

    script = patch_once(
        script,
        old_core_sha,
        new_core_sha
    )

    ast.parse(script)

    (JOB / "train.py").write_text(script)

    (JOB / "td3_core_goal_contrast.py").write_text(
        core_source
    )

    shutil.copy2(
        PAIRS,
        JOB / "goal_pairs.npz"
    )

    check(
        sha(JOB / "td3_core_goal_contrast.py")
        == new_core_sha,
        "T19_CORE_COPY_INVALID"
    )

    check(
        sha(JOB / "goal_pairs.npz") == sha(PAIRS),
        "T19_GOAL_PAIRS_CHANGED"
    )

    manifest = copy.deepcopy(
        parent["manifest"]
    )

    manifest["experiment"] = "STAGE_TD3_T19"
    manifest["source_core"] = new_core_sha
    manifest["target_frames"] = TARGET
    manifest["near_goal_beta"] = BETA
    manifest["near_goal_channel"] = goal["index"]
    manifest["near_goal_encoding"] = goal

    state = copy.deepcopy(parent)
    state["manifest"] = manifest

    checkpoint_dir = JOB / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = (
        checkpoint_dir / "episode_0008.pt"
    )

    temporary = (
        checkpoint_dir / "initial.pt.partial"
    )

    torch.save(state, temporary)
    os.replace(temporary, checkpoint)

    atomic_json(
        JOB / "status.json",
        {
            "state": "RUNNING",
            "frames": INITIAL,
            "target_frames": TARGET,
            "checkpoint": "episode_0008.pt",
            "checkpoint_sha256": sha(checkpoint),
            "automatic_promotion": False
        }
    )

    files = {
        name: sha(JOB / name)
        for name in (
            "train.py",
            "td3_core_goal_contrast.py",
            "goal_pairs.npz"
        )
    }

    atomic_json(
        prepared,
        {
            "experiment": "STAGE_TD3_T19",
            "parent_checkpoint_sha256": PARENT_SHA,
            "initial_frames": INITIAL,
            "target_frames": TARGET,
            "single_change":
                "near_goal_actor_throttle_loss",
            "beta": BETA,
            "goal_distance_channel": goal,
            "files": files,
            "automatic_promotion": False
        }
    )

    print("T19_NEW_BRANCH=CREATED")
    print("T19_PARENT_FRAMES=", INITIAL)
    print("T19_TARGET_FRAMES=", TARGET)
    print("T19_NEW_TRAINING_FRAMES=4000")
    print("T19_NEW_CORE_SHA256=", new_core_sha)
    print("T19_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T19_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
