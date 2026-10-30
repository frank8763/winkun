from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback

import mujoco
import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t12_goal_contrast_20261027_v1"

T9 = H / "mkmini_stage_td3_t9_regularized_20261027_v1"
T6 = H / "mkmini_stage_td3_t6_balanced_20261027_v1"

SOURCE = T9 / "train.py"
CORE = T9 / "td3_core_regularized.py"
PARENT = T9 / "checkpoints/episode_0002.pt"
T6_PARENT = T6 / "checkpoints/episode_0002.pt"

HELPER = H / (
    "mkmini_stage_td3_independent_"
    "20261027_105202_577031_v1/td3_smoke.py"
)

STAGE = H / (
    "mkmini_stage_c147_fixedgoal_sac_20261027_v1/"
    "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = H / (
    "mkmini_stage_c122s0_fullcap_"
    "20261026_125601_1360056_v1/"
    "train/work_resume.pt"
)

REFERENCE = T9 / "checkpoints/episode_0004.pt"

EXPECTED = {
    CORE:
        "f0d8b64ab797b02fbbb3caa6e5be84a33def431c9f086e76be21eb4586da91b8",
    T6_PARENT:
        "ee66ed967bf8bc2477d2d6c49f008d84b405b4650b8745c95d2e97e79d590150",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
    REFERENCE:
        "fdbcc174597d2980b1541734e2b5d832bd0dbb4c324f8528419a152e777b9f75",
}

BETA = 0.05
MARGIN = 0.4
PAIR_COUNT = 20


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def emit(name, value):
    print(
        name,
        json.dumps(value, ensure_ascii=False, default=str),
        flush=True
    )


def replace_once(source, before, after):
    need(
        source.count(before) == 1,
        "PATCH_ANCHOR_INVALID:" + before[:100]
    )
    return source.replace(before, after, 1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )
    need(spec is not None and spec.loader is not None,
         "MODULE_IMPORT_INVALID")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path, data):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(tmp, path)


def verify_sources():
    for path, expected in EXPECTED.items():
        need(
            path.is_file() and sha(path) == expected,
            "SOURCE_HASH_MISMATCH:" + str(path)
        )

    need(SOURCE.is_file() and PARENT.is_file(),
         "T9_SOURCE_OR_PARENT_MISSING")

    ast.parse(SOURCE.read_text())
    ast.parse(CORE.read_text())

    t6 = torch.load(
        T6_PARENT,
        map_location="cpu",
        weights_only=False
    )
    t9 = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    need(
        t6["frames"] == t9["frames"] == 1675,
        "INITIAL_FRAME_MISMATCH"
    )

    need(
        t6["updates"] == t9["updates"] == 710,
        "INITIAL_UPDATE_MISMATCH"
    )

    need(
        t9["episode_index"] == 2,
        "INITIAL_EPISODE_MISMATCH"
    )

    for network in (
        "actor",
        "actor_target",
        "critic",
        "critic_target"
    ):
        for key, tensor in t6[network].items():
            need(
                torch.equal(tensor, t9[network][key]),
                "PARENT_WEIGHTS_DIFFER:"
                + network + ":" + key
            )

    need(
        t9["manifest"]["source_core"] == EXPECTED[CORE],
        "T9_CORE_MANIFEST_MISMATCH"
    )

    emit("TD3_T12_PARENT=PASS", {
        "frames": 1675,
        "updates": 710,
        "actor_critic_identical_to_T6": True,
        "T9_reference_sha256": EXPECTED[REFERENCE]
    })

    return t9


def build_pairs():
    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load_module(
        HELPER, "stage_td3_t12_helper"
    )
    legacy = load_module(
        STAGE, "stage_td3_t12_stage"
    )

    for path, expected in legacy.EXPECTED.items():
        need(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_DEPENDENCY_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC, "stage_td3_t12_runner"
    )

    _, v, native_mod, prep, hashes = runner.load_native()

    need(
        isinstance(hashes, dict) and len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]
    make_base = legacy.make_short_task(runner.make_task)

    left_states = []
    right_states = []
    seeds = []

    for offset in range(80):
        if len(left_states) >= PAIR_COUNT:
            break

        seed = 2029560000 + offset
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
                    -20.0,
                    0.8
                )
            )

            start = np.asarray(
                env.data.qpos[:2], dtype=np.float64
            ).copy()

            yaw = float(env._pose()[2])

            right_goal = start + 0.8 * np.asarray([
                math.cos(yaw + math.radians(20)),
                math.sin(yaw + math.radians(20))
            ])

            left_goal = np.asarray(
                env.goal_xy, dtype=np.float64
            ).copy()

            # 仅用来检查最终目标位置是否安全。
            # 不进行MuJoCo物理积分。
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ctrl = env.data.ctrl.copy()

            try:
                env.data.qpos[:2] = right_goal
                mujoco.mj_forward(env.model, env.data)
                unsafe = bool(env._collision())
            finally:
                env.data.qpos[:] = qpos
                env.data.qvel[:] = qvel
                if env.data.ctrl.size:
                    env.data.ctrl[:] = ctrl
                mujoco.mj_forward(env.model, env.data)

            if unsafe:
                continue

            observations = {}

            for side, goal in (
                ("LEFT", left_goal),
                ("RIGHT", right_goal),
                ("RECHECK", left_goal)
            ):
                env.goal_xy = goal.copy()
                env.prev_goal_dist = float(
                    env._goal_state()[1]
                )

                obs = np.asarray(
                    env._observation(update_history=False),
                    dtype=np.float32
                ).reshape(-1).copy()

                need(
                    obs.shape == (112,)
                    and np.isfinite(obs).all(),
                    "PAIRED_OBSERVATION_INVALID"
                )

                observations[side] = obs

            left = observations["LEFT"]
            right = observations["RIGHT"]
            repeat = observations["RECHECK"]

            need(
                np.max(np.abs(left - repeat)) < 1e-4,
                "MATCHED_OBSERVATION_NOT_REPEATABLE"
            )

            need(
                np.max(np.abs(left[:48] - right[:48])) < 1e-4,
                "MATCHED_LIDAR_CHANGED"
            )

            changed = np.flatnonzero(
                np.abs(left - right) > 1e-5
            ).tolist()

            need(
                changed == [49, 52],
                "GOAL_CHANNELS_DIFFER_FROM_T11:"
                + str(changed)
            )

            left_states.append(left)
            right_states.append(right)
            seeds.append(int(actual_seed))

            emit("TD3_T12_PAIR=", {
                "count": len(left_states),
                "seed": int(actual_seed),
                "changed_indices": changed,
                "lidar_unchanged": True
            })

        finally:
            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()

    need(
        len(left_states) == PAIR_COUNT,
        "INSUFFICIENT_VALID_GOAL_PAIRS:"
        + str(len(left_states))
    )

    path = JOB / "goal_pairs.npz"
    tmp = JOB / "goal_pairs.npz.partial"

    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            left=np.stack(left_states),
            right=np.stack(right_states),
            seeds=np.asarray(seeds, dtype=np.int64)
        )

    os.replace(tmp, path)

    return sha(path)


def make_core(source):
    lines = source.splitlines(keepends=True)

    indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "actor_loss.backward()"
    ]

    need(
        len(indices) == 1,
        "ACTOR_BACKWARD_ANCHOR_INVALID"
    )

    index = indices[0]
    line = lines[index]

    indent = line[:len(line) - len(line.lstrip())]

    insertion = (
        f"{indent}t12_left, t12_right = self.t12_pairs\n"
        f"{indent}t12_l = self.actor(t12_left)[:, 1]\n"
        f"{indent}t12_r = self.actor(t12_right)[:, 1]\n"
        f"{indent}t12_gap = t12_r - t12_l\n"
        f"{indent}t12_goal_loss = "
        f"torch.relu({MARGIN} - t12_gap).square().mean()\n"
        f"{indent}actor_loss = actor_loss + "
        f"{BETA} * t12_goal_loss\n"
    )

    lines.insert(index, insertion)

    result = "".join(lines)
    ast.parse(result)

    need(
        result.count("t12_goal_loss =") == 1,
        "GOAL_LOSS_PATCH_FAILED"
    )

    return result


def make_train(source, core_sha):
    result = source

    result = replace_once(
        result,
        'JOB = H / "mkmini_stage_td3_t9_regularized_20261027_v1"',
        'JOB = H / "mkmini_stage_td3_t12_goal_contrast_20261027_v1"'
    )

    result = replace_once(
        result,
        'CORE = JOB / "td3_core_regularized.py"',
        'CORE = JOB / "td3_core_goal_contrast.py"'
    )

    result = replace_once(
        result,
        '"experiment": "STAGE_TD3_T9"',
        '"experiment": "STAGE_TD3_T12"'
    )

    result = replace_once(
        result,
        EXPECTED[CORE],
        core_sha
    )

    result = replace_once(
        result,
        '"training_seed": 20261027,',
        '"training_seed": 20261027,\n'
        '        "goal_pair_dataset_sha256": '
        'sha(JOB / "goal_pairs.npz"),\n'
        f'        "goal_rank_margin": {MARGIN},\n'
        f'        "goal_rank_beta": {BETA},'
    )

    result = replace_once(
        result,
        "    agent = create_agent(core)\n",
        "    agent = create_agent(core)\n"
        '    t12_data = np.load('
        'JOB / "goal_pairs.npz", allow_pickle=False)\n'
        '    t12_left = np.asarray('
        't12_data["left"], dtype=np.float32)\n'
        '    t12_right = np.asarray('
        't12_data["right"], dtype=np.float32)\n'
        '    check(t12_left.shape == (20, 112) '
        'and t12_right.shape == (20, 112), '
        '"T12_PAIR_SHAPE_INVALID")\n'
        '    agent.t12_pairs = ('
        'torch.from_numpy(t12_left), '
        'torch.from_numpy(t12_right))\n'
        '    emit("TD3_T12_GOAL_PAIR_GATE=PASS", '
        '{"pairs": len(t12_left), '
        '"margin": 0.4, "beta": 0.05})\n'
    )

    ast.parse(result)
    return result


def main():
    JOB.mkdir(parents=True, exist_ok=True)

    print("STAGE_ONLY=TRUE")
    print("NO_EXTERNAL_PLANNER=TRUE")
    print("NO_WAYPOINTS=TRUE")
    print("FROZEN_MODEL_WRITE=FALSE")
    print("T12_ONLY_CHANGE=GOAL_CONTRASTIVE_ACTOR_LOSS")

    parent_state = verify_sources()

    prepared = JOB / "prepare.json"

    if prepared.is_file():
        info = json.loads(prepared.read_text())

        for name, expected_hash in info["files"].items():
            path = JOB / name
            need(
                path.is_file() and sha(path) == expected_hash,
                "EXISTING_T12_FILE_CHANGED:" + name
            )

        status = json.loads(
            (JOB / "status.json").read_text()
        )

        checkpoint = (
            JOB / "checkpoints" / status["checkpoint"]
        )

        need(
            checkpoint.is_file()
            and sha(checkpoint)
            == status["checkpoint_sha256"],
            "T12_RESUME_CHECKPOINT_INVALID"
        )

        print("TD3_T12_PREPARE=EXISTING_VERIFIED")
        print("TD3_T12_READY=TRUE")
        return

    for relative in (
        "train.py",
        "td3_core_goal_contrast.py",
        "goal_pairs.npz",
        "status.json",
        "checkpoints/episode_0002.pt"
    ):
        need(
            not (JOB / relative).exists(),
            "PARTIAL_T12_DIRECTORY:"
            + relative
        )

    dataset_sha = build_pairs()

    core_source = make_core(CORE.read_text())
    new_core_sha = hashlib.sha256(
        core_source.encode()
    ).hexdigest()

    train_source = make_train(
        SOURCE.read_text(),
        new_core_sha
    )

    core_path = JOB / "td3_core_goal_contrast.py"
    train_path = JOB / "train.py"

    core_path.write_text(core_source)
    train_path.write_text(train_source)

    manifest = dict(parent_state["manifest"])

    manifest["experiment"] = "STAGE_TD3_T12"
    manifest["source_core"] = new_core_sha
    manifest["goal_pair_dataset_sha256"] = dataset_sha
    manifest["goal_rank_margin"] = MARGIN
    manifest["goal_rank_beta"] = BETA

    new_state = copy.deepcopy(parent_state)
    new_state["manifest"] = manifest

    checkpoint_dir = JOB / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_dir / "episode_0002.pt"
    temporary = checkpoint_dir / "initial.pt.partial"

    torch.save(new_state, temporary)
    os.replace(temporary, checkpoint)

    status = {
        "state": "RUNNING",
        "frames": 1675,
        "target_frames": 4000,
        "checkpoint": "episode_0002.pt",
        "checkpoint_sha256": sha(checkpoint),
        "automatic_promotion": False
    }

    atomic_json(JOB / "status.json", status)

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
            "experiment": "STAGE_TD3_T12",
            "parent_t9_sha256": sha(PARENT),
            "parent_t6_sha256": EXPECTED[T6_PARENT],
            "goal_pairs": PAIR_COUNT,
            "actor_goal_margin": MARGIN,
            "actor_goal_beta": BETA,
            "initial_frames": 1675,
            "target_frames": 4000,
            "files": files,
            "automatic_promotion": False
        }
    )

    emit("TD3_T12_PREPARE=PASS", {
        "pairs": PAIR_COUNT,
        "parent_frames": 1675,
        "target_frames": 4000,
        "dataset_sha256": dataset_sha,
        "modified_core_sha256": new_core_sha,
        "frozen_model_modified": False
    })

    print("TD3_T12_READY=TRUE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("TD3_T12_PREPARE=BLOCKED", {
            "error": repr(error)
        })
        traceback.print_exc()
    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
