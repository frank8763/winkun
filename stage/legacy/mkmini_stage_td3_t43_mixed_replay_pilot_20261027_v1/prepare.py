from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import re
import textwrap
import traceback

import numpy as np
import torch

H = Path.home()

JOB = H / "mkmini_stage_td3_t43_mixed_replay_pilot_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"

T31 = (
    H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"
    / "t31a"
)

T41 = H / "mkmini_stage_td3_t41_dual_replay_foundation_20261027_v1"

T42 = H / "mkmini_stage_td3_t42_verified_3m_replay_20261027_v1"

PARENT = T16 / "checkpoints/episode_0008.pt"
CORE = T31 / "td3_core_goal_contrast.py"
SOURCE = T31 / "train.py"
PAIRS = T31 / "goal_pairs.npz"

HISTORY = T41 / "t16_frozen_experience.pt"
SUCCESS = T42 / "verified_3m_success.npz"

PARENT_SHA = (
    "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
)

CORE_SHA = (
    "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
)

HISTORY_SHA = (
    "440e3887596e6900dde36e59c69d1823be6cf6948b3c92ac3e4c60e9eafb6de0"
)

SUCCESS_SHA = (
    "743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b"
)

INITIAL = 13491
TARGET = 25491

# 每回合仍使用原生Stage目标生成器。
# 这只是混合经验机制先导测试。
TASKS = (
    (20, 3.0),
    (20, 0.4),
    (-20, 3.0),
    (-20, 1.2),
)


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def atomic_json(path, obj):
    temp = path.with_name(path.name + ".partial")

    temp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(temp, path)


def patch_assignments(source, values):
    tree = ast.parse(source)

    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in values
            ):
                check(
                    target.id not in found,
                    "DUPLICATE_ASSIGNMENT:" + target.id
                )
                found[target.id] = node

    check(
        set(found) == set(values),
        "MISSING_ASSIGNMENTS:"
        + repr(set(values) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [values[name] + "\n"]

    patched = "".join(lines)
    ast.parse(patched)

    return patched


def patch_sampling(source):

    pattern = re.compile(
        r"(?m)^(?P<ind>[ \t]+)ids = rng\.choice\(\n"
        r"(?P=ind)    len\(replay\),\n"
        r"(?P=ind)    size=64,\n"
        r"(?P=ind)    replace=False\n"
        r"(?P=ind)\)\n"
        r"\n"
        r"(?P=ind)samples = \[\n"
        r"(?P=ind)    replay\[int\(i\)\]\n"
        r"(?P=ind)    for i in ids\n"
        r"(?P=ind)\]\n"
    )

    matches = list(pattern.finditer(source))

    check(
        len(matches) == 1,
        "SAMPLING_ANCHOR_NOT_UNIQUE:"
        + str(len(matches))
    )

    match = matches[0]
    indent = match.group("ind")

    addition = textwrap.dedent("""
    # T43: 持久历史回放 + 真实3米成功轨迹。
    # 只改变训练批次组成，不覆盖推理动作。

    if not hasattr(agent, "_t43_history"):
        import hashlib as _t43_hashlib
        import numpy as _t43_np
        import torch as _t43_torch

        def _t43_sha(path):
            h = _t43_hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(
                    lambda: f.read(1048576), b""
                ):
                    h.update(chunk)
            return h.hexdigest()

        if _t43_sha(HISTORY_PATH) != HISTORY_HASH:
            raise RuntimeError("T43_HISTORY_HASH_FAILED")

        if _t43_sha(SUCCESS_PATH) != SUCCESS_HASH:
            raise RuntimeError("T43_SUCCESS_HASH_FAILED")

        _history_obj = _t43_torch.load(
            HISTORY_PATH,
            map_location="cpu",
            weights_only=False
        )

        if (
            _history_obj.get("parent_checkpoint_sha256")
            != PARENT_HASH
            or not _history_obj.get(
                "episode_segmentation_proven"
            )
        ):
            raise RuntimeError("T43_HISTORY_PROVENANCE_INVALID")

        _old = _history_obj["replay"]
        _episodes = _history_obj["episodes"]

        if len(_old) != 13491 or len(_episodes) != 8:
            raise RuntimeError("T43_HISTORY_LENGTH_INVALID")

        _good_indices = []
        _offset = 0

        for _episode in _episodes:
            _count = int(_episode["physics_frames"])

            if (
                _episode["success"]
                and not _episode["collision"]
            ):
                _good_indices.extend(
                    range(_offset, _offset + _count)
                )

            _offset += _count

        if _offset != len(_old) or len(_good_indices) < 8:
            raise RuntimeError("T43_HISTORY_SEGMENTS_INVALID")

        with _t43_np.load(SUCCESS_PATH) as _archive:
            _s = _t43_np.asarray(
                _archive["states"],
                dtype=_t43_np.float32
            ).copy()

            _a = _t43_np.asarray(
                _archive["raw_actions"],
                dtype=_t43_np.float32
            ).copy()

            _r = _t43_np.asarray(
                _archive["rewards"],
                dtype=_t43_np.float32
            ).copy()

            _ns = _t43_np.asarray(
                _archive["next_states"],
                dtype=_t43_np.float32
            ).copy()

            _d = _t43_np.asarray(
                _archive["dones"],
                dtype=_t43_np.float32
            ).copy()

            _offsets = _archive[
                "episode_offsets"
            ].tolist()

        if (
            _s.shape != (20319, 112)
            or _a.shape != (20319, 2)
            or _r.shape != (20319,)
            or _ns.shape != (20319, 112)
            or _d.shape != (20319,)
            or _offsets != [0, 10612, 20319]
            or _t43_np.flatnonzero(_d).tolist()
                != [10611, 20318]
            or not all(
                _t43_np.isfinite(x).all()
                for x in (_s, _a, _r, _ns, _d)
            )
        ):
            raise RuntimeError("T43_3M_TRANSITIONS_INVALID")

        agent._t43_history = _old
        agent._t43_old_success_indices = (
            _t43_np.asarray(
                _good_indices,
                dtype=_t43_np.int64
            )
        )

        agent._t43_3m = (
            _s, _a, _r, _ns, _d
        )

        agent._t43_batches = 0

        print(
            "T43_ARCHIVES_VERIFIED=PASS",
            "OLD_TRANSITIONS=13491",
            "OLD_SUCCESS_TRANSITIONS=",
            len(_good_indices),
            "THREE_M_TRANSITIONS=20319",
            flush=True
        )

    # 使用独立、可由更新计数重建的随机源。
    # 不额外消耗原训练器的探索随机数。
    _mix_rng = _t43_np.random.default_rng(
        202610274300000 + int(agent.total_updates)
    )

    _hist = agent._t43_history
    _good = agent._t43_old_success_indices

    _old_random_ids = _mix_rng.choice(
        len(_hist),
        size=8,
        replace=False
    )

    _old_good_ids = _mix_rng.choice(
        _good,
        size=8,
        replace=False
    )

    # 两侧真实3米成功轨迹各8条。
    _left_ids = _mix_rng.choice(
        10612,
        size=8,
        replace=False
    )

    _right_ids = _mix_rng.choice(
        _t43_np.arange(10612, 20319),
        size=8,
        replace=False
    )

    _s, _a, _r, _ns, _d = agent._t43_3m

    def _three_m_transition(i):
        i = int(i)
        return (
            _s[i],
            _a[i],
            float(_r[i]),
            _ns[i],
            float(_d[i]),
            "LEFT" if i < 10612 else "RIGHT"
        )

    # 保留32条滚动经验；
    # 将其余32条替换为独立历史样本。
    samples[32:40] = [
        _hist[int(i)] for i in _old_good_ids
    ]

    samples[40:48] = [
        _hist[int(i)] for i in _old_random_ids
    ]

    samples[48:56] = [
        _three_m_transition(i) for i in _left_ids
    ]

    samples[56:64] = [
        _three_m_transition(i) for i in _right_ids
    ]

    if len(samples) != 64:
        raise RuntimeError("T43_MIXED_BATCH_INVALID")

    agent._t43_batches += 1

    if agent._t43_batches == 1:
        print(
            "T43_MIXED_REPLAY_ACTIVE=PASS",
            "ONLINE=32",
            "OLD_SUCCESS=8",
            "OLD_ALL=8",
            "THREE_M_LEFT=8",
            "THREE_M_RIGHT=8",
            "BATCH=64",
            flush=True
        )

    if agent._t43_batches % 2000 == 0:
        print(
            "T43_MIXED_BATCH_COUNT=",
            agent._t43_batches,
            flush=True
        )
    """).strip("\n")

    addition = (
        addition.replace(
            "HISTORY_PATH", repr(str(HISTORY))
        ).replace(
            "SUCCESS_PATH", repr(str(SUCCESS))
        ).replace(
            "HISTORY_HASH", repr(HISTORY_SHA)
        ).replace(
            "SUCCESS_HASH", repr(SUCCESS_SHA)
        ).replace(
            "PARENT_HASH", repr(PARENT_SHA)
        )
    )

    insertion = "".join(
        indent + line + "\n"
        for line in addition.splitlines()
    )

    patched = (
        source[:match.end()]
        + insertion
        + source[match.end():]
    )

    ast.parse(patched)

    check(
        patched.count(
            "T43_MIXED_REPLAY_ACTIVE=PASS"
        ) == 1,
        "T43_PATCH_NOT_UNIQUE"
    )

    return patched


def main():
    print("T43_STAGE_ONLY=TRUE")
    print("T43_PARENT_FROZEN=TRUE")
    print("T43_PLANNER=FALSE")
    print("T43_ACTION_OVERRIDE=FALSE")

    fixed = {
        PARENT: PARENT_SHA,
        CORE: CORE_SHA,
        HISTORY: HISTORY_SHA,
        SUCCESS: SUCCESS_SHA,
    }

    for path, expected in fixed.items():
        check(
            path.is_file() and sha(path) == expected,
            "IMMUTABLE_SOURCE_CHANGED:" + str(path)
        )

    original_manifest = json.loads(
        (T31 / "prepare.json").read_text()
    )

    for name, expected in original_manifest["files"].items():
        path = T31 / name

        check(
            path.is_file()
            and sha(path) == expected,
            "BASE_TRAINER_CHANGED:" + name
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and parent.get("frames") == INITIAL
        and parent.get("updates") == 6618
        and parent.get("episode_index") == 8,
        "PARENT_METADATA_INVALID"
    )

    check(
        parent["manifest"]["source_core"] == CORE_SHA,
        "PARENT_CORE_PROVENANCE_INVALID"
    )

    with np.load(SUCCESS) as data:
        check(
            data["states"].shape == (20319, 112)
            and data["raw_actions"].shape == (20319, 2)
            and data["episode_offsets"].tolist()
                == [0, 10612, 20319],
            "SUCCESS_DATA_INVALID"
        )

    source = patch_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 12000",
            "WALL": "WALL = 1500",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    source = patch_sampling(source)

    old_tag = '"experiment": "STAGE_TD3_T31A"'
    new_tag = '"experiment": "STAGE_TD3_T43R"'

    check(
        source.count(old_tag) == 1,
        "EXPERIMENT_ANCHOR_INVALID"
    )

    source = source.replace(old_tag, new_tag, 1)

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": CORE.read_bytes(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in files.items()
    }

    manifest = {
        "experiment": "STAGE_TD3_T43R",
        "parent_checkpoint_sha256": PARENT_SHA,
        "history_archive_sha256": HISTORY_SHA,
        "success_archive_sha256": SUCCESS_SHA,
        "initial_frames": INITIAL,
        "target_frames": TARGET,
        "tasks": TASKS,
        "batch_composition": {
            "online": 32,
            "old_success": 8,
            "old_all": 8,
            "three_m_left": 8,
            "three_m_right": 8,
        },
        "files": hashes,
        "automatic_promotion": False,
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():
        check(
            json.loads(
                manifest_path.read_text()
            ) == json.loads(
                json.dumps(manifest)
            ),
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, expected in hashes.items():
            check(
                (JOB / name).is_file()
                and sha(JOB / name) == expected,
                "EXISTING_FILE_CHANGED:" + name
            )

        check(
            status_path.is_file(),
            "EXISTING_STATUS_MISSING"
        )

        status = json.loads(status_path.read_text())

        checkpoint = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            checkpoint.is_file()
            and sha(checkpoint)
                == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T43_EXISTING_BRANCH=VERIFIED")

    else:
        for name in files:
            check(
                not (JOB / name).exists(),
                "PARTIAL_BRANCH_FILE:" + name
            )

        check(
            not status_path.exists(),
            "PARTIAL_BRANCH_STATUS"
        )

        ckdir = JOB / "checkpoints"

        check(
            not (ckdir / "episode_0008.pt").exists(),
            "PARTIAL_BRANCH_CHECKPOINT"
        )

        for name, content in files.items():
            (JOB / name).write_bytes(content)

        for name, expected in hashes.items():
            check(
                sha(JOB / name) == expected,
                "GENERATED_FILE_SHA_FAILED"
            )

        new_state = copy.deepcopy(parent)

        new_state["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        new_state["manifest"]["experiment"] = (
            "STAGE_TD3_T43R"
        )

        new_state["manifest"]["target_frames"] = TARGET
        new_state["manifest"]["source_core"] = CORE_SHA

        ckdir.mkdir(parents=True, exist_ok=True)

        checkpoint = ckdir / "episode_0008.pt"

        temporary = ckdir / "parent.partial"

        torch.save(new_state, temporary)
        os.replace(temporary, checkpoint)

        atomic_json(
            status_path,
            {
                "state": "RUNNING",
                "frames": INITIAL,
                "target_frames": TARGET,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        atomic_json(
            manifest_path,
            manifest
        )

        print("T43_BRANCH_CREATED=TRUE")

    print("T43_INITIAL_FRAMES=", INITIAL)
    print("T43_TARGET_FRAMES=", TARGET)
    print("T43_NEW_FRAMES=", TARGET - INITIAL)
    print("T43_TASKS=", TASKS)
    print("T43_ARCHIVE_HASHES=PASS")
    print("T43_READY=TRUE")


try:
    main()

except Exception as error:
    print("T43_PREPARE=BLOCKED", repr(error))
    traceback.print_exc()

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
