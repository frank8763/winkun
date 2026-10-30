from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import textwrap
import traceback

import numpy as np
import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t46_success_anchored_20261027_v1"

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T43 = H / "mkmini_stage_td3_t43_mixed_replay_pilot_20261027_v1"
T42 = H / "mkmini_stage_td3_t42_verified_3m_replay_20261027_v1"

PARENT = T16 / "checkpoints/episode_0008.pt"
SOURCE = T43 / "train.py"
CORE = T43 / "td3_core_goal_contrast.py"
PAIRS = T43 / "goal_pairs.npz"
SUCCESS = T42 / "verified_3m_success.npz"

PARENT_SHA = "21501f7c03ddc2cf291f1033daf279820c1c564934189df51fef26670e61c41a"
CORE_SHA = "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68"
SUCCESS_SHA = "743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b"

INITIAL = 13491
TARGET = 16491

# 与已验证成功目标相近，但训练起点由原Stage任务生成器产生。
# 本次仅为3000帧机制试验，不宣称完成随机3米训练。
TASKS = (
    (-17.3, 3.0),
    (13.6, 3.0),
    (-20.0, 0.4),
    (20.0, 0.4),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def write_json(path, obj):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    )
    os.replace(temp, path)


def patch_assignments(source, replacements):
    tree = ast.parse(source)
    nodes = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in replacements:
                        check(
                            target.id not in nodes,
                            "DUPLICATE_ASSIGNMENT:" + target.id
                        )
                        nodes[target.id] = node

    check(
        set(nodes) == set(replacements),
        "MISSING_ASSIGNMENTS:"
        + repr(set(replacements) - set(nodes))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        nodes.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[node.lineno - 1:node.end_lineno] = [
            replacements[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


def patch_core(source):
    lines = source.splitlines(keepends=True)

    matches = [
        i for i, line in enumerate(lines)
        if line.strip() == "actor_loss.backward()"
    ]

    check(
        len(matches) == 1,
        "ACTOR_BACKWARD_ANCHOR_INVALID"
    )

    index = matches[0]
    line = lines[index]
    indent = line[:len(line) - len(line.lstrip())]

    block = textwrap.dedent("""
    # T46: only preserve actions from two verified
    # 3m success trajectories. Training-time only.
    if not hasattr(self, "_t46_success_obs"):
        import hashlib as _bc_hashlib
        import numpy as _bc_numpy

        _bc_path = SUCCESS_PATH_LITERAL
        _bc_hash = _bc_hashlib.sha256()

        with open(_bc_path, "rb") as _bc_file:
            for _bc_chunk in iter(
                lambda: _bc_file.read(1048576), b""
            ):
                _bc_hash.update(_bc_chunk)

        if _bc_hash.hexdigest() != SUCCESS_HASH_LITERAL:
            raise RuntimeError("T46_SUCCESS_ARCHIVE_CHANGED")

        with _bc_numpy.load(_bc_path) as _bc_npz:
            _bc_obs = _bc_numpy.asarray(
                _bc_npz["states"],
                dtype=_bc_numpy.float32
            ).copy()

            _bc_actions = _bc_numpy.asarray(
                _bc_npz["raw_actions"],
                dtype=_bc_numpy.float32
            ).copy()

            _bc_offsets = (
                _bc_npz["episode_offsets"].tolist()
            )

        if (
            _bc_obs.shape != (20319, 112)
            or _bc_actions.shape != (20319, 2)
            or _bc_offsets != [0, 10612, 20319]
            or not _bc_numpy.isfinite(_bc_obs).all()
            or not _bc_numpy.isfinite(_bc_actions).all()
        ):
            raise RuntimeError("T46_SUCCESS_SHAPE_INVALID")

        self._t46_success_obs = torch.as_tensor(
            _bc_obs,
            dtype=state.dtype,
            device=state.device
        )

        self._t46_success_actions = torch.as_tensor(
            _bc_actions,
            dtype=state.dtype,
            device=state.device
        )

        print(
            "T46_SUCCESS_ARCHIVE_LOADED=PASS",
            "LEFT=10612",
            "RIGHT=9707",
            flush=True
        )

    # 左右各128个成功状态，防止一个方向主导损失。
    # 根据更新次数确定采样，不改变原训练器探索随机数。
    _bc_rng = _bc_numpy.random.default_rng(
        202610274600000 + int(self.total_updates)
    )

    _bc_left = _bc_rng.integers(
        0, 10612, size=128
    )

    _bc_right = _bc_rng.integers(
        10612, 20319, size=128
    )

    _bc_indices = torch.as_tensor(
        _bc_numpy.concatenate(
            [_bc_left, _bc_right]
        ),
        dtype=torch.long,
        device=state.device
    )

    _bc_predictions = self.actor(
        self._t46_success_obs[_bc_indices]
    )

    _bc_targets = (
        self._t46_success_actions[_bc_indices]
    )

    _bc_loss = (
        _bc_predictions - _bc_targets
    ).square().mean()

    actor_loss = actor_loss + 1.0 * _bc_loss

    if not hasattr(self, "_t46_bc_announced"):
        self._t46_bc_announced = True
        print(
            "T46_SUCCESS_BC_ACTIVE=PASS",
            "BATCH=256",
            "LEFT=128",
            "RIGHT=128",
            "WEIGHT=1.0",
            flush=True
        )
    """).strip("\n")

    block = block.replace(
        "SUCCESS_PATH_LITERAL", repr(str(SUCCESS))
    ).replace(
        "SUCCESS_HASH_LITERAL", repr(SUCCESS_SHA)
    )

    insertion = "".join(
        indent + part + "\n"
        for part in block.splitlines()
    )

    lines.insert(index, insertion)

    result = "".join(lines)
    ast.parse(result)

    check(
        result.count("T46_SUCCESS_BC_ACTIVE=PASS") == 1,
        "BC_PATCH_FAILED"
    )

    return result


def main():
    print("T46_STAGE_ONLY=TRUE")
    print("T46_PARENT_FROZEN=TRUE")
    print("T46_PLANNER=FALSE")
    print("T46_ACTION_OVERRIDE=FALSE")

    for path, expected in (
        (PARENT, PARENT_SHA),
        (CORE, CORE_SHA),
        (SUCCESS, SUCCESS_SHA),
    ):
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_CHANGED:" + str(path)
        )

    original = json.loads(
        (T43 / "prepare.json").read_text()
    )

    for name, expected in original["files"].items():
        path = T43 / name

        check(
            path.is_file() and sha(path) == expected,
            "T43_SOURCE_CHANGED:" + name
        )

    with np.load(SUCCESS) as archive:
        check(
            archive["states"].shape == (20319, 112)
            and archive["raw_actions"].shape == (20319, 2)
            and archive["episode_offsets"].tolist()
                == [0, 10612, 20319],
            "SUCCESS_ARCHIVE_INVALID"
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
        and parent.get("episode_index") == 8
        and parent["manifest"]["source_core"] == CORE_SHA,
        "PARENT_INVALID"
    )

    new_core = patch_core(CORE.read_text())

    new_core_sha = hashlib.sha256(
        new_core.encode()
    ).hexdigest()

    source = patch_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 12000",
            "WALL": "WALL = 1200",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    check(
        source.count('"experiment": "STAGE_TD3_T43R"') == 1,
        "EXPERIMENT_TAG_INVALID"
    )

    source = source.replace(
        '"experiment": "STAGE_TD3_T43R"',
        '"experiment": "STAGE_TD3_T46B"',
        1
    )

    check(
        source.count(CORE_SHA) == 1,
        "CORE_HASH_ANCHOR_INVALID"
    )

    source = source.replace(
        CORE_SHA,
        new_core_sha,
        1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": new_core.encode(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    manifest = {
        "experiment": "STAGE_TD3_T46B",
        "parent_checkpoint_sha256": PARENT_SHA,
        "success_archive_sha256": SUCCESS_SHA,
        "initial_frames": INITIAL,
        "target_frames": TARGET,
        "tasks": TASKS,
        "bc_weight": 1.0,
        "mixed_replay_inherited": True,
        "files": hashes,
        "automatic_promotion": False,
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == json.loads(json.dumps(manifest)),
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():
            check(
                (JOB / name).is_file()
                and sha(JOB / name) == digest,
                "EXISTING_FILE_CHANGED:" + name
            )

        status = json.loads(status_path.read_text())

        ckpt = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            ckpt.is_file()
            and sha(ckpt) == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_CHANGED"
        )

        print("T46_EXISTING_BRANCH=VERIFIED")

    else:
        for name in files:
            check(
                not (JOB / name).exists(),
                "PARTIAL_BRANCH:" + name
            )

        check(
            not status_path.exists(),
            "PARTIAL_STATUS"
        )

        ckdir = JOB / "checkpoints"

        check(
            not (ckdir / "episode_0008.pt").exists(),
            "PARTIAL_PARENT"
        )

        for name, content in files.items():
            (JOB / name).write_bytes(content)

        for name, digest in hashes.items():
            check(
                sha(JOB / name) == digest,
                "GENERATED_HASH_FAILED:" + name
            )

        branch = copy.deepcopy(parent)
        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T46B"
        )

        branch["manifest"]["source_core"] = new_core_sha
        branch["manifest"]["target_frames"] = TARGET

        ckdir.mkdir(parents=True, exist_ok=True)

        ckpt = ckdir / "episode_0008.pt"
        temp = ckdir / "parent.partial"

        torch.save(branch, temp)
        os.replace(temp, ckpt)

        write_json(
            status_path,
            {
                "state": "RUNNING",
                "frames": INITIAL,
                "target_frames": TARGET,
                "checkpoint": ckpt.name,
                "checkpoint_sha256": sha(ckpt),
                "automatic_promotion": False,
            }
        )

        write_json(manifest_path, manifest)

        print("T46_BRANCH_CREATED=TRUE")

    print("T46_NEW_TRAINING_FRAMES=", TARGET - INITIAL)
    print("T46_CORE_SHA256=", new_core_sha)
    print("T46_READY=TRUE")


try:
    main()
except Exception as exc:
    print("T46_PREPARE=BLOCKED", repr(exc))
    traceback.print_exc()
    raise
finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
