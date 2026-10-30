from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import traceback

import torch

H = Path.home()
JOB = H / "mkmini_stage_td3_t59_wheel_clearance_20261027_v1"
BASE = H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"

PARENT = BASE / "checkpoints/episode_0010.pt"
TRAIN = BASE / "train.py"
CORE = BASE / "td3_core_goal_contrast.py"
PAIRS = BASE / "goal_pairs.npz"

PARENT_SHA = (
    "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0"
)

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07"
)

START = 22491
TARGET = 30491

# episode_index=10，因此先执行右侧3米，再执行左侧3米。
# 原环境障碍物不变。
TASKS = (
    (-20.0, 1.2),
    (20.0, 1.2),
    (20.0, 3.0),
    (-20.0, 3.0),
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def save_json(path, obj):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(
        json.dumps(
            obj, indent=2, ensure_ascii=False
        ) + "\n"
    )
    os.replace(temp, path)


def patch_assignments(source, changes):
    tree = ast.parse(source)
    found = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in changes
            ):
                check(
                    target.id not in found,
                    "DUPLICATE:" + target.id
                )
                found[target.id] = node

    check(
        set(found) == set(changes),
        "MISSING_ASSIGNMENTS:"
        + repr(set(changes) - set(found))
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda x: x[1].lineno,
        reverse=True
    ):
        lines[node.lineno-1:node.end_lineno] = [
            changes[name] + "\n"
        ]

    result = "".join(lines)
    ast.parse(result)
    return result


HELPER = '''
_T59_MODEL_CACHE = {}
_T59_COUNTER = 0
_T59_NEAR_COUNT = 0


def _t59_external_clearance_reward(env):
    import math
    import numpy as np
    import mujoco

    global _T59_COUNTER
    global _T59_NEAR_COUNT

    model = env.model
    data = env.data

    key = id(model)

    if key not in _T59_MODEL_CACHE:

        expected = {
            23: "mkmini_base",
            25: "left_front_wheel",
            27: "right_front_wheel",
            29: "left_rear_wheel",
            31: "right_rear_wheel",
        }

        for gid, expected_body in expected.items():

            if gid >= int(model.ngeom):
                raise RuntimeError(
                    "T59_GEOM_ID_MISSING:" + str(gid)
                )

            bid = int(model.geom_bodyid[gid])

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                bid
            )

            if name != expected_body:
                raise RuntimeError(
                    "T59_BODY_MISMATCH:"
                    + str(gid)
                    + ":"
                    + str(name)
                )

        chassis_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            23
        )

        if chassis_name != "chassis_collision":
            raise RuntimeError(
                "T59_CHASSIS_GEOM_CHANGED"
            )

        obstacles = []

        for gid in range(int(model.ngeom)):

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                gid
            )

            if name and name.startswith(
                "real_rect_obs_"
            ):
                obstacles.append(gid)

        obstacle3 = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "real_rect_obs_3"
        )

        if (
            obstacle3 != 4
            or len(obstacles) == 0
        ):
            raise RuntimeError(
                "T59_OBSTACLE_GEOMETRY_CHANGED"
            )

        _T59_MODEL_CACHE[key] = tuple(obstacles)

        print(
            "T59_GEOMETRY_GATE=PASS",
            "FRONT=(25,27)",
            "REAR=(29,31)",
            "CHASSIS=23",
            "RECT_OBSTACLES=",
            len(obstacles),
            flush=True
        )

    obstacles = _T59_MODEL_CACHE[key]

    max_range = 0.35
    fromto = np.zeros(6, dtype=np.float64)

    def distance(gid):

        nearest = max_range

        for oid in obstacles:

            d = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    int(gid),
                    int(oid),
                    max_range,
                    fromto
                )
            )

            # 防止某些MuJoCo版本在无近距离
            # 接触时返回极大浮点数。
            if math.isnan(d):
                raise RuntimeError(
                    "T59_DISTANCE_NAN"
                )

            if d == float("-inf"):
                raise RuntimeError(
                    "T59_DISTANCE_NEGATIVE_INFINITY"
                )

            d = min(max_range, d)

            nearest = min(
                nearest, d
            )

        return nearest

    front = min(
        distance(25),
        distance(27)
    )

    rear = min(
        distance(29),
        distance(31)
    )

    chassis = distance(23)

    def risk(d, margin):
        return max(
            0.0,
            (margin - d) / margin
        ) ** 2

    # 仅用于训练奖励，不修改Actor动作。
    # 左右前轮对称处理，重点避免前轮擦碰。
    penalty = (
        0.018 * risk(front, 0.20)
        + 0.010 * risk(rear, 0.16)
        + 0.012 * risk(chassis, 0.20)
    )

    _T59_COUNTER += 1

    if _T59_COUNTER == 1:

        print(
            "T59_EXTERNAL_SAFETY_REWARD_ACTIVE=PASS",
            "FRONT_M=",
            round(front, 5),
            "REAR_M=",
            round(rear, 5),
            "CHASSIS_M=",
            round(chassis, 5),
            flush=True
        )

    if penalty > 0.0:

        _T59_NEAR_COUNT += 1

        if _T59_NEAR_COUNT == 1:
            print(
                "T59_EXTERNAL_NEAR_OBSTACLE=PASS",
                "FRONT_M=",
                round(front, 5),
                "REAR_M=",
                round(rear, 5),
                "CHASSIS_M=",
                round(chassis, 5),
                "PENALTY=",
                round(penalty, 7),
                flush=True
            )

    if _T59_COUNTER % 1000 == 0:

        print(
            "T59_SAFETY_PROGRESS",
            "FRAMES=",
            _T59_COUNTER,
            "NEAR_FRAMES=",
            _T59_NEAR_COUNT,
            "FRONT_M=",
            round(front, 4),
            "REAR_M=",
            round(rear, 4),
            "CHASSIS_M=",
            round(chassis, 4),
            flush=True
        )

    return float(penalty)
'''


def patch_training_reward(source):

    tree = ast.parse(source)

    steps = []
    appends = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "step_checked"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "N"
            ):
                steps.append(node)

        if isinstance(node, ast.Expr):

            call = node.value

            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "replay"
            ):
                appends.append(node)

    check(
        len(steps) == 1
        and len(appends) == 1,
        "T59_TRANSITION_ANCHORS_INVALID"
    )

    check(
        steps[0].end_lineno < appends[0].lineno,
        "T59_TRANSITION_ORDER_INVALID"
    )

    segment = ast.get_source_segment(
        source,
        appends[0]
    )

    check(
        segment is not None
        and "reward" in segment,
        "T59_REPLAY_REWARD_NOT_FOUND"
    )

    lines = source.splitlines(keepends=True)

    node = appends[0]
    current_line = lines[node.lineno - 1]

    indent = current_line[
        :len(current_line)
        - len(current_line.lstrip())
    ]

    insertion = (
        indent
        + "_t59_penalty = "
        + "_t59_external_clearance_reward(env)\n"
        + indent
        + "reward = float(reward) - _t59_penalty\n"
    )

    lines.insert(
        node.lineno - 1,
        insertion
    )

    result = "".join(lines)
    ast.parse(result)

    tree = ast.parse(result)

    mains = [
        n for n in tree.body
        if (
            isinstance(n, ast.FunctionDef)
            and n.name == "main"
        )
    ]

    check(
        len(mains) == 1,
        "T59_MAIN_ANCHOR_INVALID"
    )

    lines = result.splitlines(keepends=True)
    lines.insert(
        mains[0].lineno - 1,
        "\n" + HELPER + "\n"
    )

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count(
            "T59_EXTERNAL_SAFETY_REWARD_ACTIVE=PASS"
        ) == 1,
        "T59_PATCH_FAILED"
    )

    return result


def main():

    print("T59_STAGE_ONLY=TRUE")
    print("T59_PARENT_FROZEN=TRUE")
    print("T59_PLANNER=FALSE")
    print("T59_WAYPOINTS=FALSE")
    print("T59_ACTION_OVERRIDE=FALSE")
    print("T59_ENV_MODIFICATION=FALSE")

    for path, expected in (
        (PARENT, PARENT_SHA),
        (CORE, CORE_SHA),
    ):

        check(
            path.is_file()
            and sha(path) == expected,
            "SOURCE_SHA_CHANGED:" + str(path)
        )

    old_manifest = json.loads(
        (BASE / "prepare.json").read_text()
    )

    for name, digest in old_manifest["files"].items():

        check(
            (BASE / name).is_file()
            and sha(BASE / name) == digest,
            "PARENT_SOURCE_CHANGED:" + name
        )

    parent = torch.load(
        PARENT,
        map_location="cpu",
        weights_only=False
    )

    check(
        parent.get("algorithm") == "TD3"
        and int(parent["frames"]) == START
        and int(parent["episode_index"]) == 10
        and parent["manifest"]["source_core"]
            == CORE_SHA,
        "PARENT_METADATA_INVALID"
    )

    source = patch_assignments(
        TRAIN.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "TARGET": f"TARGET = {TARGET}",
            "EP_CAP": "EP_CAP = 4000",
            "WALL": "WALL = 2600",
            "TASKS": f"TASKS = {TASKS!r}",
        }
    )

    source = patch_training_reward(source)

    old_tag = '"experiment": "STAGE_TD3_T53"'

    check(
        source.count(old_tag) == 1,
        "T59_EXPERIMENT_ANCHOR_INVALID"
    )

    source = source.replace(
        old_tag,
        '"experiment": "STAGE_TD3_T59"',
        1
    )

    ast.parse(source)

    files = {
        "train.py": source.encode(),
        "td3_core_goal_contrast.py": CORE.read_bytes(),
        "goal_pairs.npz": PAIRS.read_bytes(),
    }

    hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()
    }

    manifest = {
        "experiment": "STAGE_TD3_T59",
        "parent_sha256": PARENT_SHA,
        "initial_frames": START,
        "target_frames": TARGET,
        "episode_cap": 4000,
        "tasks": [list(t) for t in TASKS],
        "external_clearance_reward": True,
        "wheel_and_chassis": True,
        "self_contacts_excluded": True,
        "model_modified": False,
        "automatic_promotion": False,
        "files": hashes,
    }

    manifest_path = JOB / "prepare.json"
    status_path = JOB / "status.json"

    if manifest_path.exists():

        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "EXISTING_MANIFEST_CHANGED"
        )

        for name, digest in hashes.items():

            check(
                (JOB / name).is_file()
                and sha(JOB / name) == digest,
                "EXISTING_SOURCE_CHANGED:" + name
            )

        status = json.loads(
            status_path.read_text()
        )

        ck = (
            JOB / "checkpoints"
            / status["checkpoint"]
        )

        check(
            ck.is_file()
            and sha(ck)
            == status["checkpoint_sha256"],
            "EXISTING_CHECKPOINT_INVALID"
        )

        print("T59_EXISTING_BRANCH=VERIFIED")

    else:

        check(
            not status_path.exists(),
            "PARTIAL_STATUS_FOUND"
        )

        for name in files:
            check(
                not (JOB / name).exists(),
                "PARTIAL_SOURCE_FOUND:" + name
            )

        for name, content in files.items():
            (JOB / name).write_bytes(content)

        for name, digest in hashes.items():

            check(
                sha(JOB / name) == digest,
                "GENERATED_SHA_INVALID:" + name
            )

        branch = copy.deepcopy(parent)

        branch["manifest"] = copy.deepcopy(
            parent["manifest"]
        )

        branch["manifest"]["experiment"] = (
            "STAGE_TD3_T59"
        )

        branch["manifest"]["target_frames"] = TARGET

        ckdir = JOB / "checkpoints"
        ckdir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = (
            ckdir / "episode_0010.pt"
        )

        check(
            not checkpoint.exists(),
            "PARENT_CHECKPOINT_ALREADY_EXISTS"
        )

        temp = (
            ckdir / "parent.partial"
        )

        torch.save(branch, temp)
        os.replace(temp, checkpoint)

        save_json(
            status_path,
            {
                "state": "RUNNING",
                "frames": START,
                "target_frames": TARGET,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha(checkpoint),
                "automatic_promotion": False,
            }
        )

        save_json(
            manifest_path,
            manifest
        )

        print("T59_BRANCH_CREATED=TRUE")

    print("T59_START_FRAMES=", START)
    print("T59_TARGET_FRAMES=", TARGET)
    print("T59_NEW_FRAMES=", TARGET - START)

    print("T59_GEOMETRY_SOURCE=T58_VERIFIED")
    print("T59_REWARD_PATCH=PASS")
    print("T59_READY=TRUE")


try:
    main()

except Exception as error:

    print(
        "T59_PREPARE=BLOCKED",
        repr(error)
    )

    traceback.print_exc()

    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
