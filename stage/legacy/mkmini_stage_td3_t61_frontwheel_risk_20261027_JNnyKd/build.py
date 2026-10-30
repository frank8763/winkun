from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

ROOT = Path.home()
SOURCE_DIR = Path(sys.argv[1])
JOB = Path(sys.argv[2])

SOURCE = SOURCE_DIR / "evaluate.py"
SETUP = SOURCE_DIR / "t58_setup.json"

MODELS = {
    "T53": {
        "path": (
            ROOT /
            "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
            "/checkpoints/episode_0010.pt"
        ),
        "sha": (
            "8d4ad4008a7647dbe96ad67b66f0a22"
            "c0d51c035744d25259560d1e936e4e8e0"
        ),
        "frames": 22491,
    },
    "T59": {
        "path": (
            ROOT /
            "mkmini_stage_td3_t59_wheel_clearance_20261027_v1"
            "/checkpoints/episode_0012.pt"
        ),
        "sha": (
            "d04fb9645bde46ffc5e06b10175406d6"
            "6111edc0fc97fc3fb4af254e4e5a4f65"
        ),
        "frames": 30491,
    },
}

CORE_SHA = (
    "35f97fa8f9a88c0807339801691b4e01"
    "f06f58967d9520fb4f136a6a09c79a07"
)

TASKS = [(2029654802, 18.8, 3.0)]


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


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


TRACE = '''
if frame % 200 == 0 or collision:
    import mujoco as _t61_mj
    import numpy as _t61_np

    _t61_model = env.model
    _t61_data = env.data

    if "_t61_obstacles" not in trace:
        trace["_t61_obstacles"] = tuple(
            i for i in range(
                int(_t61_model.ngeom)
            )
            if (
                _t61_mj.mj_id2name(
                    _t61_model,
                    _t61_mj.mjtObj.mjOBJ_GEOM,
                    i
                ) or ""
            ).startswith("real_rect_obs_")
        )

        if not trace["_t61_obstacles"]:
            raise RuntimeError(
                "T61_NO_RECTANGULAR_OBSTACLES"
            )

    _t61_buffer = _t61_np.zeros(
        6,
        dtype=_t61_np.float64
    )

    def _t61_nearest(gid):
        result_distance = 0.5
        result_obstacle = None

        for obstacle in trace["_t61_obstacles"]:
            d = float(
                _t61_mj.mj_geomDistance(
                    _t61_model,
                    _t61_data,
                    gid,
                    obstacle,
                    0.5,
                    _t61_buffer
                )
            )

            if not _t61_np.isfinite(d):
                raise RuntimeError(
                    "T61_NONFINITE_DISTANCE"
                )

            if d < result_distance:
                result_distance = d
                result_obstacle = obstacle

        return (
            round(result_distance, 6),
            result_obstacle
        )

    _t61_results = {
        "left_front": _t61_nearest(25),
        "right_front": _t61_nearest(27),
        "left_rear": _t61_nearest(29),
        "right_rear": _t61_nearest(31),
        "chassis": _t61_nearest(23),
    }

    _t61_right_obs3 = float(
        _t61_mj.mj_geomDistance(
            _t61_model,
            _t61_data,
            27,
            4,
            0.5,
            _t61_buffer
        )
    )

    _t61_joints = {}

    for _t61_joint in (
        "left_front_steer_joint",
        "right_front_steer_joint"
    ):
        jid = _t61_mj.mj_name2id(
            _t61_model,
            _t61_mj.mjtObj.mjOBJ_JOINT,
            _t61_joint
        )

        if jid >= 0:
            adr = int(
                _t61_model.jnt_qposadr[jid]
            )

            _t61_joints[_t61_joint] = float(
                _t61_data.qpos[adr]
            )

    print(
        "T61_RISK",
        json.dumps({
            "frame": frame,
            "collision": collision,
            "remaining_m": distance,
            "action": (
                action_data.tolist()
                if action_data is not None
                else None
            ),
            "right_front_to_obs3_m":
                _t61_right_obs3,
            "nearest_geometry": _t61_results,
            "steering_joints": _t61_joints
        }),
        flush=True
    )
'''


def inject(source):
    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_t57_capture"
        )
    ]

    check(
        len(functions) == 1,
        "CAPTURE_FUNCTION_INVALID"
    )

    anchors = []

    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if (
            isinstance(target, ast.Name)
            and target.id == "frame"
        ):
            anchors.append(node)

    check(
        len(anchors) == 1,
        "FRAME_ANCHOR_INVALID"
    )

    node = anchors[0]
    lines = source.splitlines(keepends=True)

    original = lines[node.lineno - 1]

    indent = original[
        :len(original) - len(original.lstrip())
    ]

    block = "".join(
        indent + line + "\n"
        if line.strip()
        else "\n"
        for line in TRACE.strip().splitlines()
    )

    lines.insert(node.end_lineno, block)

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count('"T61_RISK"') == 1,
        "TRACE_NOT_UNIQUE"
    )

    return result


def main():
    print("T61_STAGE_ONLY=TRUE")
    print("T61_TRAINING=FALSE")
    print("T61_PLANNER=FALSE")
    print("T61_ACTION_OVERRIDE=FALSE")
    print("T61_ENV_MODIFICATION=FALSE")
    print("T61_MODEL_WRITE=FALSE")

    check(
        SOURCE.is_file() and SETUP.is_file(),
        "T58_EVALUATOR_MISSING"
    )

    meta = json.loads(SETUP.read_text())

    check(
        sha(SOURCE) == meta["generated_sha256"],
        "T58_EVALUATOR_CHANGED"
    )

    import torch

    for name, info in MODELS.items():
        ckpt = info["path"]

        check(
            ckpt.is_file()
            and sha(ckpt) == info["sha"],
            "CHECKPOINT_CHANGED:" + name
        )

        state = torch.load(
            ckpt,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3"
            and int(state["frames"])
                == info["frames"]
            and state["manifest"]["source_core"]
                == CORE_SHA,
            "CHECKPOINT_METADATA_INVALID:"
            + name
        )

        print("T61_MODEL_VERIFIED=", name)

    model_code = ["MODELS = {"]

    for name, info in MODELS.items():
        model_code.extend([
            f"    {name!r}: {{",
            f"        'path': Path({str(info['path'])!r}),",
            f"        'sha': {info['sha']!r},",
            f"        'frames': {info['frames']},",
            f"        'core_sha': {CORE_SHA!r},",
            "    },",
        ])

    model_code.append("}")

    source = patch_assignments(
        SOURCE.read_text(),
        {
            "JOB": f"JOB = Path({str(JOB)!r})",
            "MODELS": "\n".join(model_code),
            "TASKS": f"TASKS = {TASKS!r}",
            "WALL": "WALL = 3000",
        }
    )

    source = source.replace(
        "T55", "T59"
    )

    source = inject(source)

    output = JOB / "evaluate.py"

    check(
        not output.exists(),
        "OUTPUT_ALREADY_EXISTS"
    )

    output.write_text(source)

    (JOB / "t61_setup.json").write_text(
        json.dumps({
            "experiment": "STAGE_TD3_T61",
            "source_sha256": sha(SOURCE),
            "evaluator_sha256": sha(output),
            "models": {
                n: x["sha"]
                for n, x in MODELS.items()
            },
            "tasks": TASKS,
            "automatic_promotion": False,
        }, indent=2) + "\n"
    )

    print("T61_EVALUATOR_SHA256=", sha(output))
    print("T61_RUNS=2")
    print("T61_READY=TRUE")


try:
    main()

except Exception as error:
    print("T61_BUILD=BLOCKED", repr(error))
    traceback.print_exc()
    raise

finally:
    print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
