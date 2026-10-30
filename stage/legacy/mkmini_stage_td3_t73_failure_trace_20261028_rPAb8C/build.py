from pathlib import Path
import ast
import hashlib
import os

JOB = Path(os.environ["T73_JOB"])
T71 = Path(os.environ["T71_JOB"])

SOURCE = T71 / "T71B_pilot4/evaluate.py"
STATUS = T71 / "T71B/status.json"

import json

def check(value, reason):
    if not value:
        raise RuntimeError(reason)

def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()

check(SOURCE.is_file(), "EVALUATOR_MISSING")

status = json.loads(STATUS.read_text())

ck = (
    T71 / "T71B/checkpoints"
    / status["checkpoint"]
)

check(
    status["state"] == "PILOT_COMPLETED_NONCOLLAPSED"
    and int(status["frames"]) == 190491
    and ck.is_file()
    and sha(ck) == status["checkpoint_sha256"],
    "T71B_CHECKPOINT_INVALID"
)

source = SOURCE.read_text()
ast.parse(source)

def replace_top(code, name, replacement):
    tree = ast.parse(code)

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
    lines = code.splitlines(keepends=True)

    lines[n.lineno-1:n.end_lineno] = [
        name + " = " + replacement + "\n"
    ]

    return "".join(lines)

source = replace_top(
    source,
    "JOB",
    f"Path({str(JOB)!r})"
)

# 只复现两个已确认失败的场景。
source = replace_top(
    source,
    "TASKS",
    repr([
        (2029654802, 18.8, 3.0),
        (2029654801, -12.5, 3.0)
    ])
)

HELPER = r'''
import json as _t73_json
import math as _t73_math
from pathlib import Path as _T73Path

_T73_TRACE = _T73Path(__T73_TRACE_PATH__)

def _t73_record(
    name, angle, env, obs, raw, action,
    frame, remaining, info, goal
):
    if name != "T71B":
        return

    if (
        frame != 1
        and frame % 100 != 0
        and not info.get("collision", False)
        and not info.get("reach_goal", False)
    ):
        return

    import mujoco as mj
    import numpy as np

    model = env.model
    data = env.data

    # 优先通过自由关节定位机器人主体。
    bodies = [
        int(model.jnt_bodyid[i])
        for i in range(model.njnt)
        if int(model.jnt_type[i])
            == int(mj.mjtJoint.mjJNT_FREE)
    ]

    if not bodies:
        bodies = [
            i for i in range(1, model.nbody)
            if any(
                s in (
                    mj.mj_id2name(
                        model,
                        mj.mjtObj.mjOBJ_BODY,
                        i
                    ) or ""
                ).lower()
                for s in ("base", "chassis", "robot")
            )
        ]

    body = bodies[0] if bodies else None

    position = None
    yaw = None
    body_name = None

    if body is not None:
        position = [
            float(x)
            for x in data.xpos[body][:3]
        ]

        matrix = np.asarray(
            data.xmat[body]
        ).reshape(3, 3)

        # 这是刚体局部X轴朝向，不预设它就是车头。
        yaw = float(
            _t73_math.atan2(
                matrix[1, 0],
                matrix[0, 0]
            )
        )

        body_name = mj.mj_id2name(
            model,
            mj.mjtObj.mjOBJ_BODY,
            body
        )

    goal_xy = None
    goal_direction_error = None

    if goal is not None:
        try:
            g = np.asarray(
                goal, dtype=float
            ).reshape(-1)

            if g.size >= 2:
                goal_xy = [
                    float(g[0]),
                    float(g[1])
                ]
        except (TypeError, ValueError):
            pass

    if (
        goal_xy is not None
        and position is not None
        and yaw is not None
    ):
        bearing = _t73_math.atan2(
            goal_xy[1] - position[1],
            goal_xy[0] - position[0]
        )

        error = _t73_math.atan2(
            _t73_math.sin(bearing-yaw),
            _t73_math.cos(bearing-yaw)
        )

        goal_direction_error = float(
            _t73_math.degrees(error)
        )

    joints = {}
    joint_names = []

    for i in range(model.njnt):
        jname = mj.mj_id2name(
            model,
            mj.mjtObj.mjOBJ_JOINT,
            i
        )

        if jname:
            joint_names.append(jname)

        if not jname:
            continue

        lower = jname.lower()

        if (
            "steer" in lower
            or "turn" in lower
            or "pivot" in lower
        ):
            address = int(
                model.jnt_qposadr[i]
            )

            joints[jname] = float(
                data.qpos[address]
            )

    sensor_names = [
        mj.mj_id2name(
            model,
            mj.mjtObj.mjOBJ_SENSOR,
            i
        )
        for i in range(model.nsensor)
    ]

    if not getattr(
        _t73_record, "_announced", False
    ):
        _t73_record._announced = True

        print(
            "T73_NATIVE_BODY=",
            body_name,
            flush=True
        )
        print(
            "T73_NATIVE_JOINTS=",
            _t73_json.dumps(joint_names),
            flush=True
        )
        print(
            "T73_NATIVE_SENSORS=",
            _t73_json.dumps(sensor_names),
            flush=True
        )

    raw_obs = np.asarray(
        obs, dtype=float
    ).reshape(-1)

    lidar_observation = (
        float(np.min(raw_obs[8:40]))
        if raw_obs.size >= 40
        else None
    )

    contacts = []

    if info.get("collision", False):
        for k in range(min(
            int(data.ncon), 16
        )):
            c = data.contact[k]

            contacts.append({
                "geom1": mj.mj_id2name(
                    model,
                    mj.mjtObj.mjOBJ_GEOM,
                    int(c.geom1)
                ),
                "geom2": mj.mj_id2name(
                    model,
                    mj.mjtObj.mjOBJ_GEOM,
                    int(c.geom2)
                ),
                "distance": float(c.dist)
            })

    row = {
        "model": name,
        "angle_deg": float(angle),
        "frame": int(frame),
        "remaining_m": float(remaining),
        "goal_xy": goal_xy,
        "body_name": body_name,
        "position_xyz": position,
        "body_x_axis_yaw_deg": (
            _t73_math.degrees(yaw)
            if yaw is not None
            else None
        ),
        "goal_error_body_x_deg":
            goal_direction_error,
        "raw_action": [
            float(v) for v in raw
        ],
        "actual_env_action": [
            float(v) for v in action
        ],
        "mujoco_ctrl": [
            float(v) for v in data.ctrl
        ],
        "steering_joints": joints,
        "lidar_observation_min":
            lidar_observation,
        "sensordata_first_16": [
            float(v)
            for v in data.sensordata[:16]
        ],
        "collision": bool(
            info.get("collision", False)
        ),
        "reach_goal": bool(
            info.get("reach_goal", False)
        ),
        "contacts": contacts
    }

    with _T73_TRACE.open(
        "a", encoding="utf-8"
    ) as f:
        f.write(
            _t73_json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )
'''

HELPER = HELPER.replace(
    "__T73_TRACE_PATH__",
    repr(str(JOB / "trajectory.jsonl"))
)

source = HELPER + "\n" + source

# 在原有评估循环中，最短距离更新完成后记录状态。
# 不触碰Actor、动作映射和MuJoCo步进调用。
tree = ast.parse(source)

matches = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue

    if not any(
        isinstance(t, ast.Name)
        and t.id == "closest"
        for t in node.targets
    ):
        continue

    value = node.value

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "min"
    ):
        matches.append(node)

check(
    len(matches) == 1,
    "CLOSEST_DISTANCE_ANCHOR_INVALID"
)

node = matches[0]
lines = source.splitlines(keepends=True)

indent = (
    lines[node.lineno-1][
        :len(lines[node.lineno-1])
        - len(lines[node.lineno-1].lstrip())
    ]
)

insertion = (
    indent
    + "_t73_record("
    + "name, angle, env, obs, raw, action, "
    + "frame, remaining, info, "
    + "locals().get('goal', None))\n"
)

lines.insert(node.end_lineno, insertion)

source = "".join(lines)

ast.parse(source)

script = JOB / "evaluate_trace.py"
script.write_text(source)

print("T73_CHECKPOINT_SHA256=", sha(ck))
print("T73_ORIGINAL_EVALUATOR_SHA256=", sha(SOURCE))
print("T73_TRACE_SCRIPT_SHA256=", sha(script))
print("T73_SOURCE_BUILD=PASS")
print("ORIGINAL_MODEL_MODIFIED=FALSE")
print("ORIGINAL_EVALUATOR_MODIFIED=FALSE")
print("ACTION_OVERRIDE=FALSE")
print("PLANNER=FALSE")
print("INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE")
