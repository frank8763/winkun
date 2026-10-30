from pathlib import Path
import ast
import hashlib
import json
import sys
import traceback

H = Path.home()
JOB = Path(sys.argv[1])

SOURCE_DIR = (
    H / "mkmini_stage_td3_t56_t55_vs_t53_20261027_jnqylp"
)

SOURCE = SOURCE_DIR / "evaluate.py"
SOURCE_META = SOURCE_DIR / "t56_setup.json"

PARENTS = {
    "T53": (
        H / "mkmini_stage_td3_t53_actor_anchor_20261027_v1"
        / "checkpoints/episode_0010.pt",
        "8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0"
    ),
    "T55": (
        H / "mkmini_stage_td3_t55_neargoal_reward_20261027_v1"
        / "checkpoints/episode_0015.pt",
        "e093967ce1191e0ca1a3d72e15a20338413e9e4d308d90ba99ad0fccb133bbac"
    ),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


TRACE_CODE = r'''
_T57_ENV = None
_T57_TRACE = {}


def _t57_capture(step_fn, env, *args, **kwargs):
    import json
    import numpy as np

    global _T57_ENV, _T57_TRACE

    if _T57_ENV is not env:
        _T57_ENV = env
        _T57_TRACE = {
            "frames": 0,
            "prev_action": None,
            "delta_count": 0,
            "steer_delta_sum": 0.0,
            "steer_delta_max": 0.0,
            "throttle_delta_sum": 0.0,
            "throttle_delta_max": 0.0,
            "last_distance": None,
        }

        print(
            "T57_NEW_MUJOCO_RUN=TRUE",
            flush=True
        )

    result = step_fn(env, *args, **kwargs)

    trace = _T57_TRACE
    trace["frames"] += 1

    action = (
        args[0]
        if args
        else kwargs.get("action")
    )

    action_data = None

    try:
        if action is not None:
            action_data = np.asarray(
                action,
                dtype=float
            ).reshape(-1)

            if action_data.shape != (2,):
                action_data = None

    except Exception:
        pass

    if action_data is not None:
        old = trace["prev_action"]

        if old is not None:
            dt = abs(
                float(action_data[0]) - old[0]
            )

            ds = abs(
                float(action_data[1]) - old[1]
            )

            trace["delta_count"] += 1

            trace["throttle_delta_sum"] += dt
            trace["steer_delta_sum"] += ds

            trace["throttle_delta_max"] = max(
                trace["throttle_delta_max"], dt
            )

            trace["steer_delta_max"] = max(
                trace["steer_delta_max"], ds
            )

        trace["prev_action"] = (
            float(action_data[0]),
            float(action_data[1])
        )

    info = {}

    if (
        isinstance(result, (tuple, list))
        and len(result) >= 5
        and isinstance(result[4], dict)
    ):
        info = result[4]

    collision = bool(
        info.get("collision", False)
    )

    success = bool(
        info.get("reach_goal", False)
    )

    distance = None

    try:
        distance = float(
            env._goal_state()[1]
        )
        trace["last_distance"] = distance

    except Exception:
        pass

    frame = trace["frames"]

    if (
        frame % 1000 == 0
        or collision
        or success
    ):
        n = max(1, trace["delta_count"])

        row = {
            "frame": frame,
            "distance_m": distance,
            "collision": collision,
            "success": success,
            "action": (
                action_data.tolist()
                if action_data is not None
                else None
            ),
            "mean_abs_delta_steer":
                trace["steer_delta_sum"] / n,
            "max_abs_delta_steer":
                trace["steer_delta_max"],
            "mean_abs_delta_throttle":
                trace["throttle_delta_sum"] / n,
            "max_abs_delta_throttle":
                trace["throttle_delta_max"],
        }

        try:
            row["qpos_first3"] = (
                env.data.qpos[:3].tolist()
            )
            row["qvel_first3"] = (
                env.data.qvel[:3].tolist()
            )

        except Exception:
            pass

        print(
            "T57_CONTROL_TRACE",
            json.dumps(
                row,
                ensure_ascii=False
            ),
            flush=True
        )

    if collision:
        contacts = []

        try:
            import mujoco

            model = env.model
            data = env.data

            for index in range(
                min(int(data.ncon), 80)
            ):
                contact = data.contact[index]

                g1 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1)
                )

                g2 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom2)
                )

                contacts.append({
                    "geom1": g1,
                    "geom2": g2,
                    "distance": float(
                        contact.dist
                    ),
                    "position": (
                        contact.pos.tolist()
                    )
                })

            contacts.sort(
                key=lambda x: x["distance"]
            )

            print(
                "T57_MUJOCO_CONTACT_COUNT=",
                int(data.ncon),
                flush=True
            )

        except Exception as error:
            print(
                "T57_CONTACT_INSPECTION_ERROR=",
                repr(error),
                flush=True
            )

        print(
            "T57_COLLISION_CONTACTS",
            json.dumps(
                contacts[:20],
                ensure_ascii=False
            ),
            flush=True
        )

        if not contacts:
            print(
                "T57_CONTACT_GEOMETRY_UNRESOLVED=TRUE",
                flush=True
            )

    return result
'''


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
                    "DUPLICATE_ASSIGNMENT:"
                    + target.id
                )
                found[target.id] = node

    check(
        set(found) == set(changes),
        "ASSIGNMENT_ANCHOR_INVALID"
    )

    lines = source.splitlines(keepends=True)

    for name, node in sorted(
        found.items(),
        key=lambda item: item[1].lineno,
        reverse=True
    ):
        lines[
            node.lineno - 1:node.end_lineno
        ] = [changes[name] + "\n"]

    result = "".join(lines)
    ast.parse(result)
    return result


def inject_trace(source):
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step_checked"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "N"
        )
    ]

    print(
        "T57_STEP_ANCHOR_COUNT=",
        len(calls)
    )

    check(
        len(calls) == 1,
        "STEP_CALL_ANCHOR_NOT_UNIQUE"
    )

    node = calls[0]
    lines = source.splitlines(keepends=True)

    start = (
        sum(
            len(x)
            for x in lines[:node.lineno - 1]
        )
        + node.col_offset
    )

    end = (
        sum(
            len(x)
            for x in lines[:node.end_lineno - 1]
        )
        + node.end_col_offset
    )

    original = source[start:end]

    check(
        original.startswith("N.step_checked(")
        and original.endswith(")"),
        "STEP_CALL_FORMAT_UNEXPECTED"
    )

    replacement = (
        "_t57_capture("
        + original[:]
        + ")"
    )

    result = (
        source[:start]
        + replacement
        + source[end:]
    )

    ast.parse(result)

    lines = result.splitlines(keepends=True)
    tree = ast.parse(result)

    mains = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "main"
        )
    ]

    check(
        len(mains) == 1,
        "MAIN_FUNCTION_NOT_UNIQUE"
    )

    position = mains[0].lineno - 1

    lines.insert(
        position,
        "\n" + TRACE_CODE + "\n"
    )

    result = "".join(lines)

    ast.parse(result)

    check(
        result.count(
            "T57_COLLISION_CONTACTS"
        ) == 1,
        "TRACE_PATCH_FAILED"
    )

    return result


def main():
    print("T57_STAGE_ONLY=TRUE")
    print("T57_TRAINING=FALSE")
    print("T57_PLANNER=FALSE")
    print("T57_ACTION_OVERRIDE=FALSE")
    print("T57_ENV_MODIFICATION=FALSE")

    check(
        SOURCE.is_file()
        and SOURCE_META.is_file(),
        "T56_VERIFIED_SOURCE_MISSING"
    )

    meta = json.loads(
        SOURCE_META.read_text()
    )

    check(
        sha(SOURCE)
        == meta["generated_evaluator_sha256"],
        "T56_EVALUATOR_CHANGED"
    )

    for name, (path, digest) in PARENTS.items():
        check(
            path.is_file()
            and sha(path) == digest,
            "CHECKPOINT_CHANGED:" + name
        )

        print(
            "T57_MODEL_VERIFIED=",
            name
        )

    original = SOURCE.read_text()

    result = patch_assignments(
        original,
        {
            "JOB": (
                f"JOB = Path({str(JOB)!r})"
            ),
            "WALL": "WALL = 5500",
        }
    )

    result = result.replace(
        "T56", "T57"
    )

    result = inject_trace(result)

    target = JOB / "evaluate.py"

    check(
        not target.exists(),
        "EVALUATOR_ALREADY_EXISTS"
    )

    target.write_text(result)

    information = {
        "experiment": "STAGE_TD3_T57",
        "source_evaluator_sha256": sha(SOURCE),
        "generated_evaluator_sha256": sha(target),
        "trace": "MUJOCO_CONTACT_AND_ACTION",
        "models": list(PARENTS),
        "training": False,
        "automatic_promotion": False,
    }

    # 不创建评估器的manifest.json
    (JOB / "t57_setup.json").write_text(
        json.dumps(
            information,
            indent=2
        ) + "\n"
    )

    print(
        "T57_EVALUATOR_SHA256=",
        sha(target)
    )

    print("T57_TRACE_READY=TRUE")


try:
    main()

except Exception as error:
    print(
        "T57_PREPARE=BLOCKED",
        repr(error)
    )
    traceback.print_exc()
    raise

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
