from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

H = Path.home()
JOB = Path('/home/linghu/mkmini_stage_td3_t61_frontwheel_risk_20261027_JNnyKd')

T16 = H / "mkmini_stage_td3_t16_long_finetune_20261027_v1"
T31 = H / "mkmini_stage_td3_t31_lidar_risk_20261027_v1"

CORE = T16 / "td3_core_goal_contrast.py"

HELPER = (
    H / "mkmini_stage_td3_independent_20261027_105202_577031_v1"
    / "td3_smoke.py"
)

STAGE = (
    H / "mkmini_stage_c147_fixedgoal_sac_20261027_v1"
    / "GOAL_COMMIT_24000/train/train.py"
)

FROZEN = (
    H / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
)

SOURCES = {
    CORE:
        "58b3752da0451a0b8eef8f5557401e9ccb4813cae8e30191c151f12b4a542d68",
    HELPER:
        "6f01a39f90bd071974701eac417b6801252af2ee99582ce9247d717d742bfc8d",
    STAGE:
        "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    FROZEN:
        "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

MODELS = {
    'T53': {
        'path': Path('/home/linghu/mkmini_stage_td3_t53_actor_anchor_20261027_v1/checkpoints/episode_0010.pt'),
        'sha': '8d4ad4008a7647dbe96ad67b66f0a22c0d51c035744d25259560d1e936e4e8e0',
        'frames': 22491,
        'core_sha': '35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07',
    },
    'T59': {
        'path': Path('/home/linghu/mkmini_stage_td3_t59_wheel_clearance_20261027_v1/checkpoints/episode_0012.pt'),
        'sha': 'd04fb9645bde46ffc5e06b10175406d66111edc0fc97fc3fb4af254e4e5a4f65',
        'frames': 30491,
        'core_sha': '35f97fa8f9a88c0807339801691b4e01f06f58967d9520fb4f136a6a09c79a07',
    },
}

SEEDS = (2029653601, 2029653602)

TASKS = [(2029654802, 18.8, 3.0)]

CAP = 14000
WALL = 3000


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)

    return h.hexdigest()


def check(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def emit(tag, data):
    print(
        tag,
        json.dumps(data, ensure_ascii=False),
        flush=True
    )


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )

    check(
        spec is not None and spec.loader is not None,
        "MODULE_IMPORT_INVALID:" + name
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def atomic_json(path, data):
    tmp = Path(str(path) + ".partial")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ) + "\n"
    )

    os.replace(tmp, path)


def prepare():
    print("STAGE_ONLY=TRUE")
    print("TRAINING=FALSE")
    print("PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("ACTION_OVERRIDE=FALSE")
    print("MODEL_PROMOTION=FALSE")

    for path, expected in SOURCES.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_INTEGRITY_FAILED:" + str(path)
        )

    core = load(CORE, "t32_stage_core")
    torch.set_num_threads(2)

    actors = {}

    for name, config in MODELS.items():
        path = config["path"]

        check(
            path.is_file() and sha(path) == config["sha"],
            "MODEL_SHA_FAILED:" + name
        )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        check(
            state.get("algorithm") == "TD3"
            and int(state["frames"]) == config["frames"],
            "MODEL_METADATA_FAILED:" + name
        )

        check(
            state["manifest"]["source_core"]
            == config["core_sha"],
            "MODEL_CORE_PROVENANCE_FAILED:" + name
        )

        actor = core.Actor(112, 2)
        actor.load_state_dict(
            state["actor"],
            strict=True
        )
        actor.eval()

        actors[name] = actor

        print(
            "T57_MODEL_VERIFIED=",
            name,
            "FRAMES=",
            config["frames"],
            flush=True
        )

    manifest = {
        "experiment": "STAGE_TD3_T57",
        "models": {
            name: item["sha"]
            for name, item in MODELS.items()
        },
        "tasks": TASKS,
        "physics_cap": CAP,
        "training": False,
        "noise": 0,
        "automatic_promotion": False
    }

    manifest_path = JOB / "manifest.json"

    if manifest_path.exists():
        check(
            json.loads(manifest_path.read_text())
            == manifest,
            "T57_MANIFEST_CHANGED"
        )
    else:
        atomic_json(manifest_path, manifest)

    os.environ["C98_JOB"] = str(JOB)
    os.environ["TD3_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    helper = load(
        HELPER,
        "t32_stage_helper"
    )

    legacy = load(
        STAGE,
        "t32_stage_legacy"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file()
            and legacy.sha(path) == expected,
            "NATIVE_SOURCE_CHANGED:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC,
        "t32_stage_native"
    )

    _, v, native_mod, prep, _ = runner.load_native()

    check(
        len(prep) >= 4,
        "NATIVE_BOOTSTRAP_INVALID"
    )

    N = prep[3]

    make_base = legacy.make_short_task(
        runner.make_task
    )

    print("T57_PREFLIGHT=PASS", flush=True)

    return actors, helper, make_base, v, native_mod, N


def run_case(
    seed,
    angle,
    distance,
    name,
    actor,
    helper,
    make_base,
    v,
    native_mod,
    N,
    deadline
):
    env = None
    renderer = None

    try:
        env, renderer, obs, actual_seed = (
            helper.make_goal_task(
                make_base,
                v,
                native_mod,
                N,
                seed,
                angle,
                distance
            )
        )

        env.max_steps = CAP

        obs = np.asarray(
            obs,
            dtype=np.float32
        ).reshape(-1)

        check(
            obs.shape == (112,)
            and np.isfinite(obs).all(),
            "INITIAL_OBSERVATION_INVALID"
        )

        start = np.asarray(
            env.data.qpos[:2],
            dtype=float
        ).tolist()

        goal = np.asarray(
            env.goal_xy,
            dtype=float
        ).tolist()

        initial = float(
            env._goal_state()[1]
        )

        check(
            abs(initial - distance) < 0.04,
            "ACTUAL_TASK_DISTANCE_INVALID"
        )

        closest = initial
        remaining = initial

        success = False
        collision = False
        reason = "CAP"

        danger_frames = 0
        danger_throttle_sum = 0.0

        for frame in range(1, CAP + 1):
            if time.monotonic() >= deadline:
                return None

            with torch.no_grad():
                raw = actor(
                    torch.from_numpy(
                        obs.reshape(1, 112)
                    )
                ).cpu().numpy()[0]

            check(
                raw.shape == (2,)
                and np.isfinite(raw).all(),
                "ACTOR_OUTPUT_INVALID"
            )

            throttle = float(
                np.clip(
                    (float(raw[0]) + 1.0) / 2.0,
                    0.0,
                    1.0
                )
            )

            steer = float(
                np.clip(
                    raw[1],
                    -1.0,
                    1.0
                )
            )

            # 仅统计，不干预控制。
            forward_lidar = float(
                np.min(obs[8:40])
            )

            if forward_lidar < 0.10:
                danger_frames += 1
                danger_throttle_sum += throttle

            action = np.asarray(
                [throttle, steer],
                dtype=np.float32
            )

            next_obs, _, terminated, truncated, info = (
                _t57_capture(N.step_checked, env, action)
            )

            obs = np.asarray(
                next_obs,
                dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "STEP_OBSERVATION_INVALID"
            )

            remaining = float(
                env._goal_state()[1]
            )

            closest = min(
                closest,
                remaining
            )

            success = bool(
                info.get("reach_goal", False)
            )

            collision = bool(
                info.get("collision", False)
            )

            if frame % 3500 == 0:
                emit("T57_PROGRESS", {
                    "model": name,
                    "seed": seed,
                    "angle": angle,
                    "frame": frame,
                    "remaining_m": round(
                        remaining, 4
                    )
                })

            if (
                success
                or collision
                or terminated
                or truncated
            ):
                reason = (
                    "GOAL" if success
                    else "CONTACT" if collision
                    else "TERMINATED" if terminated
                    else "TRUNCATED"
                )
                break

        return {
            "model": name,
            "checkpoint_sha256": MODELS[name]["sha"],
            "requested_seed": seed,
            "actual_seed": int(actual_seed),
            "angle_deg": angle,
            "distance_m": distance,
            "start_xy": start,
            "goal_xy": goal,
            "frames": frame,
            "closest_m": closest,
            "remaining_m": remaining,
            "success": success,
            "collision": collision,
            "end": reason,
            "danger_frames": danger_frames,
            "danger_mean_throttle": (
                danger_throttle_sum / danger_frames
                if danger_frames
                else None
            ),
            "training": False,
            "action_override": False
        }

    finally:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()




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
            def _t58_describe_geom(_gid):
                _bid = int(model.geom_bodyid[_gid])
                _original_body = _bid
                _chain = []
                _seen = set()

                while _bid not in _seen:
                    _seen.add(_bid)

                    _body_name = mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        _bid
                    )

                    _chain.append({
                        "body_id": _bid,
                        "body_name": _body_name,
                        "joint_count": int(
                            model.body_jntnum[_bid]
                        ),
                    })

                    if _bid == 0:
                        break

                    _next = int(
                        model.body_parentid[_bid]
                    )

                    if _next == _bid:
                        break

                    _bid = _next

                return {
                    "geom_id": _gid,
                    "geom_name": mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        _gid
                    ),
                    "body_id": _original_body,
                    "body_chain": _chain,
                    "geom_type": int(
                        model.geom_type[_gid]
                    ),
                    "geom_group": int(
                        model.geom_group[_gid]
                    ),
                    "contype": int(
                        model.geom_contype[_gid]
                    ),
                    "conaffinity": int(
                        model.geom_conaffinity[_gid]
                    ),
                    "geom_world_xyz": (
                        data.geom_xpos[_gid].tolist()
                    ),
                }


            _t58_details = []

            for _idx in range(int(data.ncon)):
                _contact = data.contact[_idx]

                _g1 = int(_contact.geom1)
                _g2 = int(_contact.geom2)

                _t58_details.append({
                    "contact_index": _idx,
                    "geom1": _t58_describe_geom(_g1),
                    "geom2": _t58_describe_geom(_g2),
                    "penetration_m": float(
                        _contact.dist
                    ),
                    "contact_world_xyz": (
                        _contact.pos.tolist()
                    ),
                })

            print(
                "T58_CONTACT_IDENTITIES",
                json.dumps(
                    _t58_details,
                    ensure_ascii=False
                ),
                flush=True
            )

            _t58_joints = []

            for _jid in range(int(model.njnt)):
                _jname = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    _jid
                )

                _lower = (_jname or "").lower()

                if any(
                    word in _lower
                    for word in (
                        "steer",
                        "front",
                        "ackermann",
                    )
                ):
                    _qa = int(
                        model.jnt_qposadr[_jid]
                    )
                    _va = int(
                        model.jnt_dofadr[_jid]
                    )

                    _t58_joints.append({
                        "joint_id": _jid,
                        "name": _jname,
                        "qpos": float(
                            data.qpos[_qa]
                        ),
                        "qvel": float(
                            data.qvel[_va]
                        ),
                    })

            print(
                "T58_STEERING_JOINTS",
                json.dumps(
                    _t58_joints,
                    ensure_ascii=False
                ),
                flush=True
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

def main():
    JOB.mkdir(
        parents=True,
        exist_ok=True
    )

    actors, helper, make_base, v, native_mod, N = (
        prepare()
    )

    deadline = time.monotonic() + WALL

    complete_pairs = []

    for index, (seed, angle, distance) in enumerate(
        TASKS, 1
    ):
        group = []

        for name, actor in actors.items():
            result_path = (
                JOB / f"task_{index:02d}_{name}.json"
            )

            if result_path.exists():
                result = json.loads(
                    result_path.read_text()
                )

                check(
                    result["model"] == name
                    and result["checkpoint_sha256"]
                    == MODELS[name]["sha"]
                    and result["requested_seed"] == seed
                    and result["angle_deg"] == angle
                    and result["distance_m"] == distance
                    and result["training"] is False,
                    "RESUME_CASE_IDENTITY_FAILED"
                )

                print(
                    "T57_RESUMED_CASE=",
                    index,
                    name
                )

            else:
                result = run_case(
                    seed,
                    angle,
                    distance,
                    name,
                    actor,
                    helper,
                    make_base,
                    v,
                    native_mod,
                    N,
                    deadline
                )

                if result is None:
                    print(
                        "T57_STATE=PAUSED_RESUMABLE"
                    )
                    return

                atomic_json(
                    result_path,
                    result
                )

                emit("T57_CASE", {
                    "task": index,
                    **result
                })

            group.append(result)

        # 严格配对：必须相同起点、目标及实际场景种子。
        baseline = group[0]

        for result in group[1:]:
            check(
                result["actual_seed"]
                == baseline["actual_seed"],
                "PAIR_SEED_MISMATCH"
            )

            check(
                np.allclose(
                    result["start_xy"],
                    baseline["start_xy"],
                    atol=1e-7,
                    rtol=0
                ),
                "PAIR_START_MISMATCH"
            )

            check(
                np.allclose(
                    result["goal_xy"],
                    baseline["goal_xy"],
                    atol=1e-7,
                    rtol=0
                ),
                "PAIR_GOAL_MISMATCH"
            )

        complete_pairs.append(group)

        emit("T57_PAIR", {
            "task": index,
            "seed": seed,
            "angle": angle,
            "results": {
                result["model"]: {
                    "end": result["end"],
                    "closest_m": round(
                        result["closest_m"], 4
                    ),
                    "frames": result["frames"]
                }
                for result in group
            }
        })

    for name in MODELS:
        rows = [
            result
            for group in complete_pairs
            for result in group
            if result["model"] == name
        ]

        emit("T57_MODEL", {
            "model": name,
            "runs": len(rows),
            "success": sum(
                r["success"] for r in rows
            ),
            "collisions": sum(
                r["collision"] for r in rows
            ),
            "left_success": sum(
                r["success"]
                for r in rows
                if r["angle_deg"] < 0
            ),
            "right_success": sum(
                r["success"]
                for r in rows
                if r["angle_deg"] > 0
            ),
            "mean_closest_m": round(
                sum(
                    r["closest_m"]
                    for r in rows
                ) / len(rows),
                5
            )
        })

    check(
        len(complete_pairs) == len(TASKS),
        "EVALUATION_INCOMPLETE"
    )

    check(
        sha(FROZEN) == SOURCES[FROZEN],
        "FROZEN_MODEL_CHANGED"
    )

    for name, config in MODELS.items():
        check(
            sha(config["path"]) == config["sha"],
            "MODEL_CHANGED_DURING_EVALUATION:" + name
        )

    print("T57_PAIR_GATE=PASS")
    print("T57_STATE=EVALUATED")
    print("MODEL_PROMOTED=FALSE")


try:
    main()

except Exception as error:
    print(
        "T57_BLOCKED=",
        repr(error)
    )
    traceback.print_exc()

finally:
    print(
        "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE"
    )
