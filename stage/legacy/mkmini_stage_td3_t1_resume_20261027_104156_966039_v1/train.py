DISTANCES_C81 = (1.0, 2.0, 3.0, 3.0, 1.5, 2.0, 3.0, 3.0)
from pathlib import Path
import os
import sys
import time
import json
import csv
import math
import gc
import hashlib
import fcntl
import traceback
import random
import importlib.util
import argparse

import numpy as np
import torch
import mujoco as mj

H = Path.home()
JOB = Path(os.environ["C98_JOB"])

SRC = H / (
    "mkmini_stage_g20f6c19_turn_learning_20261022_221621_577031_v1"
    "/stage_c13_B30/train_g20f6c13_global_navigation.py"
)

CK = H / (
    "mkmini_stage_g20f6c19_verified_relay_20261023_085122_577031_v4"
    "/B30/night/branch_D_combined20_20/out/"
    "full_scene_candidate_research_only.pt"
)

P2 = H / (
    "mkmini_stage_g19_p2_multiarm_20261018_222036_624927500"
    "/p2.py"
)

EXPECTED = {
    SRC: "26c29eddd2c677a42af5f9c08aa347609c8b0272fdbd7dc9fde22bf7d7c390a8",
    CK: "fe126fc0a9594a8f6ec6fed1842780c31db2acf92d40c7bd6a63292cec5e1484",
    P2: "38abc30c271f46a1d0822cc7cc580bf4b8d95ff27ba048ac908ff8fb0add3f54",
}

SECONDS = 900
MAX_EXTRA = 24000
CAP = int(os.environ.get("C22_CAP", "4600"))

TRAIN_ANGLES = (-30, 30, -25, 25, -20, 20, -15, 15, -10, 10, -5, 5, 0, -35, 35, -40, 40, -45, 45)
TESTS = (
    (2029041101, -30),
    (2029041102, 30),
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def output(name, obj):
    target = JOB / name
    tmp = target.with_name(target.name + ".partial")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    )
    os.replace(tmp, target)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    check(spec is not None and spec.loader is not None, "BAD_MODULE_SPEC")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def make_short_task(original):
    # Stage-only start/goal curriculum.
    # This function generates one final goal, never waypoints.
    def make(v, mod, N, seed, angle_deg, *,
             cleared=True, limit=CAP):

        env = None
        renderer = None

        try:
            env, _, _, _ = N.create(
                mod.PR, "reference", "medium", int(seed)
            )
            env.max_steps = int(limit)
            lidar_range_m = float(
                getattr(env, "lidar_max_range", float("nan"))
            )
            check(
                math.isfinite(lidar_range_m)
                and 1.0 <= lidar_range_m <= 100.0,
                "INVALID_LIDAR_MAX_RANGE"
            )
            print(
                "C22_LIDAR_MAX_RANGE_M=", lidar_range_m,
                flush=True
            )

            q0 = np.asarray(
                env.data.qpos, dtype=np.float64
            ).copy()
            v0 = np.asarray(
                env.data.qvel, dtype=np.float64
            ).copy()

            yaw = float(env._pose()[2])
            theta = yaw + math.radians(float(angle_deg))
            direction = np.asarray(
                [math.cos(theta), math.sin(theta)],
                dtype=np.float64
            )

            def set_pose(xy):
                env.data.qpos[:] = q0
                env.data.qpos[:2] = np.asarray(
                    xy, dtype=np.float64
                )
                env.data.qvel[:] = v0
                mj.mj_forward(env.model, env.data)

            def lidar_minimum():
                scan = np.asarray(
                    env._lidar48(), dtype=np.float64
                ).reshape(-1)

                if scan.shape != (48,):
                    raise RuntimeError("INVALID_LIDAR_SHAPE")

                if not np.isfinite(scan).all():
                    raise RuntimeError("INVALID_LIDAR_VALUES")

                return float(scan.min()) * lidar_range_m

            # Full geometry grid, including the previously verified
            # high-clearance area. No obstacles are modified.
            grid = np.arange(-4.0, 4.0001, 0.25)

            positions = [
                (float(x), float(y))
                for x in grid
                for y in grid
            ]

            rng = random.Random(
                int(seed) * 1009 + int(angle_deg) * 97
            )
            rng.shuffle(positions)

            stats = {
                "tested_starts": 0,
                "collision_free_starts": 0,
                "starts_ge_075m": 0,
                "tested_goals": 0,
                "collision_free_goals": 0,
                "accepted_pairs": 0,
                "max_start_lidar_m": 0.0,
            }

            accepted = []

            for xy in positions:
                start = np.asarray(xy, dtype=np.float64)

                set_pose(start)
                stats["tested_starts"] += 1

                if env._collision():
                    continue

                stats["collision_free_starts"] += 1

                sc = lidar_minimum()
                stats["max_start_lidar_m"] = max(
                    stats["max_start_lidar_m"], sc
                )

                if sc < 0.75:
                    continue

                stats["starts_ge_075m"] += 1

                for distance in ((DISTANCES_C81[(int(seed) - 2029368000) % 8] if 2029368000 <= int(seed) < 2029369000 else 3.0),):
                    if distance <= float(env.success_radius) + 0.35:
                        continue

                    goal = start + float(distance) * direction

                    set_pose(goal)
                    stats["tested_goals"] += 1

                    if env._collision():
                        continue

                    stats["collision_free_goals"] += 1

                    gc = lidar_minimum()

                    if gc < 0.35:
                        continue

                    accepted.append((
                        start.copy(),
                        goal.copy(),
                        float(sc),
                        float(gc),
                        float(distance)
                    ))

                    stats["accepted_pairs"] += 1
                    break

                if len(accepted) >= 8:
                    break

            print(
                "C22_TASK_SEARCH_STATS=",
                json.dumps({
                    "seed": int(seed),
                    "angle_deg": int(angle_deg),
                    **stats,
                }, ensure_ascii=False),
                flush=True
            )

            check(
                len(accepted) > 0,
                "NO_SAFE_SHORT_TASK_FOR_ANGLE_" + str(angle_deg)
            )

            selected = accepted[
                abs(int(seed)) % len(accepted)
            ]

            start, goal, sc, gc, distance = selected

            set_pose(start)

            check(
                not env._collision(),
                "FINAL_START_COLLISION"
            )

            check(
                lidar_minimum() >= 0.75,
                "FINAL_START_CLEARANCE_INVALID"
            )

            env.goal_xy = goal.copy()
            env.prev_goal_dist = float(
                env._goal_state()[1]
            )

            obs = np.asarray(
                env._observation(update_history=False),
                dtype=np.float32
            ).reshape(-1)

            lidar = np.asarray(
                env._lidar48(), dtype=np.float32
            ).reshape(-1)

            check(
                obs.shape == (112,)
                and np.isfinite(obs).all(),
                "INVALID_INITIAL_OBSERVATION"
            )

            check(
                np.allclose(
                    obs[:48], lidar,
                    atol=1e-5, rtol=0
                ),
                "INITIAL_LIDAR_NOT_SYNCHRONIZED"
            )

            actual_distance = float(
                np.linalg.norm(
                    np.asarray(env.goal_xy) -
                    np.asarray(env.data.qpos[:2])
                )
            )

            actual_bearing = math.degrees(
                float(env._goal_state()[2])
            )

            check(
                abs(actual_distance - distance) < 0.02,
                "GOAL_DISTANCE_MISMATCH"
            )

            check(
                abs(actual_bearing - float(angle_deg)) < 1.0,
                "GOAL_BEARING_MISMATCH"
            )

            renderer = mj.Renderer(
                env.model, height=120, width=160
            )
            depth = v.depth_frame(renderer, env)

            print(
                "C22_REAL_SHORT_TASK=",
                json.dumps({
                    "seed": int(seed),
                    "angle_deg": int(angle_deg),
                    "actual_bearing_deg": actual_bearing,
                    "start_xy": start.tolist(),
                    "goal_xy": goal.tolist(),
                    "distance_m": actual_distance,
                    "start_lidar_min_m": sc,
                    "goal_lidar_min_m": gc,
                    "obstacles_removed": 0,
                    "planner": False,
                    "waypoints": False,
                }, ensure_ascii=False),
                flush=True
            )

            return env, renderer, obs, depth, 0

        except Exception:
            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()
            raise


    def c81_checked_make(*args, **kwargs):
        result = make(*args, **kwargs)

        seed = int(
            args[3] if len(args) > 3
            else kwargs["seed"]
        )

        if 2029368000 <= seed < 2029369000:

            expected = DISTANCES_C81[
                (seed - 2029368000) % 8
            ]

            env = result[0]

            actual = float(np.linalg.norm(
                np.asarray(env.goal_xy)
                - np.asarray(env.data.qpos[:2])
            ))

            if abs(actual - expected) >= 0.02:

                try:
                    if result[1] is not None:
                        result[1].close()

                    env.close()

                finally:
                    raise RuntimeError(
                        "C81_REAL_DISTANCE_FAILED:"
                        + str((seed, actual, expected))
                    )

            print(
                "C81_REAL_TASK_GATE=PASS",
                "SEED=", seed,
                "ACTUAL_M=", round(actual, 6),
                "EXPECTED_M=", expected,
                flush=True
            )

        return result

    return c81_checked_make


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    JOB.mkdir(parents=True, exist_ok=True)

    lock = open(
        H / ".mkmini_stage_g20f6c22_single_instance.lock",
        "a+"
    )

    try:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError:
        print("TRAIN_NOT_STARTED=ANOTHER_C22_ACTIVE")
        return

    print("=" * 78)
    print("STAGE G20-F6C22 — SHORT-GOAL SAC TRAINING")
    print("BUDGET_SECONDS=", SECONDS)
    print("MAX_ADDITIONAL_FRAMES=", MAX_EXTRA)
    print("STAGE_ONLY=TRUE")
    print("EXTERNAL_PLANNER=FALSE")
    print("WAYPOINTS=FALSE")
    print("AUTOMATIC_MODEL_PROMOTION=FALSE")
    print("FROZEN_PARENT_WRITE=FALSE")
    print("=" * 78)

    for path, expected in EXPECTED.items():
        check(
            path.is_file() and sha(path) == expected,
            "SOURCE_HASH_FAILED_" + str(path)
        )
        print("SOURCE_VERIFIED=", path)

    runner = module(SRC, "stage_c22_B30_train")

    p2, v, native_mod, prep, hashes = runner.load_native()

    check(isinstance(hashes, dict), "SOURCE_GATE_NOT_DICT")

    sources = dict(hashes)
    sources[str(P2)] = sha(P2)

    C, N = prep[2], prep[3]

    work = JOB / "work_resume.pt"

    if not args.resume:
        raise RuntimeError("C64_ABORT: --resume is mandatory")
    if not work.is_file():
        raise RuntimeError("C64_ABORT: C62 resume checkpoint missing")

    state_path = (
        work
        if args.resume and work.is_file()
        else CK
    )

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False
    )

    check(
        state.get("source_hashes") == sources,
        "RESUME_SOURCE_MANIFEST_MISMATCH"
    )

    check(
        state.get("branch") == "depth_conservative",
        "RESUME_BRANCH_INVALID"
    )

    cfg = state["branch_config"]
    base_frames = int(state["frames"])
    expected_frames = int(os.environ["C64_EXPECTED_PARENT_FRAMES"])
    if base_frames != expected_frames:
        raise RuntimeError(
            f"C64_ABORT: expected {expected_frames} frames, "
            f"loaded {base_frames}"
        )

    del state
    gc.collect()

    v.TRAIN_COUNTER["frames"] = base_frames

    replay = C.Replay(v.MAXMEM)

    policy, episodes = p2.resume(
        v, native_mod, prep,
        "depth_conservative",
        cfg, replay,
        state_path, sources
    )
    print("C33_REPLAY_BEFORE=", len(replay.data), flush=True)
    if len(replay.data) < 96:
        raise RuntimeError("C98_REPLAY_RESTORE_TOO_SMALL")
    print("C98_REPLAY_RETAINED=",len(replay.data),flush=True)
    print("C33_REPLAY_MODE=PRESERVED", "AFTER=", len(replay.data), flush=True)
    _c122_cp = policy.agent.cp
    _c122_old_lr = [float(g['lr']) for g in _c122_cp.param_groups]
    if not _c122_old_lr or any(abs(v-3e-6)>1e-10 for v in _c122_old_lr):
        raise RuntimeError('C122_CRITIC_LR_BEFORE_UNEXPECTED:'+repr(_c122_old_lr))
    for _c122_group in _c122_cp.param_groups:
        _c122_group['lr'] = 3e-6
    print('C122_CRITIC_LR_APPLIED=PASS', 'BEFORE=', _c122_old_lr, 'AFTER=',
          [float(g['lr']) for g in _c122_cp.param_groups], flush=True)
    import random as _c33_random
    import numpy as _c33_np
    import torch as _c33_torch
    _c33_random.seed(20261033)
    _c33_np.random.seed(20261033)
    _c33_torch.manual_seed(20261033)
    if _c33_torch.cuda.is_available():
        _c33_torch.cuda.manual_seed_all(20261033)

    policy.update = (
        lambda batch: p2.sac_update(v, policy, batch)
    )

    # C122S0: conservative training-only optimizer adjustment.
    # Inference actions remain direct SAC actor outputs.
    _c109_actor_optimizer = policy.agent.ap
    _c109_vision_optimizer = policy.vp

    _c109_old_actor_lrs = [
        float(g["lr"])
        for g in _c109_actor_optimizer.param_groups
    ]

    _c109_old_vision_lrs = (
        [
            float(g["lr"])
            for g in _c109_vision_optimizer.param_groups
        ]
        if _c109_vision_optimizer is not None
        else []
    )

    if (
        not _c109_old_actor_lrs
        or any(
            lr <= 0
            for lr in (
                _c109_old_actor_lrs
                + _c109_old_vision_lrs
            )
        )
    ):
        raise RuntimeError(
            "C122S0_OPTIMIZER_LR_INVALID"
        )

    for group in _c109_actor_optimizer.param_groups:
        group["lr"] = float(group["lr"]) * 0.2

    if _c109_vision_optimizer is not None:
        for group in _c109_vision_optimizer.param_groups:
            group["lr"] = float(group["lr"]) * 0.2

    print(
        "C122S0_OPTIMIZER_LR=",
        {
            "actor_before":
                _c109_old_actor_lrs,

            "actor_after": [
                float(g["lr"])
                for g in _c109_actor_optimizer.param_groups
            ],

            "vision_before":
                _c109_old_vision_lrs,

            "vision_after": (
                [
                    float(g["lr"])
                    for g in _c109_vision_optimizer.param_groups
                ]
                if _c109_vision_optimizer is not None
                else []
            ),

            "critic_unchanged": [
                float(g["lr"])
                for g in policy.agent.cp.param_groups
            ]
        },
        flush=True
    )


    expected_updates = int(os.environ["C64_EXPECTED_PARENT_UPDATES"])
    if int(policy.agent.updates) != expected_updates:
        raise RuntimeError(
            f"C64_ABORT: expected {expected_updates} updates, "
            f"loaded {int(policy.agent.updates)}"
        )

    print(
        "RESUME_VERIFIED=TRUE",
        "FRAMES=", base_frames,
        "UPDATES=", int(policy.agent.updates),
        flush=True
    )

    # Modify only the in-memory training task and reward settings.
    runner.make_task = make_short_task(runner.make_task)

    _c135_original_task = runner.make_task

    def _c135_near_task(
        v, mod, N, seed, angle_deg, *,
        cleared=True, limit=CAP
    ):
        env, renderer, obs, depth, relocated = (
            _c135_original_task(
                v, mod, N, seed, angle_deg,
                cleared=cleared, limit=limit
            )
        )

        # 仅修改训练任务。原有验证任务保持不变。
        if not 2029368000 <= int(seed) < 2029369000:
            return env, renderer, obs, depth, relocated

        import mujoco as _c135_mj

        try:
            requested = float(
                os.environ["C55_TRAIN_GOAL_M"]
            )

            if requested not in (0.4, 0.8, 1.0, 1.2, 1.5):
                raise RuntimeError(
                    "C135_UNEXPECTED_TRAIN_DISTANCE:"
                    + repr(requested)
                )

            xy = np.asarray(
                env.data.qpos[:2],
                dtype=np.float64
            ).copy()

            yaw = float(env._pose()[2])

            heading = yaw + math.radians(
                float(angle_deg)
            )

            final_goal = (
                xy
                + requested * np.array([
                    math.cos(heading),
                    math.sin(heading)
                ])
            )

            q0 = np.asarray(
                env.data.qpos
            ).copy()

            v0 = np.asarray(
                env.data.qvel
            ).copy()

            ctrl0 = np.asarray(
                env.data.ctrl
            ).copy()

            try:
                env.data.qpos[:2] = final_goal

                _c135_mj.mj_forward(
                    env.model, env.data
                )

                collision_at_goal = bool(
                    env._collision()
                )

            finally:
                env.data.qpos[:] = q0
                env.data.qvel[:] = v0

                if env.data.ctrl.size:
                    env.data.ctrl[:] = ctrl0

                _c135_mj.mj_forward(
                    env.model, env.data
                )

            if collision_at_goal:
                raise RuntimeError(
                    "C135_FINAL_GOAL_COLLISION:"
                    + repr((seed, requested, angle_deg))
                )

            if env._collision():
                raise RuntimeError(
                    "C135_START_COLLISION_AFTER_RESTORE"
                )

            env.goal_xy = final_goal.copy()

            env.prev_goal_dist = float(
                env._goal_state()[1]
            )

            actual = float(
                env._goal_state()[1]
            )

            bearing = math.degrees(
                float(env._goal_state()[2])
            )

            if abs(actual - requested) > 0.002:
                raise RuntimeError(
                    "C135_REAL_DISTANCE_FAILED:"
                    + repr((requested, actual))
                )

            if abs(bearing - float(angle_deg)) > 1.0:
                raise RuntimeError(
                    "C135_GOAL_BEARING_FAILED:"
                    + repr((angle_deg, bearing))
                )

            obs = np.asarray(
                env._observation(
                    update_history=False
                ),
                dtype=np.float32
            ).reshape(-1)

            if (
                obs.size != 112
                or not np.isfinite(obs).all()
            ):
                raise RuntimeError(
                    "C135_INITIAL_OBSERVATION_INVALID"
                )

            print(
                "C135_REAL_NEAR_TASK=",
                json.dumps({
                    "seed": int(seed),
                    "requested_m": requested,
                    "actual_m": actual,
                    "angle_deg": bearing,
                    "start_xy": xy.tolist(),
                    "goal_xy": final_goal.tolist(),
                    "goal_collision": False,
                    "planner": False,
                    "waypoints": False
                }),
                flush=True
            )

            return (
                env, renderer,
                obs, depth, relocated
            )

        except Exception:
            renderer.close()
            env.close()
            raise


    _c141_actual_base_task = _c135_original_task

    def _c141_task_resample(
        v, mod, N, seed, angle_deg, *,
        cleared=True, limit=CAP
    ):
        import mujoco as _c141_mj
        import numpy as _c141_np
        import math as _c141_math
        import os as _c141_os

        wanted = float(
            _c141_os.environ["C55_TRAIN_GOAL_M"]
        )

        if wanted not in (0.4, 0.8, 1.0, 1.2, 1.5):
            raise RuntimeError(
                "C141_DISTANCE_NOT_ALLOWED:"
                + str(wanted)
            )

        for attempt in range(24):
            candidate_seed = (
                int(seed)
                if attempt == 0
                else int(seed) + 101 + attempt * 41
            )

            sample = _c141_actual_base_task(
                v, mod, N, candidate_seed, angle_deg,
                cleared=cleared, limit=limit
            )

            env, renderer, obs, depth, relocated = sample

            try:
                start = _c141_np.asarray(
                    env.data.qpos[:2],
                    dtype=_c141_np.float64
                ).copy()

                heading = (
                    float(env._pose()[2])
                    + _c141_math.radians(float(angle_deg))
                )

                goal = start + wanted * _c141_np.array([
                    _c141_math.cos(heading),
                    _c141_math.sin(heading)
                ])

                q0 = env.data.qpos.copy()
                v0 = env.data.qvel.copy()
                ctrl0 = env.data.ctrl.copy()

                try:
                    env.data.qpos[:2] = goal
                    _c141_mj.mj_forward(
                        env.model, env.data
                    )

                    bad = bool(env._collision())

                finally:
                    env.data.qpos[:] = q0
                    env.data.qvel[:] = v0

                    if env.data.ctrl.size:
                        env.data.ctrl[:] = ctrl0

                    _c141_mj.mj_forward(
                        env.model, env.data
                    )

                if env._collision():
                    raise RuntimeError(
                        "C141_RESTORED_START_COLLIDES"
                    )

            except Exception:
                try:
                    renderer.close()
                finally:
                    env.close()
                raise

            if not bad:
                # C146: commit the checked goal to the
                # actual Stage training environment.
                env.goal_xy = goal.copy()

                env.prev_goal_dist = float(
                    env._goal_state()[1]
                )

                actual = float(
                    env._goal_state()[1]
                )

                actual_angle = _c141_math.degrees(
                    float(env._goal_state()[2])
                )

                if abs(actual - wanted) > 0.02:
                    raise RuntimeError(
                        "C146_RESET_GOAL_DISTANCE_MISMATCH:"
                        + repr((wanted, actual))
                    )

                if abs(
                    actual_angle - float(angle_deg)
                ) > 2.0:
                    raise RuntimeError(
                        "C146_RESET_GOAL_ANGLE_MISMATCH:"
                        + repr((angle_deg, actual_angle))
                    )

                obs = _c141_np.asarray(
                    env._observation(
                        update_history=False
                    ),
                    dtype=_c141_np.float32
                ).reshape(-1)

                if (
                    obs.size != 112
                    or not _c141_np.isfinite(obs).all()
                ):
                    raise RuntimeError(
                        "C146_RESET_OBSERVATION_INVALID"
                    )

                sample = (
                    env, renderer, obs, depth, relocated
                )

                print(
                    "C146_GOAL_COMMITTED=",
                    json.dumps({
                        "seed": int(candidate_seed),
                        "requested_m": float(wanted),
                        "actual_m": actual,
                        "requested_angle": float(angle_deg),
                        "actual_angle": actual_angle,
                        "observation_size": int(obs.size)
                    }),
                    flush=True
                )

                print(
                    "C141_SAFE_TASK=",
                    json.dumps({
                        "seed_requested": int(seed),
                        "seed_actual": candidate_seed,
                        "retry": attempt,
                        "distance_m": wanted,
                        "angle_deg": float(angle_deg),
                        "goal_xy": goal.tolist(),
                        "geometry_collision": False
                    }),
                    flush=True
                )
                return sample

            print(
                "C141_BAD_GOAL_REJECT=",
                json.dumps({
                    "seed_requested": int(seed),
                    "seed_actual": candidate_seed,
                    "retry": attempt,
                    "distance_m": wanted,
                    "angle_deg": float(angle_deg)
                }),
                flush=True
            )

            try:
                renderer.close()
            finally:
                env.close()

        raise RuntimeError(
            "C141_NO_SAFE_GOAL_AFTER_24_ATTEMPTS:"
            + repr((seed, angle_deg, wanted))
        )

    _c135_original_task = _c141_task_resample

    runner.make_task = _c135_near_task

    if os.environ.get("C22_SCREEN_ONLY") == "1":
        screening_cases = [
            (2029031000 + i, angle)
            for i, angle in enumerate(TRAIN_ANGLES)
        ]
        screening_cases += list(TESTS)

        screening_rows = []

        for seed, angle in screening_cases:
            env = None
            renderer = None

            try:
                env, renderer, obs, depth, relocated = (
                    runner.make_task(
                        v, native_mod, N,
                        seed, angle,
                        cleared=False,
                        limit=CAP
                    )
                )

                screening_rows.append({
                    "seed": int(seed),
                    "angle_deg": int(angle),
                    "distance_m": float(env._goal_state()[1]),
                    "bearing_deg": math.degrees(
                        float(env._goal_state()[2])
                    ),
                    "start_lidar_min_m": float(
                        (np.min(obs[:48]) * float(env.lidar_max_range))
                    ),
                    "collision": bool(env._collision()),
                    "obstacles_removed": int(relocated),
                })

            finally:
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()

        output("screen.json", {
            "status": "PASS",
            "cases": screening_rows,
            "source_sha256": sha(Path(__file__)),
            "training_started": False,
            "phase_touched": False,
        })

        print(
            "C22_SCREEN_PASS=TRUE",
            "CASES=", len(screening_rows),
            flush=True
        )
        return

    # Training-only exploration correction.
    # Native SAC remains responsible for all evaluation actions.
    import inspect

    episode_source = inspect.getsource(
        runner._g20f6c10_episode_once
    )

    patches = [
        (
            "if random.random() < 0.30:",
            "if random.random() < 0.15:"
        ),
        (
            "random.uniform(0.55, 1.0)",
            "random.uniform(0.45, 0.85)"
        ),
    ]

    for before, after in patches:
        check(
            episode_source.count(before) == 1,
            "EXPLORATION_ANCHOR_MISMATCH:" + before
        )
        episode_source = episode_source.replace(
            before, after, 1
        )

    compile(
        episode_source,
        str(SRC) + ":C22_EXPLORATION_ONLY",
        "exec"
    )

    exec(
        episode_source,
        runner.__dict__,
        runner.__dict__
    )

    print(
        "C22_TRAINING_EXPLORATION_PATCH=PASS",
        "RANDOM_ACTION_PROBABILITY=0.15",
        "RANDOM_THROTTLE_RANGE=0.45_TO_0.85",
        flush=True
    )

    runner.HEADING_BONUS = 240.0
    runner.REWARD_SCALE = 0.05
    print("C73_PROGRESS_SAFE_THROTTLE_ACTIVE=TRUE", flush=True)

    native_step = N.step_checked

    if os.environ.get("STAGE_TD3_T1") == "1":
        from collections import deque
        import importlib.util as td3_importlib

        _td3_spec = td3_importlib.spec_from_file_location(
            "stage_td3_t1",
            os.environ["STAGE_TD3_CORE"]
        )

        if _td3_spec is None or _td3_spec.loader is None:
            raise RuntimeError("TD3_CORE_IMPORT_FAILURE")

        _td3_mod = td3_importlib.module_from_spec(_td3_spec)
        _td3_spec.loader.exec_module(_td3_mod)

        torch.manual_seed(20261027)
        rng = np.random.default_rng(20261027)

        agent = _td3_mod.TD3(
            state_dim=112,
            action_dim=2,
            device="cpu"
        )

        memory = deque(maxlen=10000)

        results = []
        total_physics = 0
        decisions = 0
        losses = []

        deadline_t1 = time.monotonic() + 3000

        tasks = (
            (2029461000, -30, 0.8),
            (2029461001, 30, 0.8),
        )

        for seed, angle, requested in tasks:

            os.environ["C55_TRAIN_GOAL_M"] = str(requested)

            env = None
            renderer = None

            try:
                env, renderer, obs, depth, relocated = (
                    runner.make_task(
                        v,
                        native_mod,
                        N,
                        seed,
                        angle,
                        cleared=False,
                        limit=CAP
                    )
                )

                obs = np.asarray(
                    obs,
                    dtype=np.float32
                ).reshape(-1)

                actual = float(
                    env._goal_state()[1]
                )

                if (
                    obs.size != 112
                    or not np.isfinite(obs).all()
                    or abs(actual - requested) > 0.03
                ):
                    raise RuntimeError(
                        "TD3_T1_INITIAL_OBSERVATION_GOAL_INVALID"
                    )

                print(
                    "TD3_T1_GOAL_GATE=",
                    json.dumps({
                        "seed": seed,
                        "angle": angle,
                        "requested_m": requested,
                        "actual_m": actual,
                        "obs_dim": int(obs.size)
                    }),
                    flush=True
                )

                branch_frames = 0
                branch_actions = 0
                contact = False
                reached = False

                while (
                    branch_frames < 900
                    and total_physics < 2000
                ):

                    if time.monotonic() >= deadline_t1:
                        raise RuntimeError(
                            "TD3_T1_WALL_BUDGET_REACHED"
                        )

                    # 前24次动作仅用于初始探索。
                    # 之后使用TD3策略与探索噪声。
                    if decisions < 24:

                        raw = rng.uniform(
                            -1.0,
                            1.0,
                            2
                        ).astype(np.float32)

                    else:

                        raw = np.clip(
                            agent.act(obs)
                            + rng.normal(
                                0.0,
                                0.12,
                                2
                            ),
                            -1.0,
                            1.0
                        ).astype(np.float32)

                    # TD3归一化动作：
                    # 第一维映射到[0,1]油门；
                    # 第二维保持[-1,1]转向。
                    # 不加入人工导航控制器。
                    action = np.asarray(
                        [
                            (float(raw[0]) + 1.0) * 0.5,
                            float(raw[1])
                        ],
                        dtype=np.float32
                    )

                    before_t = float(
                        env.data.time
                    )

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info
                    ) = native_step(
                        env,
                        action
                    )

                    next_obs = np.asarray(
                        next_obs,
                        dtype=np.float32
                    ).reshape(-1)

                    after_t = float(
                        env.data.time
                    )

                    dt = float(
                        env.model.opt.timestep
                    )

                    nframe = int(
                        round(
                            (after_t - before_t) / dt
                        )
                    )

                    if not 1 <= nframe <= 100:
                        raise RuntimeError(
                            "TD3_T1_PHYSICS_FRAME_DELTA_INVALID:"
                            + repr(nframe)
                        )

                    if (
                        next_obs.size != 112
                        or not np.isfinite(next_obs).all()
                        or not np.isfinite(reward)
                    ):
                        raise RuntimeError(
                            "TD3_T1_STEP_DATA_INVALID"
                        )

                    done = (
                        bool(terminated)
                        or bool(truncated)
                    )

                    # Critic保存归一化动作，
                    # 环境执行动作仅进行仿射尺度转换。
                    memory.append(
                        (
                            obs.copy(),
                            raw.copy(),
                            float(reward),
                            next_obs.copy(),
                            float(done)
                        )
                    )

                    branch_frames += nframe
                    total_physics += nframe

                    decisions += 1
                    branch_actions += 1

                    obs = next_obs

                    # 使用真实MuJoCo数据更新TD3。
                    if (
                        len(memory) >= 32
                        and decisions % 3 == 0
                    ):

                        ids = rng.choice(
                            len(memory),
                            size=32,
                            replace=False
                        )

                        sample = [
                            memory[int(i)]
                            for i in ids
                        ]

                        batch = tuple(
                            np.asarray(
                                [
                                    row[k]
                                    for row in sample
                                ],
                                dtype=np.float32
                            )
                            for k in range(5)
                        )

                        loss = agent.update(
                            batch
                        )

                        if not np.isfinite(
                            loss["critic_loss"]
                        ):
                            raise RuntimeError(
                                "TD3_T1_CRITIC_LOSS_NONFINITE"
                            )

                        losses.append(
                            loss["critic_loss"]
                        )

                    contact = (
                        contact
                        or bool(
                            info.get("collision", False)
                        )
                    )

                    reached = (
                        reached
                        or bool(
                            info.get("reach_goal", False)
                        )
                    )

                    if done or contact or reached:
                        break

                result = {
                    "seed": seed,
                    "angle": angle,
                    "goal_requested_m": requested,
                    "goal_initial_m": actual,
                    "goal_final_m": float(
                        env._goal_state()[1]
                    ),
                    "physics_frames": branch_frames,
                    "decisions": branch_actions,
                    "contact": contact,
                    "goal": reached
                }

                results.append(result)

                print(
                    "TD3_T1_EPISODE=",
                    json.dumps(result),
                    flush=True
                )

            finally:

                if renderer is not None:
                    renderer.close()

                if env is not None:
                    env.close()

        report = {
            "state": (
                "PASS"
                if (
                    len(results) == 2
                    and agent.total_updates > 0
                )
                else "FAIL"
            ),
            "physics_frames": total_physics,
            "transitions": len(memory),
            "td3_updates": agent.total_updates,
            "episodes": results,
            "mean_critic_loss": (
                float(np.mean(losses))
                if losses
                else None
            ),
            "algorithm": "TD3",
            "automatic_promotion": False,
            "sac_training": False
        }

        output(
            "td3_t1_status.json",
            report
        )

        if report["state"] != "PASS":
            raise RuntimeError(
                "TD3_T1_FUNCTIONAL_GATE_FAILED:"
                + repr(report)
            )

        candidate = (
            JOB / "td3_t1_research_only.pt"
        )

        temp = candidate.with_suffix(
            ".pt.partial"
        )

        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "actor_target":
                    agent.actor_target.state_dict(),
                "critic_target":
                    agent.critic_target.state_dict(),
                "actor_optimizer":
                    agent.actor_optimizer.state_dict(),
                "critic_optimizer":
                    agent.critic_optimizer.state_dict(),
                "updates": agent.total_updates,
                "physics_frames": total_physics
            },
            temp
        )

        os.replace(temp, candidate)

        print(
            "TD3_T1_FUNCTIONAL_PASS=TRUE",
            json.dumps(report),
            flush=True
        )

        return

    clock = time.monotonic()
    deadline = clock + SECONDS
    start_frame = base_frames

    count = 0
    winner = False
    success_seen = False

    def save():
        snapshot = p2.snapshot(
            v, policy, replay, episodes,
            "depth_conservative", cfg, sources
        )

        snapshot["g20f6c22"] = {
            "experimental_short_goal_only": True,
            "automatic_promotion": False,
            "script_sha256": sha(Path(__file__)),
            "parent_checkpoint_sha256": sha(CK),
        }

        tmp = JOB / "work_resume.pt.partial"

        torch.save(snapshot, tmp)
        os.replace(tmp, work)

        del snapshot

    def evaluate(seed, angle):
        trace = []
        counter = [0]

        def recorder(env, action):
            result = native_step(env, action)

            obs, reward, terminated, truncated, info = result
            counter[0] += 1

            if (
                counter[0] % 2 == 0
                or bool(info.get("reach_goal", False))
                or bool(info.get("collision", False))
            ):
                trace.append({
                    "step": counter[0],
                    "x_m": float(env.data.qpos[0]),
                    "y_m": float(env.data.qpos[1]),
                    "yaw_rad": float(env._pose()[2]),
                    "goal_x_m": float(env.goal_xy[0]),
                    "goal_y_m": float(env.goal_xy[1]),
                    "remaining_m": float(env._goal_state()[1]),
                    "steer": float(action[1]),
                    "throttle": float(action[0]),
                    "lidar_min_m": float((np.min(obs[:48]) * float(env.lidar_max_range))),
                    "collision": bool(info.get("collision", False)),
                    "reach_goal": bool(info.get("reach_goal", False)),
                })

            return result

        N.step_checked = recorder

        try:
            with torch.no_grad():
                result = runner._g20f6c10_episode_once(
                    v, native_mod, prep, policy,
                    seed, angle,
                    train=False,
                    cap=CAP,
                    clear=False
                )
        finally:
            N.step_checked = native_step

        correct = (
            result.get("end") == "GOAL"
            and result.get("contact") == 0
            and result.get("safe") == 1
        )

        print(
            "C22_HOLDOUT=",
            json.dumps({
                "seed": seed,
                "angle": angle,
                "result": result,
                "verified_goal": correct,
            }),
            flush=True
        )

        if correct:
            evidence = {
                "result": result,
                "seed": seed,
                "angle": angle,
                "checkpoint": str(work),
                "parent_sha256": sha(CK),
                "model_promoted": False,
                "trajectory_rows": len(trace),
            }

            output("success_evidence.json", evidence)

            with (JOB / "success_trace.csv").open(
                "w", newline=""
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(trace[0].keys())
                )
                writer.writeheader()
                writer.writerows(trace)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 6))

                ax.plot(
                    [r["x_m"] for r in trace],
                    [r["y_m"] for r in trace],
                    label="SAC autonomous trajectory"
                )

                ax.scatter(
                    trace[0]["x_m"],
                    trace[0]["y_m"],
                    s=60,
                    label="Start"
                )

                ax.scatter(
                    trace[0]["goal_x_m"],
                    trace[0]["goal_y_m"],
                    s=70,
                    label="Goal"
                )

                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()

                ax.set_title(
                    "Verified short-goal Stage success"
                )

                fig.savefig(
                    JOB / "success_trajectory.png",
                    dpi=180
                )
                plt.close(fig)

            except Exception as err:
                print("PLOT_OPTIONAL_ERROR=", str(err))

        return correct

    try:
        while (
            time.monotonic() < deadline - 60
            and int(v.TRAIN_COUNTER["frames"]) - start_frame
            < MAX_EXTRA
        ):
            angle = TRAIN_ANGLES[count % len(TRAIN_ANGLES)]

            c55_distance = (0.8, 0.8, 1.0, 1.0, 0.4, 0.4, 1.2, 1.2)[count % 8]

            os.environ["C55_TRAIN_GOAL_M"] = str(
                c55_distance
            )

            print(
                "C55_CURRICULUM_REQUEST=",
                json.dumps({
                    "episode": int(count),
                    "distance_m": float(c55_distance),
                    "angle_deg": int(angle),
                    "seed": int(2029461000 + count),
                }),
                flush=True
            )

            seed = 2029461000 + count

            events = {
                "steps": 0,
                "positive_steer_steps": 0,
            }

            def training_step(env, action):
                if not events.get(
                    "c146_goal_verified", False
                ):
                    requested = float(
                        os.environ["C55_TRAIN_GOAL_M"]
                    )

                    actual = float(
                        env._goal_state()[1]
                    )

                    print(
                        "C146_RUNTIME_GOAL_GATE=",
                        json.dumps({
                            "requested_m": requested,
                            "actual_m": actual
                        }),
                        flush=True
                    )

                    if abs(actual-requested) > 0.03:
                        raise RuntimeError(
                            "C146_RUNTIME_GOAL_INVALID:"
                            + repr((requested, actual))
                        )

                    events["c146_goal_verified"] = True

                d0 = float(env._goal_state()[1])
                heading0 = float(env._goal_state()[2])

                c65_dist_before = float(env._goal_state()[1])
                c65_time_before = float(env.data.time)
                result = native_step(env, action)

                obs, reward, terminated, truncated, info = result
                c65_dist_after = float(env._goal_state()[1])
                c65_dt = max(float(env.data.time) - c65_time_before, 1e-6)
                c65_speed_toward = (c65_dist_before - c65_dist_after) / c65_dt
                if not np.isfinite(c65_speed_toward):
                    c65_speed_toward = 0.0
                c65_clearance = float(np.min(obs[:48]) * float(env.lidar_max_range))
                if not np.isfinite(c65_clearance):
                    c65_clearance = 0.0
                c65_speed_factor = float(np.clip((c65_clearance - 0.45) / 0.50, 0.0, 1.0))
                reward = float(reward) + 0.50 * float(np.clip(c65_speed_toward, 0.0, 0.34)) * c65_speed_factor
                if c65_clearance < 0.75:
                    reward -= 2.0 * float(np.clip((0.75 - c65_clearance) / 0.75, 0.0, 1.0)) * max(0.0, float(action[0]))



                # C73: signed progress and clearance-aware throttle.
                # No external controller or action override.

                c73_progress = float(
                    np.clip(c65_speed_toward, -0.30, 0.30)
                )

                c73_safe = float(
                    np.clip(
                        (c65_clearance - 0.36) / 0.64,
                        0.0, 1.0
                    )
                )

                c73_align = float(
                    np.clip(
                        (1.05 - abs(heading0)) / 1.05,
                        0.0, 1.0
                    )
                )

                c73_far = float(
                    np.clip(
                        (
                            c65_dist_before
                            - float(env.success_radius)
                        ) / 0.60,
                        0.0, 1.0
                    )
                )

                c73_pos = float(
                    np.clip(action[0], 0.0, 1.0)
                )

                c73_neg = float(
                    np.clip(-action[0], 0.0, 1.0)
                )

                reward = float(reward) + (
                    8.0 * c73_progress
                    + 0.95
                    * c73_safe
                    * c73_align
                    * c73_far
                    * c73_pos
                    - 1.05
                    * (1.0 - c73_safe)
                    * c73_pos ** 2
                    - 0.15 * c73_safe * c73_neg
                )

                # 在较开阔且远离终点的区域，
                # 避免将停车作为长期策略。
                if (
                    c65_dist_before
                    > float(env.success_radius) + 0.35
                    and c73_safe > 0.7
                    and abs(c73_progress) < 0.008
                ):
                    reward -= 0.06

                if bool(info.get("collision", False)):
                    reward -= 50.0

                if bool(info.get("reach_goal", False)):
                    reward += 120.0

                d1 = float(env._goal_state()[1])
                heading1 = float(env._goal_state()[2])

                minimum = float(
                    (np.min(np.asarray(obs, dtype=float)[:48]) * float(env.lidar_max_range))
                )

                # Training-only reward shaping.
                # Inference actions remain outputs of the SAC policy.
                reward = float(reward)

                reward += 60.0 * max(
                    -0.03,
                    min(0.03, d0 - d1)
                )

                reward += 70.0 * (
                    math.cos(heading1)
                    - math.cos(heading0)
                )

                # C104: no reward for steering command alone

                if minimum < 0.50:
                    reward -= (
                        8.0
                        * (0.50 - minimum) / 0.50
                        * max(0.0, float(action[0]))
                    )

                if bool(info.get("collision", False)):
                    reward -= 120.0

                if (
                    bool(info.get("reach_goal", False))
                    and not bool(info.get("collision", False))
                ):
                    reward += 120.0

                events["steps"] += 1

                if d1 <= 0.65:
                    events["c135_near_steps"] = (
                        events.get("c135_near_steps", 0)
                        + 1
                    )

                if events["steps"] % 1000 == 0:
                    print(
                        "C135_NEAR_EXPOSURE=",
                        json.dumps({
                            "steps": events["steps"],
                            "near_steps": events.get(
                                "c135_near_steps", 0
                            ),
                            "remaining_m": round(d1, 4),
                            "throttle": round(
                                float(action[0]), 4
                            )
                        }),
                        flush=True
                    )

                if float(action[1]) > 0:
                    events["positive_steer_steps"] += 1

                if events["steps"] % 1000 == 0:
                    print(
                        "C22_LIVE=",
                        json.dumps({
                            "episode": count + 1,
                            "steps": events["steps"],
                            "remaining_m": round(d1, 3),
                            "lidar_min_m": round(minimum, 3),
                            "steer": round(float(action[1]), 3),
                            "throttle": round(float(action[0]), 3),
                        }),
                        flush=True
                    )

                return (
                    obs, reward,
                    terminated, truncated, info
                )

            N.step_checked = training_step

            try:
                row = runner._g20f6c10_episode_once(
                    v, native_mod, prep, policy,
                    seed, angle,
                    train=True,
                    replay=replay,
                    cap=min(
                        CAP,
                        MAX_EXTRA - (
                            int(v.TRAIN_COUNTER["frames"])
                            - start_frame
                        )
                    ),
                    clear=False
                )

            finally:
                N.step_checked = native_step

            episodes += 1
            count += 1

            v.TRAIN_COUNTER["frames"] = (
                int(v.TRAIN_COUNTER["frames"])
                + int(row["steps"])
            )

            save()

            print(
                "C22_TRAIN_EPISODE=",
                json.dumps({
                    "episode": count,
                    "total_frames": int(
                        v.TRAIN_COUNTER["frames"]
                    ),
                    "updates": int(policy.agent.updates),
                    "task": row,
                    "positive_steer_physics_steps":
                        events["positive_steer_steps"],
                }),
                flush=True
            )

            output("status.json", {
                "state": "RUNNING",
                "total_frames": int(v.TRAIN_COUNTER["frames"]),
                "updates": int(policy.agent.updates),
                "episodes": episodes,
                "parent": str(CK),
                "checkpoint": str(work),
                "elapsed_seconds": round(
                    time.monotonic() - clock, 2
                ),
                "promotion": False,
            })

            if time.monotonic() > deadline - 180:
                break

            if count % 4 == 0:
                for test_seed, test_angle in TESTS:
                    if time.monotonic() > deadline - 120:
                        break

                    if evaluate(test_seed, test_angle):
                        winner = True
                        success_seen = True
                        break

            if winner:
                print(
                    "C63_SUCCESS_OBSERVED_CONTINUE_TRAINING=TRUE",
                    flush=True
                )
                winner = False

    except KeyboardInterrupt:
        print(
            "INTERRUPTED_WORK_CHECKPOINT=",
            work,
            flush=True
        )

    finally:
        N.step_checked = native_step

    output("status.json", {
        "state": (
            "TRAINING_FINISHED_SUCCESS_OBSERVED"
            if success_seen
            else "TRAINING_FINISHED_NO_SUCCESS_OBSERVED"
        ),
        "success_observed": bool(success_seen),
        "total_frames": int(v.TRAIN_COUNTER["frames"]),
        "updates": int(policy.agent.updates),
        "episodes": episodes,
        "parent": str(CK),
        "checkpoint": (
            str(work) if work.exists() else None
        ),
        "elapsed_seconds": round(
            time.monotonic() - clock, 2
        ),
        "promotion": False,
    })

    print(
        "C63_SUCCESS_OBSERVED=", success_seen,
        "CHECKPOINT=",
        work if work.exists() else "NONE",
        flush=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("C22_FAILED=TRUE", flush=True)
        raise
    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
