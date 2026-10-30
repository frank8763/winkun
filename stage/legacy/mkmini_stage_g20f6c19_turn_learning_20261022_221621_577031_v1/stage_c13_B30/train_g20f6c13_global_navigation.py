"""Stage G19-P4F: symmetric FINAL-goal curriculum, true SAC updates, no planner."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
HOME = Path.home()
P2 = HOME / 'mkmini_stage_g19_p2_multiarm_20261018_222036_624927500/p2.py'
P4E = HOME / 'mkmini_stage_g19_p4e_goal_steer_20261019_104531_069833834/summary.json'
RESUME_CK = Path('/home/linghu/mkmini_stage_g20f5c3_direction_response_retry_20261020_v1/candidate_research_only.pt')
RESUME_FRAMES = 390000
RESUME_EPISODES = 31
RESUME_UPDATES = 19422
ANG = [-35, -25, -15, -5, 0, 5, 15, 25, 35]
GAMMA = 0.99995
REWARD_SCALE = 0.05
HEADING_BONUS = 40.0
FRAME_BUDGET = 1500000
EPISODE_STEPS = 18000
TEST_STEPS = 18000

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda : f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, obj):
    p = Path(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, p)

def load_native():
    spec = importlib.util.spec_from_file_location('g19_p4f_p2', P2)
    p2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = p2
    spec.loader.exec_module(p2)
    v = p2.load_native()
    v.MAXMEM = 4096
    (mod, prep, hashes) = v.source_gate()
    return (p2, v, mod, prep, hashes)
_G20F6B0_SAFE_START_CACHE = {}

def _g20f6c10_make_task_once(v, mod, N, seed, angle_deg, *, cleared=True, limit=EPISODE_STEPS):
    env = None
    renderer = None
    try:
        (env, obs, info, original_goal) = N.create(mod.PR, 'reference', 'medium', int(seed))
        env.max_steps = int(limit)

        def lidar_minimum():
            scan = np.asarray(env._lidar48(), dtype=np.float64).reshape(-1)
            if scan.size != 48:
                return 0.0
            if not np.isfinite(scan).all():
                return 0.0
            return float(np.min(scan))
        static_signature = []
        for gid in range(env.model.ngeom):
            if int(env.model.geom_bodyid[gid]) != 0:
                continue
            name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_GEOM, gid) or ''
            position = tuple(np.round(np.asarray(env.model.geom_pos[gid], dtype=float), 3).tolist())
            size = tuple(np.round(np.asarray(env.model.geom_size[gid], dtype=float), 3).tolist())
            static_signature.append((gid, name, int(env.model.geom_type[gid]), position, size))
        original_xy = np.asarray(env.data.qpos[:2], dtype=np.float64).copy()
        yaw = float(env._pose()[2])
        cache_key = (tuple(static_signature), round(yaw, 4))
        if cache_key not in _G20F6B0_SAFE_START_CACHE:
            candidates = []
            grid = np.arange(-4.0, 4.0001, 0.25)
            for x in grid:
                for y in grid:
                    try:
                        env.data.qpos[:2] = [float(x), float(y)]
                        mj.mj_forward(env.model, env.data)
                        if env._collision():
                            continue
                        clearance = lidar_minimum()
                        if 0.15 <= clearance <= 7.5:
                            candidates.append((clearance, float(x), float(y)))
                    except Exception:
                        continue
            env.data.qpos[:2] = original_xy
            mj.mj_forward(env.model, env.data)
            candidates.sort(key=lambda item: item[0], reverse=True)
            _G20F6B0_SAFE_START_CACHE[cache_key] = candidates
            if len(_G20F6B0_SAFE_START_CACHE) > 32:
                oldest = next(iter(_G20F6B0_SAFE_START_CACHE))
                if oldest != cache_key:
                    _G20F6B0_SAFE_START_CACHE.pop(oldest, None)
        candidates = _G20F6B0_SAFE_START_CACHE[cache_key]
        if not candidates:
            raise RuntimeError('G20F6B0_NO_COLLISION_FREE_START_CANDIDATES')
        target_code = (abs(int(seed)) * 1103515245 + (int(angle_deg) + 180) * 12345) % 1000003
        preferred_distance = 6.0 + float(target_code) / 1000003.0
        distances = []
        for distance in (7.0, 6.75, 6.5, 6.25, 6.0, 5.75, 5.5, 5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0):
            distance = float(distance)
            if not any((abs(distance - previous) < 1e-06 for previous in distances)):
                distances.append(distance)
        requested_heading = yaw + math.radians(float(angle_deg))
        direction = np.array([math.cos(requested_heading), math.sin(requested_heading)], dtype=np.float64)
        selected = None
        clearance_tiers = [
            (0.65, 0.3),
            (0.5, 0.22),
            (0.35, 0.15),
        ]

        for distance in distances:
            for (start_floor, goal_floor) in clearance_tiers:
                for (start_clearance, x, y) in candidates:
                    if start_clearance < start_floor:
                        continue

                    start_xy = np.array(
                        [x, y], dtype=np.float64
                    )

                    env.data.qpos[:2] = start_xy
                    mj.mj_forward(env.model, env.data)

                    if env._collision():
                        continue

                    actual_start_clearance = lidar_minimum()
                    if actual_start_clearance < start_floor:
                        continue

                    goal_xy = (
                        start_xy
                        + float(distance) * direction
                    )

                    env.data.qpos[:2] = goal_xy
                    mj.mj_forward(env.model, env.data)

                    if env._collision():
                        continue

                    goal_clearance = lidar_minimum()
                    if goal_clearance < goal_floor:
                        continue

                    selected = (
                        start_xy.copy(),
                        goal_xy.copy(),
                        float(distance),
                        float(actual_start_clearance),
                        float(goal_clearance),
                    )
                    break

                if selected is not None:
                    break

            if selected is not None:
                break

        if selected is None:
            raise RuntimeError(f'G20F6B0_NO_SAFE_START_GOAL_PAIR:seed={seed}:angle={angle_deg}')
        (start_xy, goal_xy, requested_distance, start_clearance, goal_clearance) = selected
        env.data.qpos[:2] = start_xy
        mj.mj_forward(env.model, env.data)
        env.goal_xy = np.asarray(goal_xy, dtype=np.float64).copy()
        env.prev_goal_dist = float(env._goal_state()[1])
        actual_distance = float(np.linalg.norm(goal_xy - start_xy))
        if not 5.999 <= actual_distance <= 7.001:
            raise RuntimeError('G20F6B0_GOAL_DISTANCE_OUT_OF_RANGE')
        if env._collision():
            raise RuntimeError('G20F6B0_SELECTED_START_COLLISION')
        obs = np.asarray(env._observation(update_history=False), dtype=np.float32).reshape(-1)
        if obs.shape != (112,) or not np.isfinite(obs).all():
            raise RuntimeError('G20F6B0_INITIAL_OBSERVATION_INVALID')
        if not np.allclose(obs[:48], env._lidar48(), atol=1e-05, rtol=0):
            raise RuntimeError('G20F6B0_INITIAL_LIDAR_DESYNC')
        renderer = mj.Renderer(env.model, height=120, width=160)
        depth = v.depth_frame(renderer, env)
        print('G20F6B0_SAFE_START_GOAL=', json.dumps({'seed': int(seed), 'angle_deg': int(angle_deg), 'start_xy_m': start_xy.tolist(), 'goal_xy_m': goal_xy.tolist(), 'distance_m': round(actual_distance, 6), 'start_lidar_min_m': round(start_clearance, 4), 'goal_lidar_min_m': round(goal_clearance, 4), 'obstacles_cleared': 0}), flush=True)
        return (env, renderer, obs, depth, 0)
    except Exception:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        raise


_G20F6C10_LAST_TASK_SEED = None
_G20F6C10_LAST_TASK_RETRY = 0
_G20F6C10_MAX_TASK_SEED_RETRIES = 8


_G20F6C11_LAST_TASK_SEED = None
_G20F6C11_LAST_SEED_ATTEMPT = None
_G20F6C11_LAST_ANGLE_ATTEMPT = None
_G20F6C11_LAST_ACTUAL_ANGLE = None
_G20F6C11_ACTUAL_ANGLE_BY_REQUESTED = {}

def make_task(v, mod, N, seed, angle_deg, *, cleared=True,
              limit=EPISODE_STEPS):
    """
    G20-F6C13 task generator.

    Priority:
      1. Keep the native spawn and search globally for a target.
      2. Search safe global spawn positions only if necessary.
      3. Prefer the longest feasible target distance.
      4. Never remove obstacles or introduce an external planner.
    """
    env = None
    renderer = None

    try:
        env, obs, info, original_goal = N.create(
            mod.PR, "reference", "medium", int(seed)
        )
        env.max_steps = int(limit)

        def lidar_minimum():
            scan = np.asarray(
                env._lidar48(), dtype=np.float64
            ).reshape(-1)

            if scan.size != 48:
                return 0.0
            if not np.isfinite(scan).all():
                return 0.0

            return float(np.min(scan))

        base_qpos = np.asarray(
            env.data.qpos, dtype=np.float64
        ).copy()

        base_qvel = np.asarray(
            env.data.qvel, dtype=np.float64
        ).copy()

        native_xy = base_qpos[:2].copy()
        yaw = float(env._pose()[2])

        candidate_tests = 0
        selected = None

        def pose_at(xy):
            env.data.qpos[:] = base_qpos
            env.data.qpos[:2] = np.asarray(
                xy, dtype=np.float64
            )
            env.data.qvel[:] = base_qvel
            mj.mj_forward(env.model, env.data)

        # Requested direction first, then explore the full circle.
        # Offsets cover all 36 ten-degree directions.
        native_offsets = [0]
        for k in range(1, 18):
            native_offsets.extend([10 * k, -10 * k])
        native_offsets.append(180)

        native_radii = [
            round(7.0 - 0.25 * i, 2)
            for i in range(17)
        ]

        def find_native_goal(start_xy, offsets, radii):
            nonlocal candidate_tests

            for distance in radii:
                for offset in offsets:
                    theta = yaw + math.radians(
                        float(angle_deg) + float(offset)
                    )

                    target = start_xy + distance * np.array(
                        [math.cos(theta), math.sin(theta)],
                        dtype=np.float64
                    )

                    pose_at(target)
                    candidate_tests += 1

                    if env._collision():
                        continue

                    clearance = lidar_minimum()

                    if clearance < 0.15:
                        continue

                    return (
                        target.copy(),
                        float(distance),
                        float(clearance)
                    )

            return None

        # ----------------------------------------------------
        # 1. First try the native start. Do not move it.
        # ----------------------------------------------------
        pose_at(native_xy)

        native_collision = bool(env._collision())
        native_clearance = lidar_minimum()

        if (
            not native_collision
            and native_clearance >= 0.15
        ):
            result = find_native_goal(
                native_xy,
                native_offsets,
                native_radii
            )

            if result is not None:
                selected = (
                    native_xy.copy(),
                    float(native_clearance),
                    result,
                    False
                )

        # ----------------------------------------------------
        # 2. If the native start has no feasible target,
        #    search safe starts across the scene.
        # ----------------------------------------------------
        if selected is None:
            safe_starts = []

            # Independent global grid. This only samples
            # candidate robot positions; it does not modify
            # the environment or remove any obstacles.
            grid = np.arange(-4.0, 4.0001, 0.5)

            for x in grid:
                for y in grid:
                    xy = np.array(
                        [float(x), float(y)],
                        dtype=np.float64
                    )

                    pose_at(xy)

                    if env._collision():
                        continue

                    clearance = lidar_minimum()

                    if clearance >= 0.15:
                        safe_starts.append(
                            (float(clearance), xy.copy())
                        )

            # Prefer safer candidate starts.
            safe_starts.sort(
                key=lambda item: item[0],
                reverse=True
            )

            # Bound the fallback search cost.
            safe_starts = safe_starts[:64]

            # Coarse global search, longest distance first.
            fallback_radii = [
                7.0, 6.5, 6.0, 5.5, 5.0,
                4.5, 4.0, 3.5, 3.0
            ]

            fallback_offsets = [0]
            for k in range(1, 12):
                fallback_offsets.extend(
                    [15 * k, -15 * k]
                )
            fallback_offsets.append(180)

            for distance in fallback_radii:
                if selected is not None:
                    break

                for start_clearance, start_xy in safe_starts:
                    if selected is not None:
                        break

                    for offset in fallback_offsets:
                        theta = yaw + math.radians(
                            float(angle_deg) + float(offset)
                        )

                        target = start_xy + distance * np.array(
                            [
                                math.cos(theta),
                                math.sin(theta)
                            ],
                            dtype=np.float64
                        )

                        pose_at(target)
                        candidate_tests += 1

                        if env._collision():
                            continue

                        goal_clearance = lidar_minimum()

                        if goal_clearance < 0.15:
                            continue

                        selected = (
                            start_xy.copy(),
                            float(start_clearance),
                            (
                                target.copy(),
                                float(distance),
                                float(goal_clearance)
                            ),
                            True
                        )
                        break

        if selected is None:
            raise RuntimeError(
                "G20F6C13_NO_GLOBAL_FEASIBLE_TASK:"
                f"seed={seed}:angle={angle_deg}"
            )

        start_xy, start_clearance, goal_info, fallback_start = (
            selected
        )

        goal_xy, requested_distance, goal_clearance = goal_info

        # Restore the selected start and the original robot
        # orientation/velocity before constructing observations.
        pose_at(start_xy)

        env.goal_xy = np.asarray(
            goal_xy, dtype=np.float64
        ).copy()

        env.prev_goal_dist = float(
            env._goal_state()[1]
        )

        actual_distance = float(
            np.linalg.norm(goal_xy - start_xy)
        )

        if not 2.99 <= actual_distance <= 7.01:
            raise RuntimeError(
                "G20F6C13_GOAL_DISTANCE_OUT_OF_RANGE"
            )

        if env._collision():
            raise RuntimeError(
                "G20F6C13_SELECTED_START_COLLISION"
            )

        obs = np.asarray(
            env._observation(update_history=False),
            dtype=np.float32
        ).reshape(-1)

        if obs.shape != (112,):
            raise RuntimeError(
                "G20F6C13_INITIAL_OBSERVATION_SHAPE_INVALID"
            )

        if not np.isfinite(obs).all():
            raise RuntimeError(
                "G20F6C13_INITIAL_OBSERVATION_NONFINITE"
            )

        if not np.allclose(
            obs[:48],
            env._lidar48(),
            atol=1e-5,
            rtol=0
        ):
            raise RuntimeError(
                "G20F6C13_INITIAL_LIDAR_DESYNC"
            )

        renderer = mj.Renderer(
            env.model, height=120, width=160
        )

        depth = v.depth_frame(renderer, env)

        world_heading_deg = (
            math.degrees(
                math.atan2(
                    goal_xy[1] - start_xy[1],
                    goal_xy[0] - start_xy[0]
                )
            ) % 360.0
        )

        relative_heading_deg = (
            (world_heading_deg - math.degrees(yaw) + 180.0)
            % 360.0
        ) - 180.0

        print(
            "G20F6C13_GLOBAL_TASK=",
            json.dumps(
                {
                    "seed": int(seed),
                    "requested_angle_deg": int(angle_deg),
                    "start_xy_m": start_xy.tolist(),
                    "goal_xy_m": goal_xy.tolist(),
                    "distance_m": round(actual_distance, 4),
                    "relative_goal_heading_deg": round(
                        relative_heading_deg, 2
                    ),
                    "start_lidar_min_m": round(
                        start_clearance, 4
                    ),
                    "goal_lidar_min_m": round(
                        goal_clearance, 4
                    ),
                    "fallback_start": bool(fallback_start),
                    "candidate_tests": int(candidate_tests),
                    "obstacles_cleared": 0
                }
            ),
            flush=True
        )

        # Preserve the original five-element return interface.
        return env, renderer, obs, depth, 0

    except Exception:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        raise


def actor_cmd(v, C, obs, depth, prev, last_action, last_duration, policy, stochastic=False):
    feature = C.feat(obs, prev, last_action, last_duration)
    if policy is not None:
        (cmd, idx) = policy.choose(feature, depth, stochastic=stochastic)
        return (feature, np.asarray(cmd, np.float32).reshape(2), int(idx))
    ag = v.FROZEN_AGENT
    with torch.no_grad():
        state = ag.state(feature).reshape(1, -1)
        (all_actions, _, probs, _) = ag.actor.sample_all(state, deterministic=True)
        idx = int(probs[0].argmax().item())
        cmd = all_actions[0, idx].detach().cpu().numpy().astype(np.float32)
    return (feature, cmd.reshape(2), idx)

def _g20f6c10_directional_probe_once(v, mod, prep, policy, seed=2028999601):
    (N, C) = (prep[3], prep[2])
    vals = {}
    for deg in ANG:
        env = renderer = None
        try:
            (env, renderer, obs, depth, num_removed) = make_task(v, mod, N, seed, deg, cleared=False)
            (feat, cmd, idx) = actor_cmd(v, C, obs, depth, obs, np.zeros(2, np.float32), 1, policy)
            vals[str(deg)] = {'steer': float(cmd[1]), 'throttle': float(cmd[0]), 'duration': int(C.DURATIONS[idx])}
        finally:
            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()
    return vals

def _g20f6c10_episode_once(v, mod, prep, policy, seed, angle_deg, *, train, replay=None, cap=EPISODE_STEPS, clear=True):
    (C, N) = (prep[2], prep[3])
    env = renderer = None
    try:
        (env, renderer, obs, depth, relocated) = make_task(v, mod, N, seed, angle_deg, cleared=clear, limit=cap)
        initial_goal_dist = float(env._goal_state()[1])
        prev = obs.copy()
        prev_act = np.zeros(2, np.float32)
        prev_d = 1
        steps = 0
        count = 0
        signed_steering = {'negative': 0, 'positive': 0}
        contact = False
        reached = False
        end = 'LIMIT'
        while steps < cap:
            feat = C.feat(obs, prev, prev_act, prev_d)
            if train:
                (act, idx) = policy.choose(feat, depth, stochastic=True)
                act = np.asarray(act, np.float32).reshape(2).copy()
                if random.random() < 0.30:
                    act[0] = random.uniform(0.55, 1.0)
                    act[1] = random.uniform(-1.0, 1.0)
                    idx = random.randrange(len(C.DURATIONS))
            else:
                (_, act, idx) = actor_cmd(v, C, obs, depth, prev, prev_act, prev_d, policy, stochastic=train)
            if not np.isfinite(act).all() or np.any(np.abs(act) > 1.0001):
                raise RuntimeError('INVALID_ACTION')
            signed_steering['negative' if act[1] < 0 else 'positive'] += 1
            duration = min(int(C.DURATIONS[idx]), cap - steps)
            before = obs.copy()
            reward_total = 0.0
            actual = 0
            initial_angle = float(env._goal_state()[2])
            prev_angle = initial_angle
            terminated = truncated = False
            for j in range(duration):
                (nextobs, reward, terminated, truncated, info) = N.step_checked(env, act)
                ang = float(env._goal_state()[2])
                shaped = float(reward) + HEADING_BONUS * (math.cos(ang) - math.cos(prev_angle))
                reward_total += GAMMA ** j * REWARD_SCALE * shaped
                prev_angle = ang
                actual += 1
                steps += 1
                contact = bool(info.get('collision', False))
                reached = bool(info.get('reach_goal', False)) and (not contact)
                if terminated or truncated or contact or reached:
                    break
            nextobs = np.asarray(nextobs, np.float32).reshape(-1)
            if reached and float(np.linalg.norm(np.asarray(env.data.qpos[:2], float) - np.asarray(env.goal_xy, float))) > float(env.success_radius) + 0.01:
                raise RuntimeError('NATIVE_GOAL_RADIUS_INCONSISTENT')
            if nextobs.shape != (112,) or not np.isfinite(nextobs).all():
                raise RuntimeError('BAD_NEXT_OBS')
            newdepth = v.depth_frame(renderer, env)
            if train:
                nfeat = C.feat(nextobs, before, act, actual)
                replay.add(policy.pack(feat, depth), int(idx), act.copy(), float(reward_total), policy.pack(nfeat, newdepth), float(GAMMA ** actual), float(bool(terminated or contact)))
                if len(replay) >= 96 and steps % 16 < actual:
                    policy.update(replay.sample(64))
            prev = before
            obs = nextobs
            depth = newdepth
            prev_act = act.copy()
            prev_d = actual
            count += 1
            if terminated or truncated or contact or reached:
                end = 'GOAL' if reached else 'CONTACT' if contact else 'TRUNCATED' if truncated else 'TERMINATED'
                break
        xy = np.asarray(env.data.qpos[:2], float)
        remaining = float(np.linalg.norm(xy - env.goal_xy))
        return {'seed': int(seed), 'angle_deg': int(angle_deg), 'steps': int(steps), 'end': end, 'safe': int(reached), 'contact': int(contact), 'start_goal_distance_m': round(initial_goal_dist, 6), 'remaining_m': round(remaining, 6), 'progress_m': round(initial_goal_dist - remaining, 6), 'steer_sign_decisions': signed_steering, 'decisions': count, 'cleared_geoms': relocated}
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()



def directional_probe(v, mod, prep, policy, seed=2028999601):
    global _G20F6C11_ACTUAL_ANGLE_BY_REQUESTED

    last_error = None

    for attempt in range(3):
        probe_seed = int(seed) + attempt * 100003
        _G20F6C11_ACTUAL_ANGLE_BY_REQUESTED = {}

        try:
            vals = _g20f6c10_directional_probe_once(
                v, mod, prep, policy, seed=probe_seed
            )

            # 显式记录请求角度与真实生成角度，避免误读探测结果。
            for key, record in vals.items():
                requested = int(key)
                actual = _G20F6C11_ACTUAL_ANGLE_BY_REQUESTED.get(
                    requested, requested
                )

                record["requested_angle_deg"] = requested
                record["actual_angle_deg"] = int(actual)

            print(
                "G20F6C11_DIRECTIONAL_PROBE=PASS",
                json.dumps({
                    "seed": probe_seed,
                    "directions": {
                        key: {
                            "requested_angle_deg": value.get(
                                "requested_angle_deg"
                            ),
                            "actual_angle_deg": value.get(
                                "actual_angle_deg"
                            ),
                            "steer": value.get("steer"),
                        }
                        for key, value in vals.items()
                    },
                }),
                flush=True,
            )

            return vals

        except RuntimeError as exc:
            message = str(exc)

            if "G20F6C11_NO_FEASIBLE_TASK" not in message:
                raise

            last_error = message

            print(
                "G20F6C11_PROBE_RETRY=",
                json.dumps({
                    "probe_seed": probe_seed,
                    "attempt": attempt + 1,
                    "reason": message,
                }),
                flush=True,
            )

    raise RuntimeError(
        "G20F6C11_DIRECTIONAL_PROBE_FAILED:" + str(last_error)
    )



def episode(*args, **kwargs):
    signature = __import__("inspect").signature(
        _g20f6c10_episode_once
    )

    try:
        bound = signature.bind_partial(*args, **kwargs)
        requested_seed = bound.arguments.get("seed")
        requested_angle = bound.arguments.get("angle_deg")
    except Exception:
        requested_seed = None
        requested_angle = None

    row = _g20f6c10_episode_once(*args, **kwargs)

    if isinstance(row, dict):
        if requested_seed is not None:
            row.setdefault("requested_seed", int(requested_seed))

        if requested_angle is not None:
            row.setdefault(
                "requested_angle_deg", int(requested_angle)
            )

        actual_seed = globals().get(
            "_G20F6C11_LAST_TASK_SEED"
        )
        actual_angle = globals().get(
            "_G20F6C11_LAST_ACTUAL_ANGLE"
        )

        if actual_seed is not None:
            row["task_seed_used"] = int(actual_seed)

        if actual_angle is not None:
            row["task_angle_used_deg"] = int(actual_angle)

        seed_retry = globals().get(
            "_G20F6C11_LAST_SEED_ATTEMPT"
        )
        angle_retry = globals().get(
            "_G20F6C11_LAST_ANGLE_ATTEMPT"
        )

        row["task_seed_retry_count"] = seed_retry
        row["task_angle_retry_count"] = angle_retry

    return row



def main():
    global FRAME_BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--target-frames', type=int, default=None)
    args = ap.parse_args()
    if args.target_frames is not None:
        target = int(args.target_frames)
        if target <= RESUME_FRAMES or target > 1500000:
            raise RuntimeError('G20F6A1_TARGET_FRAME_OVERRIDE_INVALID')
        FRAME_BUDGET = target
        print('G20F6A1_TARGET_FRAME_OVERRIDE=TRUE', FRAME_BUDGET, flush=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if P4E.exists():
        last = json.loads(P4E.read_text())
        if last.get('gate') != 'G19_P4E_GOAL_STEER_CONTRACT_COMPLETE':
            raise RuntimeError('P4E_PREREQUISITE_INVALID')
        print('G19_P6_PREVIOUS_PLANT_SIGNS=', last.get('plant_sign'), flush=True)
    (p2, v, mod, prep, hashes) = load_native()
    (g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck) = prep
    v.FROZEN_AGENT = mod.agent_new(g5b, H, C, ck, snap, d1, device, 202710700)
    v.FROZEN_AGENT.actor.eval()
    v.seed_all(2026101906)
    cfg = copy.deepcopy(p2.ARMS['depth_conservative'])
    cfg.update(actor_lr=1e-06, vision_lr=5e-07, critic_lr=3e-05)
    cfg['course'] = ('f5a_native_near_goal_adaptive',)
    source_hashes = dict(hashes)
    source_hashes[str(P2)] = sha(P2)
    replay = C.Replay(v.MAXMEM)
    (policy, episodes) = p2.resume(v, mod, prep, 'depth_conservative', cfg, replay, RESUME_CK, source_hashes)
    if int(episodes) != RESUME_EPISODES:
        raise RuntimeError('RESUME_EPISODE_RESTORE_FAILED')
    if int(policy.agent.updates) != RESUME_UPDATES:
        raise RuntimeError('RESUME_UPDATE_RESTORE_FAILED')
    if len(replay) < 96:
        raise RuntimeError('RESUME_REPLAY_BUFFER_INVALID')
    v.TRAIN_COUNTER['frames'] = RESUME_FRAMES
    policy.agent.gamma = GAMMA
    policy.update = lambda batch: p2.sac_update(v, policy, batch)
    print('G20F6A1_RESUME_RESTORE=PASS', json.dumps({'checkpoint_sha256': sha(RESUME_CK), 'frames': v.TRAIN_COUNTER['frames'], 'episodes': episodes, 'updates': policy.agent.updates, 'replay_items': len(replay)}), flush=True)
    print('G19_P6_START=SYMMETRIC_3M_FINAL_GOALS', flush=True)
    print('STAGE_ONLY=TRUE NO_PLANNER=TRUE NO_WAYPOINTS=TRUE', flush=True)
    print('GOAL_DISTANCE_M=3.0 EPISODE_MAX_STEPS=', EPISODE_STEPS, 'PHYSICS_GAMMA=', GAMMA, flush=True)
    before = directional_probe(v, mod, prep, None)
    init = directional_probe(v, mod, prep, policy)
    print('G19_P6_FROZEN_INITIAL=', json.dumps(before), flush=True)
    print('G19_P6_STUDENT_INITIAL=', json.dumps(init), flush=True)
    prior_progress = json.loads((RESUME_CK.parent / 'progress.json').read_text())
    if prior_progress.get('status') != 'TRAINING_COMPLETE':
        raise RuntimeError('RESUME_PROGRESS_STATUS_INVALID')
    if int(prior_progress.get('frames', -1)) != RESUME_FRAMES:
        raise RuntimeError('RESUME_PROGRESS_FRAME_MISMATCH')
    if int(prior_progress.get('episodes', -1)) != int(episodes):
        raise RuntimeError('RESUME_PROGRESS_EPISODE_MISMATCH')
    if int(prior_progress.get('parent_frames', -1)) + int(prior_progress.get('new_frames', -1)) != int(prior_progress.get('frames', -1)):
        raise RuntimeError('RESUME_PROGRESS_FRAME_ARITHMETIC_MISMATCH')
    if prior_progress.get('final_checkpoint_sha256') != sha(RESUME_CK):
        raise RuntimeError('RESUME_PROGRESS_CHECKPOINT_HASH_MISMATCH')
    if float(prior_progress.get('final_checkpoint_restore_error', float('inf'))) > 1e-07:
        raise RuntimeError('RESUME_PROGRESS_RESTORE_ERROR')
    if prior_progress.get('phase_touched') is not False:
        raise RuntimeError('RESUME_PHASE_ISOLATION_INVALID')
    if prior_progress.get('automatic_promotion') is not False:
        raise RuntimeError('RESUME_PROMOTION_POLICY_INVALID')
    training_rows = prior_progress.get('training_rows')
    expected_new_episodes = int(prior_progress.get('new_episodes', -1))
    if not isinstance(training_rows, list):
        raise RuntimeError('RESUME_TRAINING_ROWS_MISSING')
    if expected_new_episodes < 0 or len(training_rows) != expected_new_episodes:
        raise RuntimeError('RESUME_TRAINING_ROWS_PROVENANCE_MISMATCH')
    print('G20F6A1_V3_PROGRESS_SCHEMA=PASS', json.dumps({'frames': int(prior_progress['frames']), 'episodes': int(prior_progress['episodes']), 'new_episodes': expected_new_episodes, 'training_rows': len(training_rows), 'checkpoint_sha256': sha(RESUME_CK)}), flush=True)
    if args.preflight_only:
        print('G20F6A1_V3_PREFLIGHT_ONLY=PASS', json.dumps({'frames': int(v.TRAIN_COUNTER['frames']), 'episodes': int(episodes), 'replay_items': len(replay), 'checkpoint_sha256': sha(RESUME_CK), 'training_rows_provenance_count': len(training_rows), 'target_frames': FRAME_BUDGET}), flush=True)
        return
    print('G20F6A1_TRAINING_START=TRUE', json.dumps({'start_frames': int(v.TRAIN_COUNTER['frames']), 'target_total_frames': FRAME_BUDGET, 'additional_frames_planned': FRAME_BUDGET - int(v.TRAIN_COUNTER['frames']), 'goal_bearings_deg': list(ANG), 'symmetric_exploration_probability': 0.4, 'heading_bonus': HEADING_BONUS, 'automatic_promotion': False}), flush=True)
    while v.TRAIN_COUNTER['frames'] < FRAME_BUDGET:
        ix = episodes % len(ANG)
        angle = ANG[ix]
        seed = 2029006500 + episodes
        task_cap = EPISODE_STEPS
        used = min(task_cap, FRAME_BUDGET - v.TRAIN_COUNTER['frames'])
        row = episode(v, mod, prep, policy, seed, angle, train=True, replay=replay, cap=used, clear=False)
        v.TRAIN_COUNTER['frames'] += int(row['steps'])
        episodes += 1
        training_rows.append(row)
        if row['cleared_geoms'] != 0:
            raise RuntimeError('TRAINING_GEOMETRY_MISMATCH')
        print('G19_P6_TRAIN_EPISODE=', json.dumps({'episode': episodes, 'frames': v.TRAIN_COUNTER['frames'], 'updates': policy.agent.updates, **row}), flush=True)
        _g20f6c10_distances = [
            float(r["start_goal_distance_m"])
            for r in training_rows
            if r.get("start_goal_distance_m") is not None
        ]
        _g20f6c10_ge6 = sum(
            d >= 6.0 for d in _g20f6c10_distances
        )
        atomic_json(out / 'progress.json', {
            'frames': int(v.TRAIN_COUNTER['frames']),
            'episodes': int(episodes),
            'rows': training_rows,
            'global_navigation_distance_summary': {
                'training_episodes': len(_g20f6c10_distances),
                'episodes_ge_6m': int(_g20f6c10_ge6),
                'fraction_ge_6m': round(
                    _g20f6c10_ge6 /
                    max(1, len(_g20f6c10_distances)), 6
                ),
                'mean_distance_m': (
                    round(
                        sum(_g20f6c10_distances) /
                        len(_g20f6c10_distances), 6
                    )
                    if _g20f6c10_distances else None
                ),
                'max_distance_m': (
                    max(_g20f6c10_distances)
                    if _g20f6c10_distances else None
                )
            }
        })
        partial_sources = dict(hashes)
        partial_sources[str(P2)] = sha(P2)
        restore_error = p2.checkpoint_roundtrip(v, mod, prep, policy, replay, episodes, 'depth_conservative', cfg, partial_sources, out / 'work_resume.pt')
        if restore_error > 1e-07:
            raise RuntimeError('WORK_CHECKPOINT_INCONSISTENT')
        print('G19_P6_WORK_CHECKPOINT=', json.dumps({'frames': v.TRAIN_COUNTER['frames'], 'episodes': episodes, 'restore_error': restore_error}), flush=True)
        if row['steps'] <= 0:
            raise RuntimeError('ZERO_FRAME_EPISODE')
    after = directional_probe(v, mod, prep, policy)
    print('G19_P6_STUDENT_AFTER=', json.dumps(after), flush=True)
    if policy.agent.updates < 20 or not all((policy.grad_actor > 0, policy.grad_critic > 0, policy.grad_encoder > 0)):
        raise RuntimeError('REAL_SAC_UPDATE_GRADIENT_NOT_PROVEN')
    results = []
    for angle in (-25, 0, 25):
        for seed in (2029007200 + angle, 2029007300 + angle):
            for (label, pol) in (('frozen', None), ('depth_p6', policy)):
                row = episode(v, mod, prep, pol, seed, angle, train=False, cap=TEST_STEPS, clear=False)
                if row['cleared_geoms'] != 0:
                    raise RuntimeError('OBSTACLES_UNEXPECTEDLY_CLEARED')
                row['model'] = label
                results.append(row)
                print('G19_P6_EVAL_CASE=', json.dumps(row), flush=True)
                atomic_json(out / 'evaluation_progress.json', {'rows': results})
    sources = dict(hashes)
    sources[str(P2)] = sha(P2)
    ckpath = out / 'full_scene_candidate_research_only.pt'
    err = p2.checkpoint_roundtrip(v, mod, prep, policy, replay, episodes, 'depth_conservative', cfg, sources, ckpath)
    state = torch.load(ckpath, map_location='cpu', weights_only=False)
    state['g19_p6_semantics'] = {'physical_step_gamma': GAMMA, 'reward_scale': REWARD_SCALE, 'heading_bonus': HEADING_BONUS, 'final_goal_m': 3.0, 'obstacle_free_curriculum': False}
    state['g20c_semantics'] = {'goal_distance_range_m': [6.0, 7.0], 'bearing_curriculum_deg': list(ANG), 'obstacle_free_curriculum': False, 'source_checkpoint_sha256': sha(RESUME_CK)}
    state['g20e2_semantics'] = {'resume_checkpoint_sha256': sha(RESUME_CK), 'resume_start_frames': RESUME_FRAMES, 'total_frame_budget': FRAME_BUDGET, 'exploration_override_probability': 0.4, 'bearing_curriculum_deg': list(ANG), 'automatic_promotion': False}
    tmp_ck = ckpath.with_suffix('.pt.partial')
    torch.save(state, tmp_ck)
    os.replace(tmp_ck, ckpath)
    v.check_sources(mod, prep, hashes)
    base_safe = sum((r['safe'] for r in results if r['model'] == 'frozen'))
    candidate_safe = sum((r['safe'] for r in results if r['model'] == 'depth_p6'))
    old_diff = init['-35']['steer']
    new_diff = after['-35']['steer']
    directional_pass = bool(after['-35']['steer'] < -0.05 and after['35']['steer'] > 0.05)
    no_contact = sum((r['contact'] for r in results if r['model'] == 'depth_p6')) <= sum((r['contact'] for r in results if r['model'] == 'frozen'))
    baseline_progress = float(np.mean([r['progress_m'] for r in results if r['model'] == 'frozen']))
    candidate_progress = float(np.mean([r['progress_m'] for r in results if r['model'] == 'depth_p6']))
    improved = bool(candidate_safe > base_safe and no_contact and (candidate_progress >= baseline_progress - 0.05))
    print('G19_P6_HOLDOUT=', json.dumps({'frozen_safe': base_safe, 'candidate_safe': candidate_safe, 'frozen_progress_m': baseline_progress, 'candidate_progress_m': candidate_progress, 'contacts_ok': no_contact, 'strict_gate': improved}), flush=True)
    report = {'gate': 'G19_P6_FULL_SCENE_INDEPENDENT_IMPROVEMENT' if improved else 'G19_P6_FULL_SCENE_NOT_YET_IMPROVED', 'training_frames': v.TRAIN_COUNTER['frames'], 'training_updates': policy.agent.updates, 'training_episodes': episodes, 'training_native_goals': sum((r['safe'] for r in training_rows)), 'training_contacts': sum((r['contact'] for r in training_rows)), 'frozen_probe': before, 'student_before': init, 'student_after': after, 'right_steer_delta': new_diff - old_diff, 'directional_sign_gate': directional_pass, 'independent_success_improved': improved, 'frozen_eval_safe': base_safe, 'student_eval_safe': candidate_safe, 'evaluation': results, 'checkpoint_restore_error': err, 'night_authorized': False, 'promotion': False, 'sources_unchanged': True, 'full_horizon_training_started': True}
    _g20f6c10_distances = [
        float(r["start_goal_distance_m"])
        for r in training_rows
        if r.get("start_goal_distance_m") is not None
    ]
    _g20f6c10_ge6 = sum(
        d >= 6.0 for d in _g20f6c10_distances
    )
    report["global_navigation_distance_summary"] = {
        "training_episodes": len(_g20f6c10_distances),
        "episodes_ge_6m": int(_g20f6c10_ge6),
        "fraction_ge_6m": round(
            _g20f6c10_ge6 /
            max(1, len(_g20f6c10_distances)), 6
        ),
        "mean_distance_m": (
            round(
                sum(_g20f6c10_distances) /
                len(_g20f6c10_distances), 6
            )
            if _g20f6c10_distances else None
        ),
        "min_distance_m": (
            min(_g20f6c10_distances)
            if _g20f6c10_distances else None
        ),
        "max_distance_m": (
            max(_g20f6c10_distances)
            if _g20f6c10_distances else None
        )
    }
    atomic_json(out / 'summary.json', report)
    print('G19_P6_FINAL=', json.dumps({k: v for (k, v) in report.items() if k not in ('evaluation', 'frozen_probe', 'student_before', 'student_after')}), flush=True)
    print('G19_P6_REPORT=', out / 'summary.json', flush=True)
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('G19_P6_BLOCK=', repr(exc), flush=True)
        traceback.print_exc()
        sys.exit(2)
