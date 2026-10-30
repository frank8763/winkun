from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np
import torch

JOB = Path(os.environ["T119_JOB"])
TRAIN = JOB / "train.py"
CKPT = JOB / "checkpoints/episode_0341.pt"
STATUS = JOB / "status.json"

# 已核对的正确 SHA256，不再使用上一版错误字符串。
EXPECTED_SHA = (
    "4539dbfb627c2177a16249a3c2a300d4b"
    "52e5caf5fc2a5802dc35b939e25897f"
)

GATES = {}
SUMMARIES = []
EVALUATION_STARTED = False


def gate(name, passed, detail=""):
    GATES[name] = bool(passed)
    print(
        f"GATE_{name}={'PASS' if passed else 'FAIL'}",
        detail
    )


def load_module(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("MODULE_SPEC_INVALID: " + str(path))

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe(v, depth=0):
    if depth > 4:
        return str(type(v).__name__)

    if isinstance(v, torch.Tensor):
        return safe(v.detach().cpu().numpy(), depth + 1)

    if isinstance(v, np.ndarray):
        if v.size <= 200:
            return v.tolist()
        return {
            "shape": list(v.shape),
            "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
        }

    if isinstance(v, np.generic):
        return v.item()

    if isinstance(v, dict):
        return {
            str(k): safe(x, depth + 1)
            for k, x in v.items()
        }

    if isinstance(v, (list, tuple)):
        return [safe(x, depth + 1) for x in v]

    if isinstance(v, float):
        return v if math.isfinite(v) else None

    if v is None or isinstance(v, (str, int, bool)):
        return v

    return str(v)[:200]


def angle_wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def body_yaw(env, body_id):
    if body_id is None:
        return None

    R = np.asarray(
        env.data.xmat[body_id],
        dtype=np.float64
    ).reshape(3, 3)

    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def choose_body(env, mujoco):
    names = []

    for i in range(env.model.nbody):
        name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            i
        )
        names.append((i, name))

    print("BODY_NAMES =", names)

    # 优先选车身主体，避免误选 worldbody。
    for token in ("base_link", "chassis", "vehicle", "base"):
        for idx, name in names:
            if idx != 0 and name and token in name.lower():
                return idx

    return 1 if env.model.nbody > 1 else None


def run_case(
    helper, make_base, v, native_mod, N, agent,
    seed, angle, distance, out_dir, mujoco,
    get_last_step_record
):
    side = "LEFT" if angle < 0 else "RIGHT"
    tag = f"seed{seed}_{side}_{abs(angle):.1f}deg"
    trace_path = out_dir / (tag + ".jsonl")

    env = None
    renderer = None
    steps = 0
    collisions = 0
    positive = 0
    negative = 0
    goal_sign_total = 0
    goal_sign_match = 0
    yaw_total = 0
    yaw_sign_match = 0
    previous_yaw = None
    final_info = {}
    terminated = False
    truncated = False

    print("\n" + "=" * 76)
    print("CASE =", tag)
    print("SEED =", seed, "TARGET_ANGLE_DEG =", angle)

    try:
        env, renderer, obs, actual_seed = helper.make_goal_task(
            make_base,
            v,
            native_mod,
            N,
            int(seed),
            float(angle),
            float(distance)
        )

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if obs.shape != (112,):
            raise RuntimeError(
                "OBS_SHAPE_INVALID: " + repr(obs.shape)
            )

        body_id = choose_body(env, mujoco)

        print("ACTUAL_SEED =", actual_seed)
        print("OBS_SHAPE =", obs.shape)
        print("INITIAL_GOAL_DISTANCE_M =", float(obs[48]) * 10.0)
        print(
            "INITIAL_GOAL_ANGLE_DEG =",
            math.degrees(math.atan2(float(obs[49]), float(obs[50])))
        )

        try:
            previous_yaw = body_yaw(env, body_id)
        except Exception:
            previous_yaw = None

        with trace_path.open("w", encoding="utf-8") as f:
            for step in range(1, 14001):

                # 仅调用冻结 actor，不加入探索噪声、不执行梯度更新。
                with torch.no_grad():
                    raw = agent.act(obs)

                if isinstance(raw, torch.Tensor):
                    raw = raw.detach().cpu().numpy()

                raw = np.asarray(raw, dtype=np.float32).reshape(-1)

                if raw.shape != (2,) or not np.isfinite(raw).all():
                    raise RuntimeError(
                        "INVALID_ACTOR_OUTPUT: " + repr(raw)
                    )

                raw = np.clip(raw, -1.0, 1.0)

                # 严格按 T119 原始代码映射动作。
                action = np.asarray(
                    [
                        (float(raw[0]) + 1.0) / 2.0,
                        float(raw[1])
                    ],
                    dtype=np.float32
                )

                # 必须通过原生物理步进接口。
                next_obs, reward, terminated, truncated, info = (
                    N.step_checked(env, action)
                )

                next_obs = np.asarray(
                    next_obs,
                    dtype=np.float32
                ).reshape(-1)

                if next_obs.shape != (112,):
                    raise RuntimeError(
                        "NEXT_OBS_SHAPE_INVALID"
                    )

                final_info = dict(info)

                goal_sin = float(obs[49])
                goal_cos = float(obs[50])
                goal_angle = math.atan2(goal_sin, goal_cos)
                steer = float(action[1])

                positive += int(steer > 0.05)
                negative += int(steer < -0.05)

                if abs(goal_sin) > 0.05 and abs(steer) > 0.05:
                    goal_sign_total += 1
                    goal_sign_match += int(goal_sin * steer > 0)

                try:
                    yaw = body_yaw(env, body_id)
                except Exception:
                    yaw = None

                yaw_delta = None

                if yaw is not None and previous_yaw is not None:
                    yaw_delta = angle_wrap(yaw - previous_yaw)

                    if abs(steer) > 0.05 and abs(yaw_delta) > 1e-5:
                        yaw_total += 1
                        yaw_sign_match += int(steer * yaw_delta > 0)

                if yaw is not None:
                    previous_yaw = yaw

                collision = bool(info.get("collision", False))
                collisions += int(collision)

                try:
                    native_record = get_last_step_record(env.data)
                except Exception:
                    native_record = None

                pos = None
                if body_id is not None:
                    pos = safe(env.data.xpos[body_id])

                # 每 20 步保留完整观测，事件发生时也强制记录。
                save_obs = (
                    step == 1
                    or step % 20 == 0
                    or collision
                    or bool(info.get("reach_goal", False))
                    or terminated
                    or truncated
                )

                row = {
                    "case": tag,
                    "step": step,
                    "seed": seed,
                    "target_angle_deg": angle,
                    "actor_raw": safe(raw),
                    "executed_action": safe(action),
                    "goal_sin": goal_sin,
                    "goal_cos": goal_cos,
                    "goal_angle_deg": math.degrees(goal_angle),
                    "goal_distance_obs_m": float(obs[48]) * 10.0,
                    "goal_distance_info_m": safe(info.get("goal_dist")),
                    "reward": float(reward),
                    "collision": collision,
                    "reach_goal": bool(info.get("reach_goal", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "body_position": pos,
                    "body_yaw_rad": yaw,
                    "yaw_delta_rad": yaw_delta,
                    "qpos": safe(env.data.qpos[:10]),
                    "qvel": safe(env.data.qvel[:10]),
                    "ctrl": safe(env.data.ctrl),
                    "native_record": safe(native_record),
                    "info": safe(info),
                    "next_observation": (
                        safe(next_obs) if save_obs else None
                    ),
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                steps = step

                if step <= 20 or step % 1000 == 0 or collision:
                    print(
                        "STEP", step,
                        "goal_deg=", round(
                            row["goal_angle_deg"], 2
                        ),
                        "steer=", round(steer, 3),
                        "yaw_delta=", (
                            None if yaw_delta is None
                            else round(yaw_delta, 5)
                        ),
                        "distance=", row["goal_distance_info_m"],
                        "collision=", collision
                    )

                if step % 100 == 0:
                    f.flush()

                if terminated or truncated:
                    break

                obs = next_obs

        success = bool(final_info.get("reach_goal", False))

        if success:
            end = "GOAL"
        elif collisions:
            end = "CONTACT"
        elif truncated:
            end = "TRUNCATED"
        else:
            end = "STEP_CAP"

        summary = {
            "case": tag,
            "seed": int(seed),
            "actual_seed": safe(actual_seed),
            "side": side,
            "angle_deg": float(angle),
            "steps": steps,
            "success": success,
            "collisions": collisions,
            "end": end,
            "remaining_m": safe(final_info.get("goal_dist")),
            "positive_steer_steps": positive,
            "negative_steer_steps": negative,
            "goal_steer_sign_match": (
                goal_sign_match / goal_sign_total
                if goal_sign_total else None
            ),
            "steer_yaw_sign_match": (
                yaw_sign_match / yaw_total
                if yaw_total else None
            ),
            "trace_file": str(trace_path),
        }

        print("CASE_SUMMARY =", json.dumps(summary, ensure_ascii=False))
        return summary

    except Exception as exc:
        print("CASE_ERROR =", repr(exc))
        traceback.print_exc()

        return {
            "case": tag,
            "seed": int(seed),
            "angle_deg": float(angle),
            "steps": steps,
            "error": repr(exc),
            "trace_file": str(trace_path),
        }

    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass


try:
    print("=" * 82)
    print("STAGE T138 — CORRECTED FROZEN POLICY EVALUATION")
    print("TRAINING=FALSE ACTOR_UPDATE=FALSE PHASE_TOUCHED=FALSE")
    print("=" * 82)

    gate(
        "FILES_EXIST",
        TRAIN.is_file() and CKPT.is_file() and STATUS.is_file()
    )

    actual_sha = (
        hashlib.sha256(CKPT.read_bytes()).hexdigest()
        if CKPT.is_file() else ""
    )

    gate("CHECKPOINT_SHA256", actual_sha == EXPECTED_SHA, actual_sha)

    status = json.loads(STATUS.read_text()) if STATUS.is_file() else {}

    gate(
        "STATUS_CHECKPOINT",
        status.get("checkpoint") == "episode_0341.pt"
        and status.get("checkpoint_sha256") == actual_sha,
        str(status.get("state"))
    )

    # 确认没有仍在运行的 T119 训练进程。
    active = []

    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue

        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            args = [x.decode(errors="replace") for x in args if x]
            if len(args) >= 2 and args[1] == str(TRAIN):
                active.append((proc.name, args[:3]))
        except Exception:
            pass

    gate("NO_T119_TRAINING_PROCESS", not active, repr(active))

    source = TRAIN.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)

    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(x, ast.Constant)
            and x.value == "__main__"
            for x in node.test.comparators
        )
        for node in ast.walk(tree)
    )

    gate("SAFE_IMPORT_GUARD", has_main_guard)

    lines = source.splitlines()
    action_region = "\n".join(lines[770:815])

    # 修正：接受原代码中的 float(raw[0])。
    throttle_map_found = (
        bool(re.search(
            r"float\s*\(\s*raw\s*\[\s*0\s*\]\s*\)\s*\+\s*1(?:\.0)?",
            action_region
        ))
        and bool(re.search(r"/\s*2(?:\.0)?", action_region))
    )

    steer_map_found = bool(re.search(
        r"float\s*\(\s*raw\s*\[\s*1\s*\]\s*\)",
        action_region
    ))

    gate("THROTTLE_MAPPING", throttle_map_found)
    gate("STEER_MAPPING", steer_map_found)

    prerequisites = (
        "FILES_EXIST",
        "CHECKPOINT_SHA256",
        "STATUS_CHECKPOINT",
        "NO_T119_TRAINING_PROCESS",
        "SAFE_IMPORT_GUARD",
        "THROTTLE_MAPPING",
        "STEER_MAPPING",
    )

    if not all(GATES.get(k, False) for k in prerequisites):
        print("T138_BLOCKED=TRUE; NO_SIMULATION_STARTED")
    else:
        train_mod = load_module(
            TRAIN,
            "t138_t119_definitions"
        )

        for name in ("HELPER", "RUNNER", "STAGE", "CORE"):
            print("SOURCE_PATH", name, "=", getattr(train_mod, name, None))

        paths_ok = all(
            Path(getattr(train_mod, name)).is_file()
            for name in ("HELPER", "RUNNER", "STAGE", "CORE")
        )

        gate("T119_SOURCE_PATHS", paths_ok)

        if paths_ok:
            helper = load_module(
                train_mod.HELPER,
                "t138_task_helper"
            )
            runner = load_module(
                train_mod.RUNNER,
                "t138_native_runner"
            )
            legacy = load_module(
                train_mod.STAGE,
                "t138_legacy_stage"
            )
            core = load_module(
                train_mod.CORE,
                "t138_td3_core"
            )

            factory_ok = (
                callable(getattr(helper, "make_goal_task", None))
                and callable(getattr(runner, "load_native", None))
                and callable(getattr(runner, "make_task", None))
                and callable(getattr(legacy, "make_short_task", None))
                and callable(getattr(train_mod, "create_agent", None))
            )

            gate("REAL_TASK_FACTORY", factory_ok)

            if factory_ok:
                print("\nACTUAL_TASK_FACTORY_SOURCE:")
                import inspect
                print(inspect.getsource(helper.make_goal_task))

                boot = runner.load_native()

                if not isinstance(boot, tuple) or len(boot) < 5:
                    raise RuntimeError(
                        "NATIVE_BOOTSTRAP_RETURN_SHAPE_INVALID"
                    )

                _, v, native_mod, prep, hashes = boot
                N = prep[3]

                make_base = legacy.make_short_task(
                    runner.make_task
                )

                agent = train_mod.create_agent(core)

                checkpoint = torch.load(
                    CKPT,
                    map_location="cpu",
                    weights_only=False
                )

                gate(
                    "CHECKPOINT_FRAME_COUNT",
                    int(checkpoint.get("frames", -1)) == 3393494
                    and int(checkpoint.get("episode_index", -1)) == 341
                )

                if GATES.get("CHECKPOINT_FRAME_COUNT", False):
                    agent.actor.load_state_dict(checkpoint["actor"])
                    agent.actor.eval()

                    actor_before = {
                        k: x.detach().cpu().clone()
                        for k, x in agent.actor.state_dict().items()
                    }

                    import mujoco

                    from training.e2e_phase10_lidar_v7_curriculum.stage9_runtime_fullplant_fix_v5_2 import (
                        get_last_step_record
                    )

                    out_dir = JOB / (
                        "t138_corrected_eval_"
                        + time.strftime("%Y%m%d_%H%M%S")
                    )
                    out_dir.mkdir(parents=True, exist_ok=False)

                    print("EVIDENCE_DIR =", out_dir)

                    # 每个种子都测试左右目标。
                    # 先比较同种子，再比较另一个已知高碰撞种子。
                    cases = [
                        (2029653601, -17.3),
                        (2029653601,  18.8),
                        (2029654802, -17.3),
                        (2029654802,  18.8),
                    ]

                    EVALUATION_STARTED = True

                    for seed, angle in cases:
                        result = run_case(
                            helper,
                            make_base,
                            v,
                            native_mod,
                            N,
                            agent,
                            seed,
                            angle,
                            3.0,
                            out_dir,
                            mujoco,
                            get_last_step_record,
                        )

                        SUMMARIES.append(result)

                    actor_after = agent.actor.state_dict()

                    actor_unchanged = all(
                        torch.equal(
                            actor_before[k],
                            actor_after[k].detach().cpu()
                        )
                        for k in actor_before
                    )

                    gate("ACTOR_UNCHANGED", actor_unchanged)

                    result_path = out_dir / "summary.json"
                    result_path.write_text(
                        json.dumps(
                            {
                                "experiment": "STAGE_T138_CORRECTED_FROZEN_EVAL",
                                "checkpoint": str(CKPT),
                                "checkpoint_sha256": actual_sha,
                                "training": False,
                                "actor_updated": False,
                                "actor_unchanged": actor_unchanged,
                                "cases": SUMMARIES,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    print("SUMMARY_FILE =", result_path)

except Exception as exc:
    print("T138_ERROR =", repr(exc))
    traceback.print_exc()

print("\n" + "=" * 82)
print("T138_FINISHED")
print("EVALUATION_STARTED =", EVALUATION_STARTED)
print("TRAINING_STARTED=FALSE")
print("ACTOR_UPDATE=FALSE")
print("PHASE_TOUCHED=FALSE")
print("GATES =", json.dumps(GATES, ensure_ascii=False))
print("=" * 82)
