from pathlib import Path
from collections import deque
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import mujoco

HOME = Path.home()
JOB = Path(os.environ["TD3_JOB"])

STAGE = (
    HOME
    / "mkmini_stage_c147_fixedgoal_sac_20261027_v1"
    / "GOAL_COMMIT_24000/train/train.py"
)

CORE = (
    HOME
    / "mkmini_stage_td3_adapter_20261027_v1"
    / "td3_core.py"
)

FROZEN = (
    HOME
    / "mkmini_stage_c122s0_fullcap_20261026_125601_1360056_v1"
    / "train/work_resume.pt"
)

EXPECTED = {
    STAGE: "4e9855bd90f523fb5c7c12ac4ea5c7ca5a7242cd08a63808ef3b3c89120c1556",
    CORE: "e78f7f981b3fdc9d576bbc08b0222d42dc1a599865796180a4b45b1adcf361b8",
    FROZEN: "cb650414dab55dbc0a913cc439fe947a7979fd0755f5aff3912fda68c5ab35d8",
}

MAX_PHYSICS = 2000
MAX_SECONDS = 3000


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def emit(key, data):
    print(
        key,
        json.dumps(data, ensure_ascii=False, default=str),
        flush=True
    )


def status(data):
    target = JOB / "status.json"
    temp = JOB / "status.json.partial"
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str)
    )
    os.replace(temp, target)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    check(
        spec is not None and spec.loader is not None,
        "MODULE_LOAD_SPEC_INVALID"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def preflight():
    for path, expected in EXPECTED.items():
        check(path.is_file(), "MISSING:" + str(path))
        check(
            sha(path) == expected,
            "SHA256_MISMATCH:" + str(path)
        )

    import ast
    ast.parse(STAGE.read_text())
    ast.parse(CORE.read_text())

    source = STAGE.read_text()

    for key in (
        "def make_short_task(",
        "N.create(",
        "runner.load_native()",
        "env._goal_state()",
    ):
        check(key in source, "STAGE_INTERFACE_MISSING:" + key)

    check(
        not (JOB / "td3_t1_research_only.pt").exists(),
        "EXISTING_TD3_OUTPUT"
    )

    emit("TD3_INDEPENDENT_PREFLIGHT=PASS", {
        "stage_sha256": sha(STAGE),
        "td3_core_sha256": sha(CORE),
        "frozen_parent_sha256": sha(FROZEN),
        "physics_budget": MAX_PHYSICS,
        "wall_budget_seconds": MAX_SECONDS,
        "sac_resume_required": False,
        "stage_only": True,
        "automatic_promotion": False
    })


class UnsafeGoal(Exception):
    pass


def make_goal_task(make_base, v, native_mod, N,
                   seed, angle, distance):

    # 沿用Stage原来的安全起点和环境创建方法。
    # 这里只设置最终目标，不生成中间路点。
    for retry in range(8):
        actual_seed = (
            seed if retry == 0
            else seed + 101 + retry * 41
        )

        env = None
        renderer = None

        try:
            env, renderer, obs, depth, relocated = make_base(
                v, native_mod, N,
                actual_seed, angle,
                cleared=False,
                limit=4600
            )

            start = np.asarray(
                env.data.qpos[:2],
                dtype=np.float64
            ).copy()

            yaw = float(env._pose()[2])
            direction = yaw + math.radians(angle)

            goal = start + distance * np.array(
                [
                    math.cos(direction),
                    math.sin(direction)
                ],
                dtype=np.float64
            )

            q0 = env.data.qpos.copy()
            v0 = env.data.qvel.copy()
            ctrl0 = env.data.ctrl.copy()

            try:
                # 仅检查最终目标位置的车体几何碰撞。
                env.data.qpos[:2] = goal
                mujoco.mj_forward(env.model, env.data)

                goal_collides = bool(env._collision())

            finally:
                env.data.qpos[:] = q0
                env.data.qvel[:] = v0

                if env.data.ctrl.size:
                    env.data.ctrl[:] = ctrl0

                mujoco.mj_forward(env.model, env.data)

            if goal_collides or env._collision():
                raise UnsafeGoal("GOAL_OR_START_COLLIDES")

            # 关键修复：把安全目标真正写入训练环境。
            env.goal_xy = goal.copy()
            env.prev_goal_dist = float(
                env._goal_state()[1]
            )

            actual_distance = float(
                env._goal_state()[1]
            )

            actual_angle = math.degrees(
                float(env._goal_state()[2])
            )

            check(
                abs(actual_distance - distance) < 0.03,
                "TD3_ACTUAL_GOAL_DISTANCE_MISMATCH"
            )

            bearing_error = (
                actual_angle - angle + 180
            ) % 360 - 180

            check(
                abs(bearing_error) < 2.0,
                "TD3_ACTUAL_GOAL_ANGLE_MISMATCH"
            )

            obs = np.asarray(
                env._observation(update_history=False),
                dtype=np.float32
            ).reshape(-1)

            check(
                obs.size == 112 and np.isfinite(obs).all(),
                "TD3_OBSERVATION_INVALID"
            )

            emit("TD3_T1_GOAL_GATE=", {
                "seed_requested": seed,
                "seed_actual": actual_seed,
                "retry": retry,
                "angle_deg": angle,
                "requested_m": distance,
                "actual_m": actual_distance,
                "observation_dim": int(obs.size),
                "goal_collision": False
            })

            return env, renderer, obs, actual_seed

        except UnsafeGoal:
            emit("TD3_T1_GOAL_RETRY=", {
                "seed": actual_seed,
                "retry": retry
            })

            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()

        except Exception:
            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()
            raise

    raise RuntimeError("TD3_NO_SAFE_GOAL_AFTER_RETRIES")


def smoke():
    check(
        not (JOB / "td3_t1_research_only.pt").exists(),
        "TD3_ALREADY_COMPLETED"
    )

    os.environ["C98_JOB"] = str(JOB)
    os.environ["C22_SCREEN_ONLY"] = "0"

    # 重要：仅导入Stage环境模块，绝不执行原SAC main()。
    legacy = load_module(
        STAGE, "stage_td3_environment_reference"
    )

    for path, expected in legacy.EXPECTED.items():
        check(
            path.is_file() and legacy.sha(path) == expected,
            "STAGE_NATIVE_DEPENDENCY_INVALID:" + str(path)
        )

    runner = legacy.module(
        legacy.SRC, "stage_td3_native_runner"
    )

    p2, v, native_mod, prep, hashes = runner.load_native()

    check(
        isinstance(hashes, dict),
        "NATIVE_SOURCE_MANIFEST_INVALID"
    )

    check(
        len(prep) >= 4,
        "STAGE_PREP_INCOMPLETE"
    )

    N = prep[3]

    check(
        callable(N.step_checked),
        "STAGE_STEP_INTERFACE_INVALID"
    )

    make_base = legacy.make_short_task(
        runner.make_task
    )

    td3mod = load_module(
        CORE, "stage_td3_verified_core"
    )

    torch.manual_seed(20261027)
    rng = np.random.default_rng(20261027)

    agent = td3mod.TD3(
        state_dim=112,
        action_dim=2,
        device="cpu"
    )

    replay = deque(maxlen=20000)

    deadline = time.monotonic() + MAX_SECONDS
    physics = 0
    decisions = 0
    episodes = []
    losses = []

    emit("TD3_INDEPENDENT_BOOTSTRAP=PASS", {
        "legacy_sac_main_called": False,
        "sac_checkpoint_loaded": False,
        "td3_actor_initialized": True,
        "td3_critic_initialized": True,
        "observation_dim": 112,
        "action_dim": 2
    })

    tasks = (
        (2029461000, -30.0, 0.8),
        (2029461001, 30.0, 0.8)
    )

    for seed, angle, distance in tasks:
        env = None
        renderer = None

        try:
            env, renderer, obs, actual_seed = make_goal_task(
                make_base, v, native_mod, N,
                seed, angle, distance
            )

            used = 0
            local_decisions = 0
            contact = False
            success = False
            end_reason = "BUDGET"

            while (
                used + 100 <= 900
                and physics + 100 <= MAX_PHYSICS
            ):
                check(
                    time.monotonic() < deadline,
                    "TD3_WALL_BUDGET_EXCEEDED"
                )

                if decisions < 32:
                    # 初始动作是随机探索，不是导航控制器。
                    raw = np.array([
                        rng.uniform(-0.6, 0.3),
                        rng.uniform(-0.8, 0.8)
                    ], dtype=np.float32)
                else:
                    raw = np.clip(
                        agent.act(obs)
                        + rng.normal(0, 0.12, 2),
                        -1.0, 1.0
                    ).astype(np.float32)

                # MK-mini控制接口：[油门0~1, 转向-1~1]
                action = np.array([
                    (float(raw[0]) + 1.0) / 2.0,
                    float(raw[1])
                ], dtype=np.float32)

                t0 = float(env.data.time)

                next_obs, reward, terminated, truncated, info = (
                    N.step_checked(env, action)
                )

                timestep = float(
                    env.model.opt.timestep
                )

                nframe = round(
                    (float(env.data.time) - t0) / timestep
                )

                check(
                    1 <= nframe <= 100,
                    "TD3_PHYSICS_FRAME_DELTA_INVALID:"
                    + str(nframe)
                )

                next_obs = np.asarray(
                    next_obs, dtype=np.float32
                ).reshape(-1)

                check(
                    next_obs.size == 112
                    and np.isfinite(next_obs).all()
                    and math.isfinite(float(reward)),
                    "TD3_TRANSITION_INVALID"
                )

                contact = bool(info.get("collision", False))
                success = bool(info.get("reach_goal", False))

                # 接触或成功为终止；
                # 时间截断不作为价值网络的真实终止状态。
                terminal = bool(
                    terminated or contact or success
                )

                replay.append((
                    obs.copy(),
                    raw.copy(),
                    float(reward),
                    next_obs.copy(),
                    float(terminal)
                ))

                obs = next_obs
                physics += nframe
                used += nframe
                decisions += 1
                local_decisions += 1

                if len(replay) >= 32 and decisions % 2 == 0:
                    ids = rng.choice(
                        len(replay), size=32, replace=False
                    )

                    samples = [
                        replay[int(i)] for i in ids
                    ]

                    batch = tuple(
                        np.asarray(
                            [row[k] for row in samples],
                            dtype=np.float32
                        )
                        for k in range(5)
                    )

                    result = agent.update(batch)

                    check(
                        math.isfinite(result["critic_loss"]),
                        "TD3_CRITIC_LOSS_NONFINITE"
                    )

                    losses.append(
                        result["critic_loss"]
                    )

                if decisions % 10 == 0:
                    emit("TD3_T1_PROGRESS=", {
                        "physics_frames": physics,
                        "transitions": len(replay),
                        "updates": agent.total_updates,
                        "current_remaining_m": float(
                            env._goal_state()[1]
                        ),
                        "contact": contact,
                        "goal": success
                    })

                    status({
                        "state": "RUNNING",
                        "physics_frames": physics,
                        "transitions": len(replay),
                        "updates": agent.total_updates,
                        "episodes": episodes,
                        "automatic_promotion": False
                    })

                if terminal or truncated:
                    end_reason = (
                        "GOAL" if success
                        else "CONTACT" if contact
                        else "TRUNCATED"
                    )
                    break

            row = {
                "seed": actual_seed,
                "angle_deg": angle,
                "requested_distance_m": distance,
                "physics_frames": used,
                "decisions": local_decisions,
                "remaining_m": float(
                    env._goal_state()[1]
                ),
                "end": end_reason,
                "contact": contact,
                "goal": success
            }

            episodes.append(row)
            emit("TD3_T1_EPISODE=", row)

        finally:
            if renderer is not None:
                renderer.close()
            if env is not None:
                env.close()

    check(
        len(episodes) == 2,
        "TD3_EPISODE_COUNT_INVALID"
    )

    check(
        0 < physics <= MAX_PHYSICS,
        "TD3_PHYSICS_BUDGET_INVALID"
    )

    check(
        agent.total_updates > 0,
        "TD3_NO_REAL_DATA_UPDATES"
    )

    check(
        sha(FROZEN) == EXPECTED[FROZEN],
        "FROZEN_PARENT_CHANGED"
    )

    checkpoint = JOB / "td3_t1_research_only.pt"
    temporary = JOB / "td3_t1_research_only.pt.partial"

    # 仅保存独立TD3研究检查点。
    # 同时保存优化器和经验回放，供后续续训使用。
    torch.save({
        "algorithm": "TD3",
        "actor": agent.actor.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "updates": agent.total_updates,
        "physics_frames": physics,
        "decisions": decisions,
        "replay": list(replay),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "stage_source_sha256": sha(STAGE),
        "td3_core_sha256": sha(CORE),
        "automatic_promotion": False
    }, temporary)

    os.replace(temporary, checkpoint)

    report = {
        "state": "FUNCTIONAL_PASS",
        "physics_frames": physics,
        "transitions": len(replay),
        "td3_updates": agent.total_updates,
        "mean_critic_loss": float(np.mean(losses)),
        "episodes": episodes,
        "checkpoint_sha256": sha(checkpoint),
        "frozen_parent_verified": True,
        "scientific_performance_confirmed": False,
        "automatic_promotion": False
    }

    status(report)
    emit("TD3_T1_FUNCTIONAL_PASS=", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke"),
        required=True
    )

    args = parser.parse_args()

    JOB.mkdir(parents=True, exist_ok=True)

    preflight()

    if args.mode == "smoke":
        smoke()


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        emit("TD3_T1_BLOCKED=", {
            "error": repr(error)
        })

        try:
            status({
                "state": "BLOCKED",
                "error": repr(error),
                "automatic_promotion": False
            })
        except Exception:
            pass

        traceback.print_exc()
        sys.exit(1)

    finally:
        print(
            "INTERACTIVE_TERMINAL_REMAINS_OPEN=TRUE",
            flush=True
        )
