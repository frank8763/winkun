from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.train_stage2a1_safety_reward import (
    set_learning_rate,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a13_terminal_braking_fix import (
    CAUTION_COLLISION_SEEDS,
    LIGHT_COLLISION_SEEDS,
    evaluate_exact_seeds,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a17_terminal_chatter_fix import (
    FOCUS_SEED,
    evaluate_focus_trace,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage4_acrp_sac_overnight import (
    SOURCE_MODEL,
    DistanceStage,
    Stage4CurriculumEnv,
    atomic_save_json,
    freeze_actor_with_hooks,
    make_fresh_model,
    set_action_noise,
)


OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage5_terminal_curriculum_daytime"
)
CONFIRMED_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/confirmed"
)
STATE_PATH = OUTPUT_ROOT / "daytime_state.json"
SUMMARY_PATH = OUTPUT_ROOT / "daytime_summary.json"

DEFAULT_HOURS = 6.0
INITIAL_CRITIC_WARMUP = 20_000
CRITIC_RESET_WARMUP = 12_000
CHUNK_TIMESTEPS = 30_000
CURRENT_EVAL_EPISODES = 36
PASS_CONFIRMATIONS_REQUIRED = 2

INITIAL_LR = 6.0e-6
MIN_LR = 1.2e-6
MAX_LR = 8.0e-6

INITIAL_NOISE = np.asarray([0.0020, 0.0040], dtype=np.float32)
MIN_NOISE = np.asarray([0.0007, 0.0014], dtype=np.float32)
MAX_NOISE = np.asarray([0.0045, 0.0090], dtype=np.float32)

MAX_COLLISION_SEEDS = 24
MAX_TIMEOUT_SEEDS = 12


# First rebuild the missing bridge between the old 0.75-1.00 m policy and S1.
# S2 is entered only after S1 passes two independent evaluations.
DAY_STAGES: tuple[DistanceStage, ...] = (
    DistanceStage(
        name="B0_0p75_1p05_terminal",
        goal_min=0.75,
        goal_max=1.05,
        angle_deg=20.0,
        max_steps=1050,
        clearance_margin=0.45,
        stage_budget_minutes=60,
        success_gate=0.60,
        collision_gate=0.10,
        progress_gate=0.72,
        speed_gate=0.035,
        timeout_gate=0.35,
    ),
    DistanceStage(
        name="B1_0p90_1p20_terminal",
        goal_min=0.90,
        goal_max=1.20,
        angle_deg=25.0,
        max_steps=1200,
        clearance_margin=0.43,
        stage_budget_minutes=65,
        success_gate=0.50,
        collision_gate=0.12,
        progress_gate=0.68,
        speed_gate=0.035,
        timeout_gate=0.42,
    ),
    DistanceStage(
        name="B2_1p00_1p30_terminal",
        goal_min=1.00,
        goal_max=1.30,
        angle_deg=30.0,
        max_steps=1350,
        clearance_margin=0.41,
        stage_budget_minutes=70,
        success_gate=0.40,
        collision_gate=0.15,
        progress_gate=0.62,
        speed_gate=0.035,
        timeout_gate=0.50,
    ),
    DistanceStage(
        name="S1_1p00_1p50_terminal",
        goal_min=1.00,
        goal_max=1.50,
        angle_deg=35.0,
        max_steps=1550,
        clearance_margin=0.40,
        stage_budget_minutes=85,
        success_gate=0.35,
        collision_gate=0.15,
        progress_gate=0.58,
        speed_gate=0.035,
        timeout_gate=0.55,
    ),
    DistanceStage(
        name="S2_1p50_2p00_terminal",
        goal_min=1.50,
        goal_max=2.00,
        angle_deg=50.0,
        max_steps=2050,
        clearance_margin=0.39,
        stage_budget_minutes=90,
        success_gate=0.20,
        collision_gate=0.18,
        progress_gate=0.48,
        speed_gate=0.032,
        timeout_gate=0.70,
    ),
)


def stage_dir(stage: DistanceStage) -> Path:
    return OUTPUT_ROOT / "stages" / stage.name


def clamp_entropy(
    model: SAC,
    minimum: float,
    maximum: float = 0.025,
) -> float:
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if log_ent_coef is None:
        return minimum

    with torch.no_grad():
        log_ent_coef.clamp_(
            min=math.log(float(minimum)),
            max=math.log(float(maximum)),
        )
    return float(torch.exp(log_ent_coef.detach()).cpu().item())


def configure_actor(model: SAC, stage_index: int) -> list[str]:
    # Keep the proven LiDAR representation frozen.
    for parameter in model.actor.parameters():
        parameter.requires_grad = False

    model.actor.mu.weight.requires_grad = True
    model.actor.mu.bias.requires_grad = True

    if stage_index >= 2:
        linear_layers = [
            (name, module)
            for name, module in model.actor.latent_pi.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("actor.latent_pi has no Linear layer")
        _, last_linear = linear_layers[-1]
        for parameter in last_linear.parameters():
            parameter.requires_grad = True

    return [
        name
        for name, parameter in model.actor.named_parameters()
        if parameter.requires_grad
    ]


class TerminalCurriculumEnv(Stage4CurriculumEnv):
    """End-to-end SAC environment with bounded progress economics.

    The policy still maps LiDAR/goal/history directly to speed and steering.
    No planner, safety shield or hand-written action override is used.
    """

    def __init__(
        self,
        stage: DistanceStage,
        previous_stage: DistanceStage | None = None,
        fixed_task: str | None = None,
        seed: int = 101000,
    ) -> None:
        super().__init__(
            stage=stage,
            previous_stage=previous_stage,
            fixed_task=fixed_task,
            seed=seed,
        )

        # Stage4 used progress_scale=18 and collision=-10. Reaching most of a
        # 1.5 m goal before collision could therefore be profitable.
        self.progress_scale = 6.0
        self.risk_scale = 0.14
        self.speed_risk_scale = 0.17
        self.safe_cruise_scale = 0.006
        self.idle_scale = 0.012

        self.terminal_zone = 0.62
        self.terminal_min_speed = 0.022
        self.terminal_entry_speed = 0.105
        self.terminal_overspeed_scale = 0.22
        self.terminal_brake_bonus_scale = 0.018
        self.risk_speed_scale = 0.12

        self._current_stage_probability = 0.68
        self._previous_stage_probability = (
            0.12 if previous_stage is not None else 0.0
        )
        self._hard_seed_probability = 0.16

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            action
        )

        goal_distance = float(info.get("goal_dist", 999.0))
        goal_angle = float(info.get("goal_angle", 0.0))
        speed = float(info.get("actual_speed_mps", 0.0))
        progress = float(info.get("stage4_progress", 0.0))
        risk_ratio = float(info.get("stage4_risk_ratio", 0.0))
        collision = bool(info.get("collision", False))
        success = bool(info.get("reach_goal", False))
        timeout = bool(info.get("timeout", False))

        success_radius = float(getattr(self, "success_radius", 0.30))
        denominator = max(
            self.terminal_zone - success_radius,
            1e-4,
        )
        terminal_ratio = float(
            np.clip(
                (self.terminal_zone - goal_distance) / denominator,
                0.0,
                1.0,
            )
        )
        target_speed = float(
            self.terminal_entry_speed * (1.0 - terminal_ratio)
            + self.terminal_min_speed * terminal_ratio
        )

        overspeed = max(0.0, speed - target_speed)
        terminal_overspeed_penalty = (
            -self.terminal_overspeed_scale
            * terminal_ratio
            * float(np.clip(overspeed / 0.06, 0.0, 2.0)) ** 2
        )

        terminal_brake_bonus = 0.0
        if (
            terminal_ratio > 0.0
            and progress > 0.0
            and speed <= target_speed + 0.012
        ):
            alignment = float(
                np.clip(math.cos(goal_angle), 0.0, 1.0)
            )
            terminal_brake_bonus = (
                self.terminal_brake_bonus_scale
                * terminal_ratio
                * alignment
            )

        # Penalize excessive speed close to sensed obstacles without changing
        # the action. This remains reward shaping for an end-to-end policy.
        safe_speed_limit = 0.035 + 0.115 * (1.0 - risk_ratio)
        risk_overspeed = max(0.0, speed - safe_speed_limit)
        risk_speed_penalty = (
            -self.risk_speed_scale
            * risk_ratio
            * float(np.clip(risk_overspeed / 0.08, 0.0, 2.0)) ** 2
        )

        initial_distance = float(
            max(getattr(self, "_initial_goal_distance", 1.0), 0.5)
        )

        # Correct Stage4's terminal economics:
        # base collision=-10, success=+12, timeout=-0.8.
        terminal_correction = 0.0
        if collision:
            desired_collision_penalty = -(24.0 + 8.0 * initial_distance)
            terminal_correction += desired_collision_penalty - (-10.0)
            if terminal_ratio > 0.0:
                terminal_correction -= 5.0 * terminal_ratio
        elif success:
            desired_success_bonus = 22.0 + 3.0 * initial_distance
            terminal_correction += desired_success_bonus - 12.0
            if speed <= target_speed + 0.018:
                terminal_correction += 3.0
        elif timeout:
            terminal_correction += -2.0 - (-0.8)

        reward = float(
            reward
            + terminal_overspeed_penalty
            + terminal_brake_bonus
            + risk_speed_penalty
            + terminal_correction
        )

        extra_priority = float(
            7.0 * float(collision)
            + 4.0 * terminal_ratio * float(collision)
            + 2.0 * terminal_ratio
        )
        info["risk_priority"] = float(
            info.get("risk_priority", 1.0) + extra_priority
        )
        info["stage5_terminal_ratio"] = terminal_ratio
        info["stage5_target_speed"] = target_speed
        info["stage5_terminal_overspeed_penalty"] = (
            terminal_overspeed_penalty
        )
        info["stage5_terminal_brake_bonus"] = terminal_brake_bonus
        info["stage5_risk_speed_penalty"] = risk_speed_penalty
        info["stage5_terminal_correction"] = terminal_correction
        info["stage5_reward"] = reward

        return observation, reward, terminated, truncated, info


def evaluate_task(
    model: SAC,
    stage: DistanceStage,
    previous_stage: DistanceStage | None,
    task: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = TerminalCurriculumEnv(
        stage=stage,
        previous_stage=previous_stage,
        fixed_task=task,
        seed=seed + 5000,
    )
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            actual_seed = seed + episode
            observation, reset_info = env.reset(seed=actual_seed)
            initial_distance = float(
                reset_info.get("goal_dist", 999.0)
            )

            final_info: dict[str, Any] = {}
            speed_sum = 0.0
            path_length = 0.0
            steps = 0
            last_speed = 0.0

            for step in range(env.max_steps):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                observation, _, terminated, truncated, info = env.step(
                    action
                )
                final_info = dict(info)
                steps = step + 1
                last_speed = float(info.get("actual_speed_mps", 0.0))
                speed_sum += last_speed
                path_length += last_speed * float(
                    max(getattr(env, "dt", 0.02), 1e-6)
                )
                if terminated or truncated:
                    break

            success = bool(final_info.get("reach_goal", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(
                not success
                and not collision
                and steps >= env.max_steps
            )
            final_distance = float(
                final_info.get("goal_dist", 999.0)
            )
            progress_ratio = float(
                np.clip(
                    (initial_distance - final_distance)
                    / max(initial_distance, 1e-6),
                    -1.0,
                    1.0,
                )
            )
            near_goal_collision = bool(
                collision and final_distance <= 0.65
            )

            rows.append(
                {
                    "seed": actual_seed,
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "near_goal_collision": near_goal_collision,
                    "steps": steps,
                    "initial_goal_distance": initial_distance,
                    "final_goal_distance": final_distance,
                    "progress_ratio": progress_ratio,
                    "mean_speed_mps": speed_sum / max(steps, 1),
                    "terminal_speed_mps": last_speed,
                    "path_length": path_length,
                }
            )
    finally:
        env.close()

    count = len(rows)
    collisions = [row for row in rows if row["collision"]]
    result = {
        "task": task,
        "episodes": count,
        "success_rate": sum(int(r["success"]) for r in rows) / count,
        "collision_rate": sum(int(r["collision"]) for r in rows) / count,
        "timeout_rate": sum(int(r["timeout"]) for r in rows) / count,
        "near_goal_collision_rate": (
            sum(int(r["near_goal_collision"]) for r in rows) / count
        ),
        "mean_progress_ratio": float(
            np.mean([r["progress_ratio"] for r in rows])
        ),
        "mean_speed_mps": float(
            np.mean([r["mean_speed_mps"] for r in rows])
        ),
        "mean_final_goal_distance": float(
            np.mean([r["final_goal_distance"] for r in rows])
        ),
        "collision_mean_final_goal_distance": (
            float(
                np.mean(
                    [r["final_goal_distance"] for r in collisions]
                )
            )
            if collisions
            else None
        ),
        "collision_mean_terminal_speed_mps": (
            float(
                np.mean(
                    [r["terminal_speed_mps"] for r in collisions]
                )
            )
            if collisions
            else None
        ),
        "collision_seeds": [
            int(r["seed"]) for r in rows if r["collision"]
        ],
        "timeout_seeds": [
            int(r["seed"]) for r in rows if r["timeout"]
        ],
        "results": rows,
    }

    print(f"\n========== EVAL {task} {count} ==========")
    for key in (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "near_goal_collision_rate",
        "mean_progress_ratio",
        "mean_speed_mps",
        "mean_final_goal_distance",
        "collision_mean_final_goal_distance",
        "collision_mean_terminal_speed_mps",
    ):
        print(key.upper(), "=", result[key])
    return result


def evaluate_stage(
    model: SAC,
    stage: DistanceStage,
    previous_stage: DistanceStage | None,
    seed_offset: int,
) -> dict[str, Any]:
    current = evaluate_task(
        model,
        stage,
        previous_stage,
        stage.name,
        CURRENT_EVAL_EPISODES,
        1_000_000 + seed_offset,
    )
    a3 = evaluate_task(
        model,
        stage,
        previous_stage,
        "a3",
        12,
        1_100_000 + seed_offset,
    )
    light = evaluate_task(
        model,
        stage,
        previous_stage,
        "light",
        15,
        1_200_000 + seed_offset,
    )

    hard_caution = evaluate_exact_seeds(
        model,
        "caution",
        CAUTION_COLLISION_SEEDS,
    )
    hard_light = evaluate_exact_seeds(
        model,
        "light",
        LIGHT_COLLISION_SEEDS,
    )
    focus = evaluate_focus_trace(model, seed=FOCUS_SEED)

    hard_collision_count = (
        int(hard_caution["collision_count"])
        + int(hard_light["collision_count"])
    )
    historical_safety = bool(
        hard_collision_count == 0
        and not bool(focus["collision"])
    )
    retention_safety = bool(
        a3["collision_rate"] <= 0.05
        and light["collision_rate"] <= 0.05
        and a3["success_rate"] >= 0.55
        and light["success_rate"] >= 0.55
    )

    gate = bool(
        current["success_rate"] >= stage.success_gate
        and current["collision_rate"] <= stage.collision_gate
        and current["mean_progress_ratio"] >= stage.progress_gate
        and current["mean_speed_mps"] >= stage.speed_gate
        and current["timeout_rate"] <= stage.timeout_gate
        and historical_safety
        and retention_safety
    )

    result = {
        "stage": asdict(stage),
        "task": stage.name,
        "success_rate": current["success_rate"],
        "collision_rate": current["collision_rate"],
        "timeout_rate": current["timeout_rate"],
        "near_goal_collision_rate": current[
            "near_goal_collision_rate"
        ],
        "progress_ratio": current["mean_progress_ratio"],
        "speed_mps": current["mean_speed_mps"],
        "mean_final_goal_distance": current[
            "mean_final_goal_distance"
        ],
        "current": current,
        "a3": a3,
        "light": light,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "historical_safety": historical_safety,
        "retention_safety": retention_safety,
        "gate_passed": gate,
    }

    print(f"\n========== STAGE SUMMARY {stage.name} ==========")
    print("GATE_PASSED =", gate)
    print("HISTORICAL_SAFETY =", historical_safety)
    print("RETENTION_SAFETY =", retention_safety)
    return result


def candidate_key(result: dict[str, Any]) -> tuple[Any, ...]:
    # Strict lexicographic ordering: safety, actual success, then collision.
    return (
        0 if result["historical_safety"] else 1,
        0 if result["retention_safety"] else 1,
        -float(result["success_rate"]),
        float(result["collision_rate"]),
        float(result["timeout_rate"]),
        -float(result["progress_ratio"]),
        -float(result["speed_mps"]),
    )


def promotion_allowed(
    candidate: dict[str, Any],
    best: dict[str, Any],
) -> tuple[bool, str]:
    if not candidate["historical_safety"]:
        return False, "HISTORICAL_SAFETY_FAILED"
    if not candidate["retention_safety"]:
        return False, "RETENTION_SAFETY_FAILED"

    c_success = float(candidate["success_rate"])
    c_collision = float(candidate["collision_rate"])
    c_timeout = float(candidate["timeout_rate"])
    c_progress = float(candidate["progress_ratio"])
    c_speed = float(candidate["speed_mps"])

    b_success = float(best["success_rate"])
    b_progress = float(best["progress_ratio"])
    b_speed = float(best["speed_mps"])

    if (
        c_success <= 1e-9
        and c_timeout >= 0.90
        and (c_progress < 0.25 or c_speed < 0.020)
    ):
        return False, "DEAD_TIMEOUT_POLICY"

    if c_success <= 1e-9 and c_collision >= 0.80:
        return False, "ALL_COLLISION_POLICY"

    if b_success > 0.0 and c_success <= 1e-9:
        return False, "LOST_EXISTING_SUCCESS"

    if (
        c_success <= b_success
        and c_progress < 0.70 * max(b_progress, 1e-6)
        and c_speed < 0.65 * max(b_speed, 1e-6)
    ):
        return False, "LOCOMOTION_COLLAPSE"

    return True, "PROMOTION_ALLOWED"


def update_seed_pool(
    evaluation: dict[str, Any],
    collision_seeds: list[int],
    timeout_seeds: list[int],
) -> tuple[list[int], list[int]]:
    current = evaluation["current"]

    collision_seeds = (
        collision_seeds
        + [int(v) for v in current["collision_seeds"]]
    )[-MAX_COLLISION_SEEDS:]
    timeout_seeds = (
        timeout_seeds
        + [int(v) for v in current["timeout_seeds"]]
    )[-MAX_TIMEOUT_SEEDS:]

    return list(dict.fromkeys(collision_seeds)), list(
        dict.fromkeys(timeout_seeds)
    )


def choose_mode(result: dict[str, Any]) -> str:
    collision = float(result["collision_rate"])
    timeout = float(result["timeout_rate"])
    near_goal_collision = float(
        result["near_goal_collision_rate"]
    )
    progress = float(result["progress_ratio"])
    speed = float(result["speed_mps"])

    if collision >= 0.35 and near_goal_collision >= 0.20:
        return "TERMINAL_BRAKING"
    if collision >= 0.35:
        return "OBSTACLE_SAFETY"
    if timeout >= 0.70 and (progress < 0.35 or speed < 0.025):
        return "ANTI_STALL"
    return "BALANCED"


def apply_mode(
    env: TerminalCurriculumEnv,
    model: SAC,
    mode: str,
    learning_rate: float,
    noise: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    entropy_floor = 0.004

    if mode == "TERMINAL_BRAKING":
        learning_rate = max(MIN_LR, learning_rate * 0.82)
        noise = np.maximum(MIN_NOISE, noise * 0.82)
        env.terminal_overspeed_scale = min(
            0.42, env.terminal_overspeed_scale * 1.18
        )
        env.terminal_brake_bonus_scale = min(
            0.030, env.terminal_brake_bonus_scale * 1.10
        )
        env.risk_speed_scale = min(
            0.20, env.risk_speed_scale * 1.08
        )
        env._hard_seed_probability = 0.20
        entropy_floor = 0.0035

    elif mode == "OBSTACLE_SAFETY":
        learning_rate = max(MIN_LR, learning_rate * 0.78)
        noise = np.maximum(MIN_NOISE, noise * 0.75)
        env.risk_scale = min(0.22, env.risk_scale * 1.12)
        env.speed_risk_scale = min(
            0.24, env.speed_risk_scale * 1.12
        )
        env._hard_seed_probability = 0.22
        entropy_floor = 0.0030

    elif mode == "ANTI_STALL":
        learning_rate = min(MAX_LR, learning_rate * 1.05)
        noise = np.minimum(MAX_NOISE, noise * 1.15)
        env.idle_scale = min(0.024, env.idle_scale * 1.12)
        env.safe_cruise_scale = min(
            0.012, env.safe_cruise_scale * 1.10
        )
        env._hard_seed_probability = 0.10
        entropy_floor = 0.0060

    else:
        env._hard_seed_probability = 0.16
        entropy_floor = 0.0040

    set_learning_rate(model, learning_rate)
    set_action_noise(model, noise)
    clamp_entropy(model, entropy_floor)

    return learning_rate, noise, entropy_floor


def save_checkpoint(
    model: SAC,
    model_path: Path,
    replay_path: Path,
) -> None:
    model.save(str(model_path.with_suffix("")))
    if model.replay_buffer is not None:
        model.save_replay_buffer(str(replay_path))


def load_checkpoint(
    model_path: Path,
    replay_path: Path,
    env: Monitor,
    device: str,
    stage_index: int,
    learning_rate: float,
    noise: np.ndarray,
) -> SAC:
    model = SAC.load(str(model_path), env=env, device=device)
    if replay_path.is_file():
        model.load_replay_buffer(str(replay_path))
    configure_actor(model, stage_index)
    set_learning_rate(model, learning_rate)
    set_action_noise(model, noise)
    return model


def rebuild_critic(
    best_model_path: Path,
    env: Monitor,
    device: str,
    stage_index: int,
    learning_rate: float,
    noise: np.ndarray,
) -> SAC:
    print("CRITIC_REBUILD_FROM_BEST_ACTOR = True")
    actor_source = SAC.load(str(best_model_path), device=device)
    model = make_fresh_model(actor_source, env, device)

    handles = freeze_actor_with_hooks(model)
    model.learn(
        total_timesteps=CRITIC_RESET_WARMUP,
        reset_num_timesteps=True,
        log_interval=8,
        progress_bar=False,
    )
    for handle in handles:
        handle.remove()

    configure_actor(model, stage_index)
    set_learning_rate(model, learning_rate)
    set_action_noise(model, noise)
    return model


def write_state(payload: dict[str, Any]) -> None:
    atomic_save_json(STATE_PATH, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_epoch = time.time()
    deadline = started_epoch + max(1.0, args.hours) * 3600.0

    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("DEVICE =", device)
    print("STAGE = STAGE5_TERMINAL_CURRICULUM_DAYTIME")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("A_STAR_OR_PLANNER = False")
    print("ACTION_OVERRIDE = False")
    print("BOUNDED_PROGRESS_REWARD = True")
    print("COLLISION_DOMINATES_PROGRESS_RETURN = True")
    print("TERMINAL_SPEED_SHAPING = True")
    print("LEXICOGRAPHIC_MODEL_SELECTION = True")
    print("HARD_SEED_CAP =", MAX_COLLISION_SEEDS + MAX_TIMEOUT_SEEDS)
    print("TOTAL_HOURS =", args.hours)

    state = None
    if args.resume and STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    if state is not None:
        stage_index = int(state["stage_index"])
        learning_rate = float(state["learning_rate"])
        noise = np.asarray(state["noise"], dtype=np.float32)
        passed_stages = list(state.get("passed_stages", []))
        collision_seeds = [
            int(v) for v in state.get("collision_seeds", [])
        ]
        timeout_seeds = [
            int(v) for v in state.get("timeout_seeds", [])
        ]
        resume_model = Path(state["best_model"])
        resume_replay = Path(state["best_replay"])
        print("RESUME = True")
    else:
        stage_index = 0
        learning_rate = INITIAL_LR
        noise = INITIAL_NOISE.copy()
        passed_stages: list[str] = []
        collision_seeds: list[int] = []
        timeout_seeds: list[int] = []
        resume_model = Path()
        resume_replay = Path()
        print("RESUME = False")

    if stage_index >= len(DAY_STAGES):
        print("DAY_STAGES_ALREADY_FINISHED = True")
        return

    stage = DAY_STAGES[stage_index]
    previous_stage = (
        DAY_STAGES[stage_index - 1] if stage_index > 0 else None
    )
    stage_dir(stage).mkdir(parents=True, exist_ok=True)
    env_base = TerminalCurriculumEnv(
        stage=stage,
        previous_stage=previous_stage,
        seed=101000 + stage_index * 1000,
    )
    monitor = Monitor(
        env_base,
        filename=str(stage_dir(stage) / "monitor.csv"),
    )

    if state is not None and resume_model.is_file():
        model = load_checkpoint(
            resume_model,
            resume_replay,
            monitor,
            device,
            stage_index,
            learning_rate,
            noise,
        )
    else:
        source = SAC.load(str(SOURCE_MODEL), device=device)
        model = make_fresh_model(source, monitor, device)

        print("INITIAL_CRITIC_WARMUP =", INITIAL_CRITIC_WARMUP)
        handles = freeze_actor_with_hooks(model)
        model.learn(
            total_timesteps=INITIAL_CRITIC_WARMUP,
            reset_num_timesteps=True,
            log_interval=8,
            progress_bar=False,
        )
        for handle in handles:
            handle.remove()

        print(
            "TRAINABLE_ACTOR_PARAMETERS =",
            configure_actor(model, stage_index),
        )
        set_learning_rate(model, learning_rate)
        set_action_noise(model, noise)

    global_records: list[dict[str, Any]] = []
    last_best_model = (
        str(resume_model) if state is not None else ""
    )
    last_best_replay = (
        str(resume_replay) if state is not None else ""
    )
    last_best_metrics: dict[str, Any] = {}

    while stage_index < len(DAY_STAGES) and time.time() < deadline:
        stage = DAY_STAGES[stage_index]
        previous_stage = (
            DAY_STAGES[stage_index - 1]
            if stage_index > 0
            else None
        )
        directory = stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            monitor.close()
        except Exception:
            pass

        env_base = TerminalCurriculumEnv(
            stage=stage,
            previous_stage=previous_stage,
            seed=101000 + stage_index * 1000,
        )
        env_base.set_hard_seeds(
            (collision_seeds + timeout_seeds)[
                -(MAX_COLLISION_SEEDS + MAX_TIMEOUT_SEEDS):
            ]
        )
        monitor = Monitor(
            env_base,
            filename=str(directory / "monitor.csv"),
        )
        model.set_env(monitor)
        print(
            "TRAINABLE_ACTOR_PARAMETERS =",
            configure_actor(model, stage_index),
        )
        set_learning_rate(model, learning_rate)
        set_action_noise(model, noise)

        best_model = directory / "best_model.zip"
        best_replay = directory / "best_replay.pkl"
        best_eval_path = directory / "best_evaluation.json"

        baseline = evaluate_stage(
            model,
            stage,
            previous_stage,
            seed_offset=stage_index * 200000,
        )
        best = baseline
        save_checkpoint(model, best_model, best_replay)
        atomic_save_json(best_eval_path, best)
        last_best_model = str(best_model)
        last_best_replay = str(best_replay)
        last_best_metrics = {
            key: best[key]
            for key in (
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "near_goal_collision_rate",
                "progress_ratio",
                "speed_mps",
                "gate_passed",
            )
        }

        consecutive_passes = 1 if baseline["gate_passed"] else 0
        no_improvement = 0
        critic_rebuilt = False
        chunk = 0
        stage_started = time.time()
        soft_stage_deadline = min(
            deadline,
            stage_started + stage.stage_budget_minutes * 60.0,
        )

        while (
            time.time() < deadline
            and stage_index < len(DAY_STAGES)
        ):
            chunk += 1

            mode = choose_mode(best)
            learning_rate, noise, entropy_floor = apply_mode(
                env_base,
                model,
                mode,
                learning_rate,
                noise,
            )

            env_base.set_hard_seeds(
                (collision_seeds + timeout_seeds)[
                    -(MAX_COLLISION_SEEDS + MAX_TIMEOUT_SEEDS):
                ]
            )

            print(f"\n========== {stage.name} CHUNK {chunk} ==========")
            print("MODE =", mode)
            print("LEARNING_RATE =", learning_rate)
            print("ACTION_NOISE =", noise.tolist())
            print("ENTROPY_FLOOR =", entropy_floor)
            print("COLLISION_SEED_COUNT =", len(collision_seeds))
            print("TIMEOUT_SEED_COUNT =", len(timeout_seeds))

            # Clamp entropy before and after shorter segments.
            for segment in range(2):
                clamp_entropy(model, entropy_floor)
                model.learn(
                    total_timesteps=CHUNK_TIMESTEPS // 2,
                    reset_num_timesteps=False,
                    log_interval=8,
                    progress_bar=False,
                )
                clamp_entropy(model, entropy_floor)

            evaluation = evaluate_stage(
                model,
                stage,
                previous_stage,
                seed_offset=(
                    stage_index * 200000 + chunk * 5000
                ),
            )
            evaluation["mode"] = mode
            evaluation["learning_rate"] = learning_rate
            evaluation["action_noise"] = noise.tolist()
            atomic_save_json(
                directory / f"chunk_{chunk:03d}_evaluation.json",
                evaluation,
            )
            model.save(
                str(directory / f"chunk_{chunk:03d}_model")
            )

            collision_seeds, timeout_seeds = update_seed_pool(
                evaluation,
                collision_seeds,
                timeout_seeds,
            )

            allowed, reason = promotion_allowed(evaluation, best)
            evaluation["promotion_allowed"] = allowed
            evaluation["promotion_reason"] = reason
            atomic_save_json(
                directory / f"chunk_{chunk:03d}_evaluation.json",
                evaluation,
            )
            print("PROMOTION_ALLOWED =", allowed)
            print("PROMOTION_REASON =", reason)

            if allowed and candidate_key(evaluation) < candidate_key(best):
                best = evaluation
                save_checkpoint(model, best_model, best_replay)
                atomic_save_json(best_eval_path, best)
                last_best_model = str(best_model)
                last_best_replay = str(best_replay)
                last_best_metrics = {
                    key: best[key]
                    for key in (
                        "success_rate",
                        "collision_rate",
                        "timeout_rate",
                        "near_goal_collision_rate",
                        "progress_ratio",
                        "speed_mps",
                        "gate_passed",
                    )
                }
                no_improvement = 0
                print("BEST_UPDATED =", chunk)
            else:
                no_improvement += 1
                print("NO_IMPROVEMENT_COUNT =", no_improvement)

            if evaluation["gate_passed"]:
                confirmation = evaluate_stage(
                    model,
                    stage,
                    previous_stage,
                    seed_offset=(
                        1_700_000
                        + stage_index * 200000
                        + chunk * 5000
                    ),
                )
                atomic_save_json(
                    directory
                    / f"chunk_{chunk:03d}_confirmation.json",
                    confirmation,
                )
                if confirmation["gate_passed"]:
                    consecutive_passes = (
                        PASS_CONFIRMATIONS_REQUIRED
                    )
                else:
                    consecutive_passes = 0
            else:
                consecutive_passes = 0

            write_state(
                {
                    "status": "TRAINING",
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "chunk": chunk,
                    "passed_stages": passed_stages,
                    "best_model": str(best_model),
                    "best_replay": str(best_replay),
                    "best_metrics": {
                        key: best[key]
                        for key in (
                            "success_rate",
                            "collision_rate",
                            "timeout_rate",
                            "near_goal_collision_rate",
                            "progress_ratio",
                            "speed_mps",
                            "gate_passed",
                        )
                    },
                    "learning_rate": learning_rate,
                    "noise": noise.tolist(),
                    "mode": mode,
                    "collision_seeds": collision_seeds,
                    "timeout_seeds": timeout_seeds,
                    "no_improvement": no_improvement,
                    "critic_rebuilt": critic_rebuilt,
                    "message": (
                        f"Stage {stage.name}, chunk {chunk}, "
                        f"mode {mode}."
                    ),
                }
            )

            global_records.append(
                {
                    "stage": stage.name,
                    "chunk": chunk,
                    "evaluation": evaluation,
                }
            )

            if consecutive_passes >= PASS_CONFIRMATIONS_REQUIRED:
                print("STAGE_PASS_CONFIRMED =", stage.name)
                pass_record = {
                    "stage": asdict(stage),
                    "best": best,
                    "passed_epoch": time.time(),
                }
                atomic_save_json(directory / "PASS.json", pass_record)

                confirmed_model = (
                    CONFIRMED_ROOT
                    / f"stage5_{stage.name}_passed_model.zip"
                )
                confirmed_eval = (
                    CONFIRMED_ROOT
                    / f"stage5_{stage.name}_evaluation.json"
                )
                shutil.copy2(best_model, confirmed_model)
                atomic_save_json(confirmed_eval, best)

                if stage.name not in passed_stages:
                    passed_stages.append(stage.name)

                stage_index += 1
                collision_seeds = []
                timeout_seeds = []
                no_improvement = 0
                critic_rebuilt = False

                if stage_index < len(DAY_STAGES):
                    next_stage = DAY_STAGES[stage_index]
                    next_previous = DAY_STAGES[stage_index - 1]
                    next_env_base = TerminalCurriculumEnv(
                        stage=next_stage,
                        previous_stage=next_previous,
                        seed=101000 + stage_index * 1000,
                    )
                    next_monitor = Monitor(
                        next_env_base,
                        filename=str(
                            stage_dir(next_stage) / "advance_monitor.csv"
                        ),
                    )
                    model = load_checkpoint(
                        best_model,
                        best_replay,
                        next_monitor,
                        device,
                        stage_index,
                        learning_rate,
                        noise,
                    )
                    monitor = next_monitor
                    env_base = next_env_base
                break

            if no_improvement >= 3:
                print("ROLLBACK_TO_STAGE_BEST = True")
                model = load_checkpoint(
                    best_model,
                    best_replay,
                    monitor,
                    device,
                    stage_index,
                    max(MIN_LR, learning_rate * 0.85),
                    np.maximum(MIN_NOISE, noise * 0.90),
                )
                learning_rate = max(
                    MIN_LR, learning_rate * 0.85
                )
                noise = np.maximum(MIN_NOISE, noise * 0.90)
                no_improvement = 0

            if (
                time.time() >= soft_stage_deadline
                and not critic_rebuilt
            ):
                model = rebuild_critic(
                    best_model,
                    monitor,
                    device,
                    stage_index,
                    max(MIN_LR, learning_rate * 0.80),
                    np.maximum(MIN_NOISE, noise * 0.85),
                )
                learning_rate = max(
                    MIN_LR, learning_rate * 0.80
                )
                noise = np.maximum(MIN_NOISE, noise * 0.85)
                critic_rebuilt = True
                soft_stage_deadline = deadline
                print("SOFT_STAGE_BUDGET_TRIGGERED_CRITIC_RESET = True")

        # End current stage loop.

    try:
        monitor.close()
    except Exception:
        pass

    finished = stage_index >= len(DAY_STAGES)
    summary = {
        "finished_all_day_stages": finished,
        "passed_stages": passed_stages,
        "stopped_stage_index": stage_index,
        "stopped_stage": (
            DAY_STAGES[stage_index].name
            if stage_index < len(DAY_STAGES)
            else None
        ),
        "elapsed_seconds": time.time() - started_epoch,
        "records": global_records,
    }
    atomic_save_json(SUMMARY_PATH, summary)

    write_state(
        {
            "status": (
                "ALL_DAY_STAGES_FINISHED"
                if finished
                else "GLOBAL_BUDGET_FINISHED"
            ),
            "stage_index": stage_index,
            "stage_name": (
                DAY_STAGES[stage_index].name
                if stage_index < len(DAY_STAGES)
                else "FINISHED"
            ),
            "passed_stages": passed_stages,
            "best_model": last_best_model,
            "best_replay": last_best_replay,
            "best_metrics": last_best_metrics,
            "learning_rate": learning_rate,
            "noise": noise.tolist(),
            "collision_seeds": collision_seeds,
            "timeout_seeds": timeout_seeds,
            "summary": str(SUMMARY_PATH),
            "elapsed_seconds": time.time() - started_epoch,
            "message": (
                "All daytime stages passed."
                if finished
                else "Global budget ended; rerun with --resume to continue."
            ),
        }
    )

    print("\nSTAGE5_DAYTIME = FINISHED")
    print("FINISHED_ALL_DAY_STAGES =", finished)
    print("PASSED_STAGES =", passed_stages)
    print(
        "STOPPED_STAGE =",
        DAY_STAGES[stage_index].name
        if stage_index < len(DAY_STAGES)
        else None,
    )
    print("STATE =", STATE_PATH)
    print("SUMMARY =", SUMMARY_PATH)


if __name__ == "__main__":
    main()
