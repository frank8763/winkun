from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.train_stage2a1_safety_reward import (
    set_learning_rate,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage2a11_overnight_autonomous import (
    OvernightObstacleEnv,
    save_json,
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
from training.e2e_phase10_lidar_v7_curriculum.train_stage2b1_light_range_expansion import (
    Stage2B1Env,
)


SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "confirmed/stage2b1a4_final_pass_model.zip"
)
OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage4_acrp_sac_overnight"
)
CONFIRMED_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/confirmed"
)

STATE_PATH = OUTPUT_ROOT / "overnight_state.json"
CONSOLE_LOG_PATH = OUTPUT_ROOT / "overnight_console.log"

TOTAL_WALL_BUDGET_SECONDS = 8 * 60 * 60
CRITIC_WARMUP_TIMESTEPS = 30_000
CHUNK_TIMESTEPS = 30_000
EVAL_EPISODES = 30
PASS_CONFIRMATIONS_REQUIRED = 2
MAX_HARD_SEEDS = 96

INITIAL_JOINT_LR = 8.0e-6
MIN_JOINT_LR = 1.5e-6
MAX_JOINT_LR = 1.2e-5

INITIAL_NOISE = np.asarray([0.0030, 0.0060], dtype=np.float32)
MIN_NOISE = np.asarray([0.0010, 0.0020], dtype=np.float32)
MAX_NOISE = np.asarray([0.0060, 0.0120], dtype=np.float32)


@dataclass(frozen=True)
class DistanceStage:
    name: str
    goal_min: float
    goal_max: float
    angle_deg: float
    max_steps: int
    clearance_margin: float
    stage_budget_minutes: int
    success_gate: float
    collision_gate: float
    progress_gate: float
    speed_gate: float
    timeout_gate: float


# Each distance is an independent stage, checkpoint and acceptance gate.
# The extra bridge stages are deliberate: Stage3C proved that jumping directly
# from roughly 1 m to 1.5-2.5 m does not work.
DISTANCE_STAGES: tuple[DistanceStage, ...] = (
    DistanceStage(
        name="S1_1p0_1p5_front",
        goal_min=1.00,
        goal_max=1.50,
        angle_deg=35.0,
        max_steps=1200,
        clearance_margin=0.38,
        stage_budget_minutes=35,
        success_gate=0.60,
        collision_gate=0.10,
        progress_gate=0.70,
        speed_gate=0.045,
        timeout_gate=0.32,
    ),
    DistanceStage(
        name="S2_1p5_2p0_front",
        goal_min=1.50,
        goal_max=2.00,
        angle_deg=50.0,
        max_steps=1550,
        clearance_margin=0.37,
        stage_budget_minutes=45,
        success_gate=0.45,
        collision_gate=0.13,
        progress_gate=0.62,
        speed_gate=0.042,
        timeout_gate=0.45,
    ),
    DistanceStage(
        name="S3_1p5_2p5_turn",
        goal_min=1.50,
        goal_max=2.50,
        angle_deg=75.0,
        max_steps=1900,
        clearance_margin=0.35,
        stage_budget_minutes=55,
        success_gate=0.30,
        collision_gate=0.16,
        progress_gate=0.55,
        speed_gate=0.040,
        timeout_gate=0.58,
    ),
    DistanceStage(
        name="S4_1p5_2p5_full",
        goal_min=1.50,
        goal_max=2.50,
        angle_deg=90.0,
        max_steps=2100,
        clearance_margin=0.34,
        stage_budget_minutes=60,
        success_gate=0.25,
        collision_gate=0.17,
        progress_gate=0.52,
        speed_gate=0.038,
        timeout_gate=0.62,
    ),
    DistanceStage(
        name="S5_2p5_3p2",
        goal_min=2.50,
        goal_max=3.20,
        angle_deg=115.0,
        max_steps=2800,
        clearance_margin=0.33,
        stage_budget_minutes=65,
        success_gate=0.18,
        collision_gate=0.18,
        progress_gate=0.44,
        speed_gate=0.035,
        timeout_gate=0.70,
    ),
    DistanceStage(
        name="S6_2p5_4p0",
        goal_min=2.50,
        goal_max=4.00,
        angle_deg=150.0,
        max_steps=3400,
        clearance_margin=0.32,
        stage_budget_minutes=70,
        success_gate=0.12,
        collision_gate=0.19,
        progress_gate=0.38,
        speed_gate=0.032,
        timeout_gate=0.76,
    ),
    DistanceStage(
        name="S7_3p5_4p5",
        goal_min=3.50,
        goal_max=4.50,
        angle_deg=170.0,
        max_steps=4100,
        clearance_margin=0.31,
        stage_budget_minutes=75,
        success_gate=0.08,
        collision_gate=0.20,
        progress_gate=0.31,
        speed_gate=0.030,
        timeout_gate=0.82,
    ),
    DistanceStage(
        name="S8_3p5_5p5",
        goal_min=3.50,
        goal_max=5.50,
        angle_deg=180.0,
        max_steps=4700,
        clearance_margin=0.30,
        stage_budget_minutes=80,
        success_gate=0.06,
        collision_gate=0.20,
        progress_gate=0.28,
        speed_gate=0.028,
        timeout_gate=0.84,
    ),
)


def stage_dir(stage: DistanceStage) -> Path:
    return OUTPUT_ROOT / "stages" / stage.name


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class RiskPrioritizedReplayBuffer(ReplayBuffer):
    """Event-based risk replay for end-to-end SAC.

    This does not modify actions. It only changes which stored transitions are
    sampled during training. Collision, near-obstacle, timeout and successful
    transitions receive larger replay probability, while a uniform component
    preserves coverage.
    """

    def __init__(
        self,
        *args: Any,
        risk_alpha: float = 0.70,
        uniform_mix: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.risk_alpha = float(np.clip(risk_alpha, 0.0, 1.0))
        self.uniform_mix = float(np.clip(uniform_mix, 0.05, 1.0))
        self.risk_priorities = np.ones(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        write_index = int(self.pos)
        super().add(obs, next_obs, action, reward, done, infos)

        for env_index in range(self.n_envs):
            info = infos[env_index] if env_index < len(infos) else {}
            explicit = float(info.get("risk_priority", 1.0))
            collision = bool(info.get("collision", False))
            success = bool(info.get("reach_goal", False))
            timeout = bool(info.get("timeout", False))
            risk_ratio = float(info.get("stage4_risk_ratio", 0.0))
            idle_ratio = float(info.get("stage4_idle_ratio", 0.0))

            priority = max(
                explicit,
                1.0
                + 8.0 * float(collision)
                + 4.0 * risk_ratio
                + 2.5 * float(success)
                + 1.5 * float(timeout)
                + 1.0 * idle_ratio,
            )
            self.risk_priorities[write_index, env_index] = float(priority)

    def sample(self, batch_size: int, env: Any | None = None):
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 1:
            return super().sample(batch_size=batch_size, env=env)

        priorities = np.max(
            self.risk_priorities[:upper_bound],
            axis=1,
        ).astype(np.float64)
        priorities = np.maximum(priorities, 1e-6)
        weighted = np.power(priorities, self.risk_alpha)
        weighted /= np.sum(weighted)

        uniform = np.full(
            upper_bound,
            1.0 / upper_bound,
            dtype=np.float64,
        )
        probabilities = (
            (1.0 - self.uniform_mix) * weighted
            + self.uniform_mix * uniform
        )
        probabilities /= probabilities.sum()

        batch_indices = np.random.choice(
            upper_bound,
            size=batch_size,
            replace=True,
            p=probabilities,
        )
        return self._get_samples(batch_indices, env=env)


def xy_from_env(env: Any) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    data = getattr(base, "data", None)
    if data is None or len(data.qpos) < 2:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(data.qpos[:2], dtype=np.float64).copy()


class Stage4CurriculumEnv(Stage2B1Env):
    """Strict end-to-end LiDAR SAC environment.

    Observation and action remain unchanged:
      LiDAR + relative goal + history -> neural policy
      -> rear-wheel speed + front steering -> MuJoCo actuators.
    """

    def __init__(
        self,
        stage: DistanceStage,
        previous_stage: DistanceStage | None = None,
        fixed_task: str | None = None,
        seed: int = 88000,
    ) -> None:
        super().__init__(fixed_task=None, seed=seed)
        self.stage = stage
        self.previous_stage = previous_stage
        self.fixed_curriculum_task = fixed_task
        self._rng4 = np.random.default_rng(seed + 9187)

        self._current_stage_probability = 0.62
        self._previous_stage_probability = (
            0.18 if previous_stage is not None else 0.0
        )
        self._hard_seed_probability = 0.28
        self._hard_seeds: list[int] = []

        self.progress_scale = 18.0
        self.risk_scale = 0.11
        self.speed_risk_scale = 0.10
        self.safe_cruise_scale = 0.009
        self.idle_scale = 0.008

        self._previous_goal_distance = 1.0
        self._previous_goal_angle = 0.0
        self._initial_goal_distance = 1.0
        self._previous_xy = np.zeros(2, dtype=np.float64)
        self._previous_stage4_action = np.zeros(2, dtype=np.float32)
        self._idle_steps = 0
        self._actual_seed = seed
        self._sampling_mode = "stage"

    @property
    def current_task_name(self) -> str:
        return self.stage.name

    @property
    def previous_task_name(self) -> str | None:
        return None if self.previous_stage is None else self.previous_stage.name

    def set_hard_seeds(self, seeds: list[int]) -> None:
        unique = []
        seen = set()
        for seed in seeds:
            value = int(seed)
            if value not in seen:
                seen.add(value)
                unique.append(value)
        self._hard_seeds = unique[-MAX_HARD_SEEDS:]

    def set_adaptation(
        self,
        *,
        current_stage_probability: float | None = None,
        hard_seed_probability: float | None = None,
        risk_scale: float | None = None,
        speed_risk_scale: float | None = None,
        safe_cruise_scale: float | None = None,
        idle_scale: float | None = None,
    ) -> None:
        if current_stage_probability is not None:
            self._current_stage_probability = float(
                np.clip(current_stage_probability, 0.45, 0.82)
            )
        if hard_seed_probability is not None:
            self._hard_seed_probability = float(
                np.clip(hard_seed_probability, 0.10, 0.55)
            )
        if risk_scale is not None:
            self.risk_scale = float(np.clip(risk_scale, 0.05, 0.25))
        if speed_risk_scale is not None:
            self.speed_risk_scale = float(
                np.clip(speed_risk_scale, 0.04, 0.25)
            )
        if safe_cruise_scale is not None:
            self.safe_cruise_scale = float(
                np.clip(safe_cruise_scale, 0.003, 0.020)
            )
        if idle_scale is not None:
            self.idle_scale = float(np.clip(idle_scale, 0.003, 0.025))

    def _apply_distance_stage(self, config: DistanceStage) -> None:
        OvernightObstacleEnv._apply_task(self, "light")
        self._active_task = config.name
        self.curriculum_stage = config.name
        self.goal_distance_min = float(config.goal_min)
        self.goal_distance_max = float(config.goal_max)
        self.goal_angle_limit = math.radians(float(config.angle_deg))
        self.goal_clearance_margin = float(config.clearance_margin)
        self.max_steps = int(config.max_steps)

    def _apply_task(self, task: str) -> None:
        if task == self.stage.name:
            self._apply_distance_stage(self.stage)
            return
        if (
            self.previous_stage is not None
            and task == self.previous_stage.name
        ):
            self._apply_distance_stage(self.previous_stage)
            return
        OvernightObstacleEnv._apply_task(self, task)

    def _sample_task(self) -> str:
        draw = float(self._rng4.random())
        current = self._current_stage_probability
        previous = self._previous_stage_probability

        if draw < current:
            return self.stage.name
        if (
            self.previous_stage is not None
            and draw < current + previous
        ):
            return self.previous_stage.name

        retention_tasks = ("a3", "light", "caution", "bridge")
        return str(self._rng4.choice(retention_tasks))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        forced_task = None if options is None else options.get("task_mode")

        if forced_task is not None:
            task = str(forced_task)
            sampling_mode = "forced"
        elif self.fixed_curriculum_task is not None:
            task = str(self.fixed_curriculum_task)
            sampling_mode = "evaluation"
        else:
            task = self._sample_task()
            sampling_mode = "curriculum"

        actual_seed: int
        replay_hard_seed = bool(
            task == self.stage.name
            and self._hard_seeds
            and self._rng4.random() < self._hard_seed_probability
        )
        if seed is not None:
            actual_seed = int(seed)
            sampling_mode = "explicit_seed"
        elif replay_hard_seed:
            actual_seed = int(self._rng4.choice(self._hard_seeds))
            sampling_mode = "hard_seed_replay"
        else:
            actual_seed = int(
                self._rng4.integers(600000, 2_000_000_000)
            )

        observation, info = OvernightObstacleEnv.reset(
            self,
            seed=actual_seed,
            options={"task_mode": task},
        )

        self._actual_seed = actual_seed
        self._sampling_mode = sampling_mode
        self._last_front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        self._previous_action.fill(0.0)
        self._previous_delta.fill(0.0)

        self._initial_goal_distance = float(
            info.get("goal_dist", 1.0)
        )
        self._previous_goal_distance = self._initial_goal_distance
        self._previous_goal_angle = float(
            info.get("goal_angle", 0.0)
        )
        self._previous_xy = xy_from_env(self)
        self._previous_stage4_action.fill(0.0)
        self._idle_steps = 0

        info["task_mode"] = task
        info["sampling_mode"] = sampling_mode
        info["training_seed"] = actual_seed
        info["actual_speed_mps"] = 0.0
        info["progress_ratio"] = 0.0
        return observation, info

    def step(self, action: np.ndarray):
        action_array = np.asarray(action, dtype=np.float32).reshape(2)

        observation, inherited_reward, terminated, truncated, info = (
            Stage2B1Env.step(self, action_array)
        )

        current_xy = xy_from_env(self)
        dt = float(max(getattr(self, "dt", 0.02), 1e-6))
        displacement = float(
            np.linalg.norm(current_xy - self._previous_xy)
        )
        actual_speed = displacement / dt

        goal_distance = float(info.get("goal_dist", 999.0))
        goal_angle = float(info.get("goal_angle", 0.0))
        front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        minimum_lidar = float(
            info.get("min_lidar_m", front_clearance)
        )
        collision = bool(info.get("collision", False))
        success = bool(info.get("reach_goal", False))
        timeout = bool(
            not success
            and not collision
            and (truncated or self.step_count >= self.max_steps)
        )

        progress = self._previous_goal_distance - goal_distance
        angle_progress = (
            abs(self._previous_goal_angle) - abs(goal_angle)
        )
        progress_ratio = float(
            np.clip(
                (
                    self._initial_goal_distance - goal_distance
                )
                / max(self._initial_goal_distance, 1e-6),
                -1.0,
                1.0,
            )
        )

        # One consistent reward scale is used for every curriculum and
        # retention task so that one critic does not see incompatible targets.
        progress_reward = self.progress_scale * float(
            np.clip(progress, -0.02, 0.02)
        )
        heading_reward = 0.010 * float(
            np.clip(
                angle_progress / math.radians(2.0),
                -1.0,
                1.0,
            )
        )
        time_penalty = -0.0008

        risk_distance = min(front_clearance, minimum_lidar)
        risk_ratio = float(
            np.clip((0.55 - risk_distance) / 0.35, 0.0, 1.0)
        )
        risk_penalty = -self.risk_scale * risk_ratio * risk_ratio
        moving_risk_penalty = (
            -self.speed_risk_scale
            * risk_ratio
            * float(np.clip(actual_speed / 0.12, 0.0, 1.5))
        )

        aligned = abs(goal_angle) <= math.radians(55.0)
        safe = risk_distance >= 0.45
        far = goal_distance >= 0.55

        safe_cruise_bonus = 0.0
        if safe and aligned and far and progress > 0.0:
            safe_cruise_bonus = self.safe_cruise_scale * float(
                np.clip(actual_speed / 0.12, 0.0, 1.0)
            )

        if safe and far and actual_speed < 0.022:
            self._idle_steps += 1
        else:
            self._idle_steps = max(0, self._idle_steps - 3)

        idle_ratio = float(
            np.clip((self._idle_steps - 24) / 50.0, 0.0, 1.0)
        )
        idle_penalty = -self.idle_scale * idle_ratio

        action_delta = action_array - self._previous_stage4_action
        smoothness_penalty = -0.0012 * float(
            np.dot(action_delta, action_delta)
        )

        terminal_reward = 0.0
        if success:
            terminal_reward = 12.0
        elif collision:
            terminal_reward = -10.0
        elif timeout:
            terminal_reward = -0.8

        chatter_penalty = float(
            info.get("stage2a17_chatter_penalty", 0.0)
        )

        reward = float(
            progress_reward
            + heading_reward
            + time_penalty
            + risk_penalty
            + moving_risk_penalty
            + safe_cruise_bonus
            + idle_penalty
            + smoothness_penalty
            + terminal_reward
            + chatter_penalty
        )

        risk_priority = float(
            1.0
            + 7.0 * float(collision)
            + 4.0 * risk_ratio
            + 3.0 * float(success)
            + 1.5 * float(timeout)
            + 1.0 * idle_ratio
        )

        self._previous_goal_distance = goal_distance
        self._previous_goal_angle = goal_angle
        self._previous_xy = current_xy
        self._previous_stage4_action = action_array.copy()

        info["task_mode"] = str(getattr(self, "_active_task", ""))
        info["sampling_mode"] = self._sampling_mode
        info["training_seed"] = self._actual_seed
        info["actual_speed_mps"] = actual_speed
        info["progress_ratio"] = progress_ratio
        info["timeout"] = timeout
        info["stage4_risk_ratio"] = risk_ratio
        info["stage4_idle_ratio"] = idle_ratio
        info["risk_priority"] = risk_priority
        info["stage4_progress"] = progress
        info["stage4_reward"] = reward
        info["inherited_reward_unused"] = float(inherited_reward)

        return observation, reward, terminated, truncated, info


def set_action_noise(model: SAC, sigma: np.ndarray) -> None:
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray(sigma, dtype=np.float32),
    )


def make_fresh_model(
    source: SAC,
    env: Monitor,
    device: str,
) -> SAC:
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=1.0e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        action_noise=NormalActionNoise(
            mean=np.zeros(2, dtype=np.float32),
            sigma=INITIAL_NOISE,
        ),
        replay_buffer_class=RiskPrioritizedReplayBuffer,
        replay_buffer_kwargs={
            "risk_alpha": 0.70,
            "uniform_mix": 0.25,
        },
        optimize_memory_usage=False,
        ent_coef="auto_0.01",
        target_entropy="auto",
        policy_kwargs=dict(source.policy_kwargs or {}),
        verbose=1,
        seed=88000,
        device=device,
    )
    model.actor.load_state_dict(source.actor.state_dict(), strict=True)
    return model


def freeze_actor_with_hooks(model: SAC):
    handles = []

    def zero_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(gradient)

    for parameter in model.actor.parameters():
        handles.append(parameter.register_hook(zero_gradient))
    return handles


def evaluate_task(
    model: SAC,
    stage: DistanceStage,
    previous_stage: DistanceStage | None,
    task: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = Stage4CurriculumEnv(
        stage=stage,
        previous_stage=previous_stage,
        fixed_task=task,
        seed=seed + 9000,
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
            path_length = 0.0
            speed_sum = 0.0
            steps = 0

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
                speed = float(info.get("actual_speed_mps", 0.0))
                speed_sum += speed
                path_length += speed * float(
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
            path_efficiency = (
                float(
                    np.clip(
                        initial_distance / max(path_length, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                if success
                else 0.0
            )

            rows.append(
                {
                    "seed": actual_seed,
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "initial_goal_distance": initial_distance,
                    "final_goal_distance": final_distance,
                    "progress_ratio": progress_ratio,
                    "mean_speed_mps": speed_sum / max(steps, 1),
                    "path_length": path_length,
                    "path_efficiency": path_efficiency,
                }
            )
    finally:
        env.close()

    count = len(rows)
    result = {
        "task": task,
        "episodes": count,
        "success_rate": sum(int(row["success"]) for row in rows) / count,
        "collision_rate": (
            sum(int(row["collision"]) for row in rows) / count
        ),
        "timeout_rate": sum(int(row["timeout"]) for row in rows) / count,
        "mean_progress_ratio": float(
            np.mean([row["progress_ratio"] for row in rows])
        ),
        "mean_speed_mps": float(
            np.mean([row["mean_speed_mps"] for row in rows])
        ),
        "mean_path_efficiency": float(
            np.mean([row["path_efficiency"] for row in rows])
        ),
        "collision_seeds": [
            int(row["seed"]) for row in rows if row["collision"]
        ],
        "timeout_seeds": [
            int(row["seed"]) for row in rows if row["timeout"]
        ],
        "results": rows,
    }

    print(f"\n========== EVAL {task} {count} ==========")
    print("SUCCESS_RATE =", result["success_rate"])
    print("COLLISION_RATE =", result["collision_rate"])
    print("TIMEOUT_RATE =", result["timeout_rate"])
    print("MEAN_PROGRESS_RATIO =", result["mean_progress_ratio"])
    print("MEAN_SPEED_MPS =", result["mean_speed_mps"])
    print("MEAN_PATH_EFFICIENCY =", result["mean_path_efficiency"])
    return result


def gate_passed(
    result: dict[str, Any],
    stage: DistanceStage,
) -> bool:
    return bool(
        result["success_rate"] >= stage.success_gate
        and result["collision_rate"] <= stage.collision_gate
        and result["progress_ratio"] >= stage.progress_gate
        and result["speed_mps"] >= stage.speed_gate
        and result["timeout_rate"] <= stage.timeout_gate
        and result["historical_safety"]
        and result["retention_safety"]
    )


def evaluate_stage(
    model: SAC,
    stage: DistanceStage,
    previous_stage: DistanceStage | None,
    seed_offset: int,
) -> dict[str, Any]:
    current = evaluate_task(
        model,
        stage=stage,
        previous_stage=previous_stage,
        task=stage.name,
        episodes=EVAL_EPISODES,
        seed=800000 + seed_offset,
    )
    a3 = evaluate_task(
        model,
        stage=stage,
        previous_stage=previous_stage,
        task="a3",
        episodes=12,
        seed=810000 + seed_offset,
    )
    light = evaluate_task(
        model,
        stage=stage,
        previous_stage=previous_stage,
        task="light",
        episodes=15,
        seed=820000 + seed_offset,
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

    result = {
        "stage": asdict(stage),
        "task": stage.name,
        "success_rate": current["success_rate"],
        "collision_rate": current["collision_rate"],
        "timeout_rate": current["timeout_rate"],
        "progress_ratio": current["mean_progress_ratio"],
        "speed_mps": current["mean_speed_mps"],
        "path_efficiency": current["mean_path_efficiency"],
        "current": current,
        "a3": a3,
        "light": light,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "hard_collision_count": hard_collision_count,
        "historical_safety": historical_safety,
        "retention_safety": retention_safety,
    }
    result["gate_passed"] = gate_passed(result, stage)
    result["selection_key"] = [
        0 if historical_safety else 1,
        0 if retention_safety else 1,
        float(current["collision_rate"]),
        -float(current["success_rate"]),
        -float(current["mean_progress_ratio"]),
        -float(current["mean_speed_mps"]),
        float(current["timeout_rate"]),
    ]

    print(f"\n========== STAGE SUMMARY {stage.name} ==========")
    print("GATE_PASSED =", result["gate_passed"])
    print("HISTORICAL_SAFETY =", historical_safety)
    print("RETENTION_SAFETY =", retention_safety)
    return result


def is_better(
    candidate: dict[str, Any],
    best: dict[str, Any],
) -> bool:
    return tuple(candidate["selection_key"]) < tuple(
        best["selection_key"]
    )


def build_state(
    *,
    global_started_epoch: float,
    stage_index: int,
    chunk: int,
    passed_stages: list[str],
    best_model_path: str,
    best_replay_path: str,
    joint_lr: float,
    noise_sigma: np.ndarray,
    consecutive_passes: int,
    hard_seeds: list[int],
    status: str,
    message: str,
) -> dict[str, Any]:
    stage_name = (
        DISTANCE_STAGES[stage_index].name
        if 0 <= stage_index < len(DISTANCE_STAGES)
        else "ALL_STAGES_FINISHED"
    )
    return {
        "global_started_epoch": global_started_epoch,
        "updated_epoch": time.time(),
        "stage_index": stage_index,
        "stage_name": stage_name,
        "chunk": chunk,
        "passed_stages": passed_stages,
        "best_model_path": best_model_path,
        "best_replay_path": best_replay_path,
        "joint_lr": joint_lr,
        "noise_sigma": np.asarray(noise_sigma).tolist(),
        "consecutive_passes": consecutive_passes,
        "hard_seeds": hard_seeds[-MAX_HARD_SEEDS:],
        "status": status,
        "message": message,
    }


def load_state() -> dict[str, Any] | None:
    if not STATE_PATH.is_file():
        return None
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def rollback_model(
    *,
    model_path: Path,
    replay_path: Path,
    env: Monitor,
    device: str,
    joint_lr: float,
    noise_sigma: np.ndarray,
) -> SAC:
    print("ROLLBACK_MODEL =", model_path)
    model = SAC.load(str(model_path), env=env, device=device)
    if replay_path.is_file():
        model.load_replay_buffer(str(replay_path))
    set_learning_rate(model, joint_lr)
    set_action_noise(model, noise_sigma)
    return model


def adapt_after_evaluation(
    result: dict[str, Any],
    *,
    joint_lr: float,
    noise_sigma: np.ndarray,
    env: Stage4CurriculumEnv,
) -> tuple[float, np.ndarray, str]:
    collision = float(result["collision_rate"])
    timeout = float(result["timeout_rate"])
    progress = float(result["progress_ratio"])
    speed = float(result["speed_mps"])

    decision = "KEEP"

    if collision > 0.35:
        joint_lr = max(MIN_JOINT_LR, joint_lr * 0.70)
        noise_sigma = np.maximum(MIN_NOISE, noise_sigma * 0.72)
        env.set_adaptation(
            current_stage_probability=0.55,
            hard_seed_probability=0.48,
            risk_scale=env.risk_scale * 1.15,
            speed_risk_scale=env.speed_risk_scale * 1.15,
            safe_cruise_scale=env.safe_cruise_scale * 0.90,
        )
        decision = "HIGH_COLLISION_CONSERVATIVE"

    elif collision <= 0.15 and (
        timeout > 0.65 or progress < 0.30 or speed < 0.025
    ):
        joint_lr = min(MAX_JOINT_LR, joint_lr * 1.08)
        noise_sigma = np.minimum(MAX_NOISE, noise_sigma * 1.18)
        env.set_adaptation(
            current_stage_probability=0.70,
            hard_seed_probability=0.24,
            safe_cruise_scale=env.safe_cruise_scale * 1.12,
            idle_scale=env.idle_scale * 1.12,
        )
        decision = "STALL_MORE_EXPLORATION"

    elif collision <= 0.22 and progress >= 0.35:
        env.set_adaptation(
            current_stage_probability=0.72,
            hard_seed_probability=0.30,
        )
        decision = "PROMISING_FOCUS_CURRENT_STAGE"

    set_learning_rate(_MODEL_HOLDER["model"], joint_lr)
    set_action_noise(_MODEL_HOLDER["model"], noise_sigma)
    return joint_lr, noise_sigma, decision


# A small holder avoids duplicating optimizer adaptation code while keeping the
# environment adaptation function easy to test.
_MODEL_HOLDER: dict[str, SAC] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from overnight_state.json when available.",
    )
    parser.add_argument(
        "--total-hours",
        type=float,
        default=8.0,
        help="Global wall-clock budget, including evaluation.",
    )
    args = parser.parse_args()

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_budget_seconds = max(
        60.0,
        float(args.total_hours) * 60.0 * 60.0,
    )

    existing_state = load_state() if args.resume else None
    if existing_state is not None:
        global_started_epoch = float(
            existing_state["global_started_epoch"]
        )
        stage_index = int(existing_state["stage_index"])
        passed_stages = list(existing_state.get("passed_stages", []))
        joint_lr = float(
            existing_state.get("joint_lr", INITIAL_JOINT_LR)
        )
        noise_sigma = np.asarray(
            existing_state.get("noise_sigma", INITIAL_NOISE.tolist()),
            dtype=np.float32,
        )
        hard_seeds = [
            int(value)
            for value in existing_state.get("hard_seeds", [])
        ]
        print("RESUME_STATE =", STATE_PATH)
    else:
        global_started_epoch = time.time()
        stage_index = 0
        passed_stages = []
        joint_lr = INITIAL_JOINT_LR
        noise_sigma = INITIAL_NOISE.copy()
        hard_seeds = []

    global_deadline = global_started_epoch + total_budget_seconds

    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("DEVICE =", device)
    print("STAGE = STAGE4_ACRP_SAC_OVERNIGHT")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("A_STAR_OR_PLANNER = False")
    print("RISK_PRIORITIZED_REPLAY = True")
    print("FAILURE_SEED_REPLAY = True")
    print("TWO_CONSECUTIVE_PASSES_REQUIRED = True")
    print("AUTOMATIC_ROLLBACK = True")
    print("TOTAL_HOURS =", args.total_hours)
    print("START_STAGE_INDEX =", stage_index)

    if stage_index >= len(DISTANCE_STAGES):
        print("ALL_STAGES_ALREADY_FINISHED = True")
        return

    # Fresh initialization only for a brand-new run. Every later stage keeps a
    # compatible critic and replay buffer because the reward scale is fixed.
    first_stage = DISTANCE_STAGES[stage_index]
    previous_stage = (
        DISTANCE_STAGES[stage_index - 1]
        if stage_index > 0
        else None
    )
    base_env = Stage4CurriculumEnv(
        stage=first_stage,
        previous_stage=previous_stage,
        seed=88000 + stage_index * 1000,
    )
    base_env.set_hard_seeds(hard_seeds)
    monitor_env = Monitor(
        base_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
    )

    if existing_state is not None:
        model_path = Path(existing_state["best_model_path"])
        replay_path = Path(existing_state["best_replay_path"])
        model = rollback_model(
            model_path=model_path,
            replay_path=replay_path,
            env=monitor_env,
            device=device,
            joint_lr=joint_lr,
            noise_sigma=noise_sigma,
        )
    else:
        source = SAC.load(str(SOURCE_MODEL), device=device)
        model = make_fresh_model(
            source=source,
            env=monitor_env,
            device=device,
        )

        print("\n========== INITIAL CRITIC WARMUP ==========")
        print("TIMESTEPS =", CRITIC_WARMUP_TIMESTEPS)
        print("ACTOR_TRANSFERRED = True")
        print("ACTOR_FROZEN_BY_ZERO_GRADIENT = True")

        handles = freeze_actor_with_hooks(model)
        model.learn(
            total_timesteps=CRITIC_WARMUP_TIMESTEPS,
            reset_num_timesteps=True,
            log_interval=8,
            progress_bar=False,
        )
        for handle in handles:
            handle.remove()

        set_learning_rate(model, joint_lr)
        set_action_noise(model, noise_sigma)

    _MODEL_HOLDER["model"] = model

    while (
        stage_index < len(DISTANCE_STAGES)
        and time.time() < global_deadline
    ):
        stage = DISTANCE_STAGES[stage_index]
        previous_stage = (
            DISTANCE_STAGES[stage_index - 1]
            if stage_index > 0
            else None
        )
        directory = stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)

        # Rebuild environment at every stage boundary.
        try:
            monitor_env.close()
        except Exception:
            pass

        base_env = Stage4CurriculumEnv(
            stage=stage,
            previous_stage=previous_stage,
            seed=88000 + stage_index * 1000,
        )
        base_env.set_hard_seeds(hard_seeds)
        monitor_env = Monitor(
            base_env,
            filename=str(directory / "monitor.csv"),
        )
        model.set_env(monitor_env)
        _MODEL_HOLDER["model"] = model
        set_learning_rate(model, joint_lr)
        set_action_noise(model, noise_sigma)

        stage_started = time.time()
        stage_deadline = min(
            global_deadline,
            stage_started + stage.stage_budget_minutes * 60.0,
        )

        best_model_path = directory / "best_model.zip"
        best_replay_path = directory / "best_replay.pkl"
        best_evaluation_path = directory / "best_evaluation.json"

        initial_eval = evaluate_stage(
            model,
            stage=stage,
            previous_stage=previous_stage,
            seed_offset=stage_index * 100000,
        )
        atomic_save_json(directory / "initial_evaluation.json", initial_eval)

        best = initial_eval
        model.save(str(directory / "best_model"))
        model.save_replay_buffer(str(best_replay_path))
        atomic_save_json(best_evaluation_path, best)

        consecutive_passes = 1 if initial_eval["gate_passed"] else 0
        chunk = 0
        plateau_count = 0

        state = build_state(
            global_started_epoch=global_started_epoch,
            stage_index=stage_index,
            chunk=chunk,
            passed_stages=passed_stages,
            best_model_path=str(best_model_path),
            best_replay_path=str(best_replay_path),
            joint_lr=joint_lr,
            noise_sigma=noise_sigma,
            consecutive_passes=consecutive_passes,
            hard_seeds=hard_seeds,
            status="TRAINING_STAGE",
            message="Initial evaluation complete.",
        )
        atomic_save_json(STATE_PATH, state)

        while time.time() < stage_deadline:
            chunk += 1
            print(
                f"\n========== {stage.name} CHUNK {chunk} =========="
            )
            print("JOINT_LR =", joint_lr)
            print("ACTION_NOISE =", noise_sigma.tolist())
            print("HARD_SEED_COUNT =", len(hard_seeds))
            print("CHUNK_TIMESTEPS =", CHUNK_TIMESTEPS)

            model.learn(
                total_timesteps=CHUNK_TIMESTEPS,
                reset_num_timesteps=False,
                log_interval=8,
                progress_bar=False,
            )

            evaluation = evaluate_stage(
                model,
                stage=stage,
                previous_stage=previous_stage,
                seed_offset=(
                    stage_index * 100000 + chunk * 5000
                ),
            )
            atomic_save_json(
                directory / f"chunk_{chunk:03d}_evaluation.json",
                evaluation,
            )
            model.save(str(directory / f"chunk_{chunk:03d}_model"))

            failure_seeds = (
                evaluation["current"]["collision_seeds"]
                + evaluation["current"]["timeout_seeds"]
            )
            hard_seeds = (
                hard_seeds + [int(seed) for seed in failure_seeds]
            )[-MAX_HARD_SEEDS:]
            base_env.set_hard_seeds(hard_seeds)

            if is_better(evaluation, best):
                best = evaluation
                plateau_count = 0
                model.save(str(directory / "best_model"))
                model.save_replay_buffer(str(best_replay_path))
                atomic_save_json(best_evaluation_path, best)
                print("BEST_UPDATED =", chunk)
            else:
                plateau_count += 1

            if evaluation["gate_passed"]:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            severe_degradation = bool(
                not evaluation["historical_safety"]
                or not evaluation["retention_safety"]
                or evaluation["collision_rate"] > 0.55
            )

            if severe_degradation:
                joint_lr = max(MIN_JOINT_LR, joint_lr * 0.65)
                noise_sigma = np.maximum(
                    MIN_NOISE,
                    noise_sigma * 0.65,
                )
                model = rollback_model(
                    model_path=best_model_path,
                    replay_path=best_replay_path,
                    env=monitor_env,
                    device=device,
                    joint_lr=joint_lr,
                    noise_sigma=noise_sigma,
                )
                _MODEL_HOLDER["model"] = model
                base_env.set_adaptation(
                    current_stage_probability=0.52,
                    hard_seed_probability=0.52,
                    risk_scale=base_env.risk_scale * 1.18,
                    speed_risk_scale=(
                        base_env.speed_risk_scale * 1.18
                    ),
                )
                print("AUTO_ROLLBACK = SEVERE_DEGRADATION")
            else:
                joint_lr, noise_sigma, decision = (
                    adapt_after_evaluation(
                        evaluation,
                        joint_lr=joint_lr,
                        noise_sigma=noise_sigma,
                        env=base_env,
                    )
                )
                print("ADAPTATION_DECISION =", decision)

            if plateau_count >= 4:
                model = rollback_model(
                    model_path=best_model_path,
                    replay_path=best_replay_path,
                    env=monitor_env,
                    device=device,
                    joint_lr=joint_lr,
                    noise_sigma=noise_sigma,
                )
                _MODEL_HOLDER["model"] = model
                plateau_count = 0
                print("AUTO_ROLLBACK = PLATEAU")

            state = build_state(
                global_started_epoch=global_started_epoch,
                stage_index=stage_index,
                chunk=chunk,
                passed_stages=passed_stages,
                best_model_path=str(best_model_path),
                best_replay_path=str(best_replay_path),
                joint_lr=joint_lr,
                noise_sigma=noise_sigma,
                consecutive_passes=consecutive_passes,
                hard_seeds=hard_seeds,
                status="TRAINING_STAGE",
                message=(
                    f"Latest gate={evaluation['gate_passed']}; "
                    f"consecutive={consecutive_passes}."
                ),
            )
            atomic_save_json(STATE_PATH, state)

            if consecutive_passes >= PASS_CONFIRMATIONS_REQUIRED:
                print("STAGE_PASS_CONFIRMED =", stage.name)
                break

        passed = (
            consecutive_passes >= PASS_CONFIRMATIONS_REQUIRED
        )

        final_stage_eval = evaluate_stage(
            SAC.load(str(best_model_path), device=device),
            stage=stage,
            previous_stage=previous_stage,
            seed_offset=(
                900000 + stage_index * 100000
            ),
        )
        atomic_save_json(
            directory / "final_stage_evaluation.json",
            final_stage_eval,
        )

        # Promotion still requires an independent final evaluation.
        passed = bool(passed and final_stage_eval["gate_passed"])

        if passed:
            pass_record = {
                "stage": asdict(stage),
                "passed_epoch": time.time(),
                "best_model": str(best_model_path),
                "best_replay": str(best_replay_path),
                "final_evaluation": final_stage_eval,
            }
            atomic_save_json(directory / "PASS.json", pass_record)

            confirmed_model = (
                CONFIRMED_ROOT
                / f"stage4_{stage.name}_passed_model.zip"
            )
            confirmed_eval = (
                CONFIRMED_ROOT
                / f"stage4_{stage.name}_evaluation.json"
            )
            shutil.copy2(best_model_path, confirmed_model)
            shutil.copy2(
                directory / "final_stage_evaluation.json",
                confirmed_eval,
            )

            if stage.name not in passed_stages:
                passed_stages.append(stage.name)

            # Continue from the accepted model and replay buffer.
            model = SAC.load(
                str(best_model_path),
                device=device,
            )
            model.load_replay_buffer(str(best_replay_path))
            set_learning_rate(model, joint_lr)
            set_action_noise(model, noise_sigma)
            _MODEL_HOLDER["model"] = model

            stage_index += 1
            hard_seeds = []

            state = build_state(
                global_started_epoch=global_started_epoch,
                stage_index=stage_index,
                chunk=0,
                passed_stages=passed_stages,
                best_model_path=str(best_model_path),
                best_replay_path=str(best_replay_path),
                joint_lr=joint_lr,
                noise_sigma=noise_sigma,
                consecutive_passes=0,
                hard_seeds=hard_seeds,
                status=(
                    "ALL_STAGES_FINISHED"
                    if stage_index >= len(DISTANCE_STAGES)
                    else "ADVANCING_TO_NEXT_STAGE"
                ),
                message=f"Passed {stage.name}.",
            )
            atomic_save_json(STATE_PATH, state)
            print("AUTO_ADVANCE_TO_STAGE_INDEX =", stage_index)
        else:
            attempt_stamp = int(time.time())
            failure_record = {
                "stage": asdict(stage),
                "failed_epoch": time.time(),
                "reason": (
                    "This attempt budget expired or the independent "
                    "final evaluation did not pass."
                ),
                "best_evaluation": best,
                "final_evaluation": final_stage_eval,
                "best_model": str(best_model_path),
                "best_replay": str(best_replay_path),
            }
            atomic_save_json(
                directory / f"FAILED_ATTEMPT_{attempt_stamp}.json",
                failure_record,
            )

            if time.time() < global_deadline:
                # Do not waste the rest of the night and do not promote an
                # unqualified policy. Roll back to this distance's best
                # checkpoint, reduce update aggressiveness and retry the same
                # distance with accumulated collision/timeout seeds.
                joint_lr = max(MIN_JOINT_LR, joint_lr * 0.82)
                noise_sigma = np.maximum(
                    MIN_NOISE,
                    noise_sigma * 0.88,
                )
                model = rollback_model(
                    model_path=best_model_path,
                    replay_path=best_replay_path,
                    env=monitor_env,
                    device=device,
                    joint_lr=joint_lr,
                    noise_sigma=noise_sigma,
                )
                _MODEL_HOLDER["model"] = model

                state = build_state(
                    global_started_epoch=global_started_epoch,
                    stage_index=stage_index,
                    chunk=0,
                    passed_stages=passed_stages,
                    best_model_path=str(best_model_path),
                    best_replay_path=str(best_replay_path),
                    joint_lr=joint_lr,
                    noise_sigma=noise_sigma,
                    consecutive_passes=0,
                    hard_seeds=hard_seeds,
                    status="RETRYING_FAILED_STAGE",
                    message=(
                        f"{stage.name} did not pass this attempt. "
                        "Rolled back and retrying the same distance; "
                        "automatic promotion remains disabled until pass."
                    ),
                )
                atomic_save_json(STATE_PATH, state)
                print("OVERNIGHT_RETRY_SAME_STAGE =", stage.name)
                continue

            state = build_state(
                global_started_epoch=global_started_epoch,
                stage_index=stage_index,
                chunk=chunk,
                passed_stages=passed_stages,
                best_model_path=str(best_model_path),
                best_replay_path=str(best_replay_path),
                joint_lr=joint_lr,
                noise_sigma=noise_sigma,
                consecutive_passes=0,
                hard_seeds=hard_seeds,
                status="GLOBAL_BUDGET_FINISHED_AT_FAILED_STAGE",
                message=(
                    f"Global overnight budget ended at {stage.name}. "
                    "The best checkpoint was preserved and no unsafe "
                    "promotion was performed."
                ),
            )
            atomic_save_json(STATE_PATH, state)
            print("OVERNIGHT_BUDGET_END_AT_STAGE =", stage.name)
            break

    try:
        monitor_env.close()
    except Exception:
        pass

    finished_all = stage_index >= len(DISTANCE_STAGES)
    global_timeout = time.time() >= global_deadline

    summary = {
        "finished_all_stages": finished_all,
        "global_timeout": global_timeout,
        "passed_stages": passed_stages,
        "stopped_stage_index": stage_index,
        "stopped_stage_name": (
            DISTANCE_STAGES[stage_index].name
            if stage_index < len(DISTANCE_STAGES)
            else None
        ),
        "state_file": str(STATE_PATH),
        "elapsed_seconds": time.time() - global_started_epoch,
    }
    atomic_save_json(OUTPUT_ROOT / "overnight_summary.json", summary)

    print("\nSTAGE4_OVERNIGHT = FINISHED")
    print("FINISHED_ALL_STAGES =", finished_all)
    print("GLOBAL_TIMEOUT =", global_timeout)
    print("PASSED_STAGES =", passed_stages)
    print("STATE_FILE =", STATE_PATH)
    print("SUMMARY =", OUTPUT_ROOT / "overnight_summary.json")


if __name__ == "__main__":
    main()
