from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.phase10_stage1a3_terminal_env import (
    Phase10Stage1A3TerminalEnv,
)


ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage2a1_safety_reward_from_a3"
)
SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage1a3_terminal_from_a2_20k/v7_stage1a3_best_model.zip"
)


@dataclass(frozen=True)
class StageConfig:
    name: str
    goal_distance_min: float
    goal_distance_max: float
    goal_angle_deg: float
    goal_clearance_margin: float
    max_steps: int
    chunk_steps: int
    max_chunks: int
    noise_speed: float
    noise_steer: float
    gate_success: float
    gate_collision: float


STAGES = (
    StageConfig(
        name="bridge",
        goal_distance_min=0.50,
        goal_distance_max=0.68,
        goal_angle_deg=30.0,
        goal_clearance_margin=0.43,
        max_steps=600,
        chunk_steps=5_000,
        max_chunks=3,
        noise_speed=0.045,
        noise_steer=0.075,
        gate_success=0.80,
        gate_collision=0.10,
    ),
    StageConfig(
        name="caution",
        goal_distance_min=0.55,
        goal_distance_max=0.74,
        goal_angle_deg=32.0,
        goal_clearance_margin=0.40,
        max_steps=650,
        chunk_steps=5_000,
        max_chunks=4,
        noise_speed=0.040,
        noise_steer=0.070,
        gate_success=0.75,
        gate_collision=0.08,
    ),
    StageConfig(
        name="light",
        goal_distance_min=0.60,
        goal_distance_max=0.82,
        goal_angle_deg=35.0,
        goal_clearance_margin=0.36,
        max_steps=700,
        chunk_steps=5_000,
        max_chunks=5,
        noise_speed=0.035,
        noise_steer=0.060,
        gate_success=0.70,
        gate_collision=0.05,
    ),
)


class SaveEveryCallback(BaseCallback):
    def __init__(
        self,
        output_dir: Path,
        every_steps: int = 5_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.output_dir = output_dir
        self.every_steps = int(every_steps)
        self.last_saved = 0

    def _on_step(self) -> bool:
        current = int(self.num_timesteps)
        if current - self.last_saved >= self.every_steps:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.output_dir / f"checkpoint_{current}"))
            self.last_saved = current
            print("CHECKPOINT_SAVED =", current)
        return True


class Stage2A1SafetyEnv(Phase10Stage1A3TerminalEnv):
    """A3-compatible obstacle-safety curriculum.

    Observation and action semantics are unchanged:
    LiDAR + relative goal + history -> SAC -> forward wheel speed + steering.

    The extra reward is outcome based. It never supplies a planner action or
    steering direction. It only rewards actual clearance improvement and
    penalizes moving quickly toward a close frontal obstacle.
    """

    def __init__(self, stage: str = "bridge") -> None:
        super().__init__(
            max_steps=600,
            wheel_command_limit=4.5,
            goal_distance_min=0.50,
            goal_distance_max=0.68,
            goal_angle_limit_deg=30.0,
            goal_clearance_margin=0.43,
        )

        # Preserve A3 terminal behaviour, but make safety matter earlier.
        self.collision_penalty = 35.0
        self.proximity_distance = 0.42
        self.proximity_scale = 0.55

        self.safety_caution_distance = 0.42
        self.safety_danger_distance = 0.27
        self.safety_critical_distance = 0.19

        self.unsafe_progress_scale = 75.0
        self.risk_speed_scale = 0.55
        self.critical_speed_scale = 1.10
        self.approach_obstacle_scale = 18.0
        self.clearance_gain_scale = 12.0
        self.straight_risk_scale = 0.09
        self.safe_steer_scale = 0.008

        self._current_action = np.zeros(2, dtype=np.float32)
        self._previous_front_clearance = self.lidar_max_range
        self._last_safety_metrics: dict[str, float] = {}
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        selected = next((item for item in STAGES if item.name == stage), None)
        if selected is None:
            raise ValueError(f"invalid Stage2A.1 stage: {stage}")

        self.stage2a1_stage = selected.name
        self.curriculum_stage = f"stage2a1_{selected.name}"
        self.goal_distance_min = selected.goal_distance_min
        self.goal_distance_max = selected.goal_distance_max
        self.goal_angle_limit = math.radians(selected.goal_angle_deg)
        self.goal_clearance_margin = selected.goal_clearance_margin
        self.max_steps = selected.max_steps

    @staticmethod
    def _ray_angles() -> np.ndarray:
        return np.linspace(
            -0.75 * math.pi,
            0.75 * math.pi,
            48,
            dtype=np.float64,
        )

    def _clearance_sectors(self) -> dict[str, float]:
        lidar_m = np.asarray(self._lidar48(), dtype=np.float64) * float(
            self.lidar_max_range
        )
        angles = self._ray_angles()

        front_mask = np.abs(angles) <= math.radians(30.0)
        front_wide_mask = np.abs(angles) <= math.radians(55.0)
        left_mask = (angles >= math.radians(15.0)) & (
            angles <= math.radians(80.0)
        )
        right_mask = (angles <= -math.radians(15.0)) & (
            angles >= -math.radians(80.0)
        )

        return {
            "front_clearance": float(np.min(lidar_m[front_mask])),
            "front_wide_clearance": float(np.min(lidar_m[front_wide_mask])),
            "left_clearance": float(np.min(lidar_m[left_mask])),
            "right_clearance": float(np.min(lidar_m[right_mask])),
            "minimum_clearance": float(np.min(lidar_m)),
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = super().reset(seed=seed, options=options)
        sectors = self._clearance_sectors()
        self._previous_front_clearance = sectors["front_clearance"]
        self._last_safety_metrics = sectors.copy()
        self._current_action[:] = 0.0
        info.update(sectors)
        return observation, info

    @staticmethod
    def _risk_ratio(
        clearance: float,
        caution_distance: float,
        danger_distance: float,
    ) -> float:
        return float(
            np.clip(
                (caution_distance - clearance)
                / max(caution_distance - danger_distance, 1.0e-6),
                0.0,
                1.0,
            )
        )

    def _safety_terms(
        self,
        progress: float,
        front_clearance: float,
        previous_front_clearance: float,
        speed_ratio: float,
        steer_ratio: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        risk = self._risk_ratio(
            front_clearance,
            self.safety_caution_distance,
            self.safety_danger_distance,
        )
        critical = self._risk_ratio(
            front_clearance,
            self.safety_danger_distance,
            self.safety_critical_distance,
        )

        obstacle_approach = max(
            previous_front_clearance - front_clearance,
            0.0,
        )
        clearance_gain = max(
            front_clearance - previous_front_clearance,
            0.0,
        )
        positive_progress = max(float(progress), 0.0)
        abs_steer = abs(float(steer_ratio))

        # Progress through a dangerous corridor should not remain profitable.
        unsafe_progress = (
            -self.unsafe_progress_scale
            * positive_progress
            * risk
        )

        # Slow down before the final collision step, not only after collision.
        risk_speed = (
            -self.risk_speed_scale
            * risk
            * risk
            * speed_ratio
            * speed_ratio
        )
        critical_speed = (
            -self.critical_speed_scale
            * critical
            * critical
            * speed_ratio
            * speed_ratio
        )

        # Penalize reducing frontal clearance while already inside the risk zone.
        obstacle_approach_penalty = (
            -self.approach_obstacle_scale
            * obstacle_approach
            * (0.25 + 0.75 * risk)
            * (0.25 + 0.75 * speed_ratio)
        )

        # Reward only an observed increase in clearance; no steering direction
        # or teacher action is exposed to the policy.
        clearance_gain_reward = (
            self.clearance_gain_scale
            * clearance_gain
            * risk
            * min(abs_steer / 0.30, 1.0)
        )

        # When the frontal sector is risky, moving straight is discouraged.
        straight_fraction = float(
            np.clip((0.25 - abs_steer) / 0.25, 0.0, 1.0)
        )
        straight_risk = (
            -self.straight_risk_scale
            * risk
            * speed_ratio
            * straight_fraction
        )

        # Far from obstacles, retain A3's smooth, mostly straight behaviour.
        safe_steer = (
            -self.safe_steer_scale
            * (1.0 - risk)
            * abs_steer
            * abs_steer
        )

        terms = {
            "unsafe_progress": unsafe_progress,
            "risk_speed": risk_speed,
            "critical_speed": critical_speed,
            "obstacle_approach": obstacle_approach_penalty,
            "clearance_gain": clearance_gain_reward,
            "straight_risk": straight_risk,
            "safe_steer": safe_steer,
        }
        metrics = {
            "front_risk": risk,
            "critical_risk": critical,
            "obstacle_approach_delta": obstacle_approach,
            "clearance_gain_delta": clearance_gain,
        }
        return terms, metrics

    def _reward(
        self,
        previous_goal_dist: float,
        goal_dist: float,
        goal_angle: float,
        collision: bool,
        min_lidar_m: float,
        speed_ratio: float,
        action_delta_sq: float,
    ) -> tuple[float, dict[str, float]]:
        base_reward, terms = super()._reward(
            previous_goal_dist=previous_goal_dist,
            goal_dist=goal_dist,
            goal_angle=goal_angle,
            collision=collision,
            min_lidar_m=min_lidar_m,
            speed_ratio=speed_ratio,
            action_delta_sq=action_delta_sq,
        )

        sectors = self._clearance_sectors()
        progress = float(previous_goal_dist - goal_dist)
        safety_terms, safety_metrics = self._safety_terms(
            progress=progress,
            front_clearance=sectors["front_clearance"],
            previous_front_clearance=self._previous_front_clearance,
            speed_ratio=float(speed_ratio),
            steer_ratio=float(self._current_action[1]),
        )

        terms.update(safety_terms)
        self._last_safety_metrics = {
            **sectors,
            **safety_metrics,
        }
        return float(base_reward + sum(safety_terms.values())), terms

    def step(self, action: np.ndarray):
        clipped = np.clip(
            np.asarray(action, dtype=np.float32).reshape(2),
            -1.0,
            1.0,
        )
        self._current_action = clipped.copy()
        observation, reward, terminated, truncated, info = super().step(clipped)

        info.update(self._last_safety_metrics)
        info["danger_zone"] = float(
            self._last_safety_metrics.get("front_risk", 0.0) > 0.0
        )
        info["critical_zone"] = float(
            self._last_safety_metrics.get("critical_risk", 0.0) > 0.0
        )
        self._previous_front_clearance = self._last_safety_metrics.get(
            "front_clearance",
            self._previous_front_clearance,
        )
        return observation, reward, terminated, truncated, info


class OriginalA3EvalEnv(Phase10Stage1A3TerminalEnv):
    pass


def make_train_env(stage: str, output_dir: Path):
    base = Stage2A1SafetyEnv(stage=stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitored = Monitor(
        base,
        filename=str(output_dir / "monitor.csv"),
        info_keywords=(
            "is_success",
            "collision",
            "goal_dist",
            "goal_angle",
            "min_lidar_m",
            "front_clearance",
            "front_risk",
            "danger_zone",
            "critical_zone",
            "speed_ratio",
            "curriculum_stage",
        ),
    )
    return base, monitored


def validate_reward_shape() -> None:
    env = Stage2A1SafetyEnv(stage="bridge")
    try:
        safe_terms, _ = env._safety_terms(
            progress=0.002,
            front_clearance=0.80,
            previous_front_clearance=0.80,
            speed_ratio=0.75,
            steer_ratio=0.0,
        )
        risky_straight_terms, _ = env._safety_terms(
            progress=0.002,
            front_clearance=0.21,
            previous_front_clearance=0.23,
            speed_ratio=0.75,
            steer_ratio=0.02,
        )
        risky_escape_terms, _ = env._safety_terms(
            progress=0.000,
            front_clearance=0.25,
            previous_front_clearance=0.23,
            speed_ratio=0.35,
            steer_ratio=0.55,
        )

        safe_total = float(sum(safe_terms.values()))
        risky_straight_total = float(sum(risky_straight_terms.values()))
        risky_escape_total = float(sum(risky_escape_terms.values()))

        print("SAFETY_REWARD_SAFE =", round(safe_total, 6))
        print(
            "SAFETY_REWARD_RISKY_STRAIGHT =",
            round(risky_straight_total, 6),
        )
        print(
            "SAFETY_REWARD_RISKY_ESCAPE =",
            round(risky_escape_total, 6),
        )

        assert risky_straight_total < safe_total - 0.20
        assert risky_escape_total > risky_straight_total
        print("STAGE2A1_SAFETY_REWARD_CHECK = PASS")
    finally:
        env.close()


def validate_sampling(stage: str, seed: int) -> None:
    env = Stage2A1SafetyEnv(stage=stage)
    distances: list[float] = []
    angles: list[float] = []
    try:
        for index in range(20):
            _, info = env.reset(seed=seed + index)
            distances.append(float(info["goal_dist"]))
            angles.append(float(info["goal_relative_angle"]))

        print(f"STAGE2A1_{stage.upper()}_SAMPLING_20 = PASS")
        print(
            f"{stage.upper()}_DISTANCE_MIN_MAX = "
            f"{min(distances):.3f}, {max(distances):.3f}"
        )
        print(
            f"{stage.upper()}_ANGLE_DEG_MIN_MAX = "
            f"{math.degrees(min(angles)):.2f}, "
            f"{math.degrees(max(angles)):.2f}"
        )
    finally:
        env.close()


def evaluate(
    model: SAC,
    stage: str,
    episodes: int,
    seed: int,
    original_a3: bool = False,
) -> dict[str, Any]:
    env = OriginalA3EvalEnv() if original_a3 else Stage2A1SafetyEnv(stage=stage)
    rows: list[dict[str, Any]] = []

    success_count = 0
    collision_count = 0
    timeout_count = 0

    try:
        for episode in range(int(episodes)):
            observation, reset_info = env.reset(seed=seed + episode)
            final_info: dict[str, Any] = {}
            minimum_clearance = float("inf")
            minimum_front_clearance = float("inf")
            danger_steps = 0
            critical_steps = 0
            total_steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                final_info = info
                total_steps += 1
                minimum_clearance = min(
                    minimum_clearance,
                    float(info.get("min_lidar_m", 6.0)),
                )
                minimum_front_clearance = min(
                    minimum_front_clearance,
                    float(info.get("front_clearance", 6.0)),
                )
                danger_steps += int(float(info.get("danger_zone", 0.0)) > 0.5)
                critical_steps += int(
                    float(info.get("critical_zone", 0.0)) > 0.5
                )
                if terminated or truncated:
                    break

            success = bool(final_info.get("reach_goal", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(truncated and not success and not collision)
            success_count += int(success)
            collision_count += int(collision)
            timeout_count += int(timeout)

            row = {
                "episode": episode + 1,
                "seed": seed + episode,
                "initial_goal_dist": float(reset_info.get("goal_dist", 999.0)),
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": total_steps,
                "final_goal_dist": float(final_info.get("goal_dist", 999.0)),
                "final_goal_angle": float(final_info.get("goal_angle", 999.0)),
                "minimum_clearance": minimum_clearance,
                "minimum_front_clearance": minimum_front_clearance,
                "danger_fraction": danger_steps / max(total_steps, 1),
                "critical_fraction": critical_steps / max(total_steps, 1),
            }
            rows.append(row)

            print(
                f"EP{episode + 1:03d} stage={stage} original={original_a3} "
                f"success={success} collision={collision} timeout={timeout} "
                f"steps={total_steps} final={row['final_goal_dist']:.3f} "
                f"clearance={minimum_clearance:.3f} "
                f"front={minimum_front_clearance:.3f} "
                f"danger={row['danger_fraction']:.3f}"
            )
    finally:
        env.close()

    summary = {
        "stage": stage,
        "original_a3": original_a3,
        "episodes": int(episodes),
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / int(episodes),
        "collision_rate": collision_count / int(episodes),
        "timeout_rate": timeout_count / int(episodes),
        "mean_final_goal_dist": float(
            np.mean([row["final_goal_dist"] for row in rows])
        ),
        "mean_minimum_clearance": float(
            np.mean([row["minimum_clearance"] for row in rows])
        ),
        "mean_minimum_front_clearance": float(
            np.mean([row["minimum_front_clearance"] for row in rows])
        ),
        "mean_danger_fraction": float(
            np.mean([row["danger_fraction"] for row in rows])
        ),
        "mean_critical_fraction": float(
            np.mean([row["critical_fraction"] for row in rows])
        ),
        "results": rows,
    }
    return summary


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n========== {title} ==========")
    for key in (
        "success_count",
        "collision_count",
        "timeout_count",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_final_goal_dist",
        "mean_minimum_clearance",
        "mean_minimum_front_clearance",
        "mean_danger_fraction",
        "mean_critical_fraction",
    ):
        print(f"{key.upper()} = {summary[key]}")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def set_learning_rate(model: SAC, learning_rate: float) -> None:
    rate = float(learning_rate)
    model.learning_rate = rate
    model.lr_schedule = lambda _: rate

    optimizers = [
        model.policy.actor.optimizer,
        model.policy.critic.optimizer,
    ]
    if getattr(model, "ent_coef_optimizer", None) is not None:
        optimizers.append(model.ent_coef_optimizer)

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = rate


def evaluation_score(summary: dict[str, Any]) -> float:
    return float(
        summary["success_rate"]
        - 2.5 * summary["collision_rate"]
        - 0.25 * summary["timeout_rate"]
        + 0.20 * min(summary["mean_minimum_clearance"] / 0.35, 1.0)
        - 0.15 * summary["mean_critical_fraction"]
    )


def stage_passed(
    config: StageConfig,
    stage_eval: dict[str, Any],
    retention_eval: dict[str, Any],
) -> bool:
    return bool(
        stage_eval["success_rate"] >= config.gate_success
        and stage_eval["collision_rate"] <= config.gate_collision
        and retention_eval["success_rate"] >= 0.80
        and retention_eval["collision_rate"] <= 0.02
    )


def train_stage(
    model: SAC,
    base_env: Stage2A1SafetyEnv,
    train_env: Monitor,
    config: StageConfig,
    total_trained_steps: int,
    device: str,
) -> tuple[SAC, int, dict[str, Any], bool]:
    base_env.set_stage(config.name)
    stage_dir = ROOT / config.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray(
            [config.noise_speed, config.noise_steer],
            dtype=np.float32,
        ),
    )

    best_score = float("-inf")
    best_stage_eval: dict[str, Any] | None = None
    passed = False

    for chunk in range(1, config.max_chunks + 1):
        print("\nTRAIN_STAGE =", config.name)
        print("TRAIN_CHUNK =", chunk, "/", config.max_chunks)
        print("CHUNK_TIMESTEPS =", config.chunk_steps)
        print(
            "ACTION_NOISE_SIGMA =",
            [config.noise_speed, config.noise_steer],
        )

        callback = SaveEveryCallback(
            output_dir=stage_dir / "checkpoints",
            every_steps=config.chunk_steps,
        )
        model.learn(
            total_timesteps=config.chunk_steps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=4,
            progress_bar=False,
        )
        total_trained_steps += config.chunk_steps

        stage_eval = evaluate(
            model,
            stage=config.name,
            episodes=30,
            seed=51000 + STAGES.index(config) * 1000,
            original_a3=False,
        )
        retention_eval = evaluate(
            model,
            stage="a3_original",
            episodes=30,
            seed=10730,
            original_a3=True,
        )
        print_summary(
            f"{config.name.upper()} CHUNK{chunk} STAGE30",
            stage_eval,
        )
        print_summary(
            f"{config.name.upper()} CHUNK{chunk} A3_RETENTION30",
            retention_eval,
        )

        result = {
            "stage": config.name,
            "chunk": chunk,
            "total_trained_steps": total_trained_steps,
            "stage_eval": stage_eval,
            "retention_eval": retention_eval,
        }
        save_json(stage_dir / f"chunk_{chunk}_evaluation.json", result)

        score = evaluation_score(stage_eval)
        print("STAGE_EVALUATION_SCORE =", score)
        if score > best_score:
            best_score = score
            best_stage_eval = stage_eval
            model.save(str(stage_dir / "best_model"))
            model.save_replay_buffer(str(stage_dir / "best_replay.pkl"))
            save_json(stage_dir / "best_evaluation.json", result)
            print("STAGE_BEST_UPDATED =", config.name, chunk)

        passed = stage_passed(config, stage_eval, retention_eval)
        print(
            f"STAGE2A1_{config.name.upper()}_GATE =",
            "PASS" if passed else "FAIL",
        )
        if passed:
            break

    if best_stage_eval is None:
        raise RuntimeError(f"no completed evaluation for stage {config.name}")

    # Continue from the safest/best checkpoint rather than the last checkpoint.
    model = SAC.load(
        str(stage_dir / "best_model.zip"),
        env=train_env,
        device=device,
    )
    model.load_replay_buffer(str(stage_dir / "best_replay.pkl"))
    set_learning_rate(model, 1.5e-5)

    return model, total_trained_steps, best_stage_eval, passed


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("STABLE_BASELINES3_VERSION =", stable_baselines3.__version__)
    print("DEVICE =", device)
    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("SOURCE_POLICY = A3")
    print("ACTION_SEMANTICS_CHANGED = False")
    print("CONTINUOUS_LIDAR_SAFETY_REWARD = True")
    print("FRONTAL_RISK_DISTANCE = 0.42")
    print("DANGER_DISTANCE = 0.27")
    print("CRITICAL_DISTANCE = 0.19")
    print("COLLISION_PENALTY = 35.0")
    print("FRESH_REPLAY_BUFFER = True")
    print("A_STAR = False")
    print("PURE_PURSUIT = False")
    print("WAYPOINT = False")
    print("TEACHER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("HER = False")
    print("MANUAL_KINEMATICS = False")

    validate_reward_shape()
    for index, config in enumerate(STAGES):
        validate_sampling(config.name, seed=46000 + index * 100)

    base_env, train_env = make_train_env(
        stage="bridge",
        output_dir=ROOT / "train",
    )
    model = SAC.load(str(SOURCE_MODEL), env=train_env, device=device)
    set_learning_rate(model, 1.5e-5)

    replay_size = (
        int(model.replay_buffer.size())
        if model.replay_buffer is not None
        else -1
    )
    print("INITIAL_REPLAY_SIZE =", replay_size)
    if replay_size != 0:
        raise RuntimeError(
            "A3 replay buffer was unexpectedly loaded; stop training"
        )

    zero_shot = evaluate(
        model,
        stage="bridge",
        episodes=30,
        seed=47000,
        original_a3=False,
    )
    retention_zero = evaluate(
        model,
        stage="a3_original",
        episodes=30,
        seed=10730,
        original_a3=True,
    )
    print_summary("STAGE2A1 ZERO-SHOT BRIDGE30", zero_shot)
    print_summary("STAGE2A1 ZERO-SHOT A3_RETENTION30", retention_zero)
    save_json(
        ROOT / "zero_shot.json",
        {"bridge": zero_shot, "a3_retention": retention_zero},
    )

    total_trained_steps = 0
    stage_records: dict[str, Any] = {}

    for config in STAGES:
        model, total_trained_steps, best_eval, passed = train_stage(
            model=model,
            base_env=base_env,
            train_env=train_env,
            config=config,
            total_trained_steps=total_trained_steps,
            device=device,
        )
        stage_records[config.name] = {
            "best_eval": best_eval,
            "passed": passed,
        }

        if not passed:
            final = {
                "status": f"STOPPED_AT_{config.name.upper()}",
                "source_model": str(SOURCE_MODEL),
                "total_trained_steps": total_trained_steps,
                "stages": stage_records,
                "recommendation": "KEEP_A3_AND_REVIEW_SAFETY_REWARD",
            }
            save_json(ROOT / "final_report.json", final)
            print("\nSTAGE2A1_TRAINING = STOPPED_AT", config.name.upper())
            print("RECOMMENDATION = KEEP_A3_AND_REVIEW_SAFETY_REWARD")
            train_env.close()
            return

    final_fixed = evaluate(
        model,
        stage="light",
        episodes=50,
        seed=10730,
        original_a3=False,
    )
    final_generalization = evaluate(
        model,
        stage="light",
        episodes=100,
        seed=60730,
        original_a3=False,
    )
    final_retention = evaluate(
        model,
        stage="a3_original",
        episodes=50,
        seed=20730,
        original_a3=True,
    )

    print_summary("STAGE2A1 LIGHT FINAL FIXED50", final_fixed)
    print_summary(
        "STAGE2A1 LIGHT FINAL GENERALIZATION100",
        final_generalization,
    )
    print_summary("STAGE2A1 FINAL A3_RETENTION50", final_retention)

    final_pass = bool(
        final_fixed["success_rate"] >= 0.70
        and final_fixed["collision_rate"] <= 0.05
        and final_generalization["success_rate"] >= 0.65
        and final_generalization["collision_rate"] <= 0.05
        and final_retention["success_rate"] >= 0.80
        and final_retention["collision_rate"] <= 0.02
    )

    model.save(str(ROOT / "stage2a1_best_model"))
    model.save_replay_buffer(str(ROOT / "stage2a1_best_replay.pkl"))

    final = {
        "status": "TRAINED",
        "source_model": str(SOURCE_MODEL),
        "total_trained_steps": total_trained_steps,
        "stages": stage_records,
        "final_fixed": final_fixed,
        "final_generalization": final_generalization,
        "final_a3_retention": final_retention,
        "recommendation": (
            "ENTER_STAGE2B" if final_pass else "KEEP_A3_AND_REVIEW_SAFETY_REWARD"
        ),
    }
    save_json(ROOT / "final_report.json", final)

    print("\nSTAGE2A1_TRAINING = FINISHED")
    print("BEST_MODEL =", ROOT / "stage2a1_best_model.zip")
    print("RECOMMENDATION =", final["recommendation"])
    train_env.close()


if __name__ == "__main__":
    main()
