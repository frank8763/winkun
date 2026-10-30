from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
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


PRIMARY_SOURCE = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "confirmed/stage2b1a4_final_pass_model.zip"
)
FALLBACK_SOURCE = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "confirmed/stage2b1a_pass_candidate_model.zip"
)
OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage3_full_indoor_auto"
)
CONFIRMED_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/confirmed"
)

# Foreground wall-clock budget. Training stops before this and then performs
# final evaluation. On the current machine, total runtime should be roughly
# two to three hours.
TRAINING_BUDGET_SECONDS = 2 * 60 * 60 + 10 * 60
MAX_CHUNKS = 14
CHUNK_TIMESTEPS = 45_000
LEARNING_RATE = 1.5e-6

FULL_TASKS: dict[str, dict[str, float | int | str]] = {
    "indoor_near": {
        "base_task": "light",
        "goal_distance_min": 1.50,
        "goal_distance_max": 2.50,
        "goal_angle_deg": 90.0,
        "goal_clearance_margin": 0.34,
        "max_steps": 1600,
    },
    "indoor_mid": {
        "base_task": "light",
        "goal_distance_min": 2.50,
        "goal_distance_max": 4.00,
        "goal_angle_deg": 150.0,
        "goal_clearance_margin": 0.32,
        "max_steps": 2600,
    },
    "indoor_full": {
        "base_task": "light",
        "goal_distance_min": 3.50,
        "goal_distance_max": 5.50,
        "goal_angle_deg": 180.0,
        "goal_clearance_margin": 0.30,
        "max_steps": 3600,
    },
}

PHASE_PROBABILITIES: tuple[dict[str, float], ...] = (
    {
        "a3": 0.10,
        "bridge": 0.10,
        "caution": 0.08,
        "light": 0.12,
        "indoor_near": 0.42,
        "indoor_mid": 0.15,
        "indoor_full": 0.03,
    },
    {
        "a3": 0.08,
        "bridge": 0.08,
        "caution": 0.07,
        "light": 0.10,
        "indoor_near": 0.30,
        "indoor_mid": 0.29,
        "indoor_full": 0.08,
    },
    {
        "a3": 0.06,
        "bridge": 0.06,
        "caution": 0.06,
        "light": 0.08,
        "indoor_near": 0.20,
        "indoor_mid": 0.34,
        "indoor_full": 0.20,
    },
    {
        "a3": 0.05,
        "bridge": 0.05,
        "caution": 0.05,
        "light": 0.07,
        "indoor_near": 0.13,
        "indoor_mid": 0.30,
        "indoor_full": 0.35,
    },
)


def choose_source() -> Path:
    if PRIMARY_SOURCE.is_file():
        return PRIMARY_SOURCE
    if FALLBACK_SOURCE.is_file():
        return FALLBACK_SOURCE
    raise FileNotFoundError(
        f"missing both source models: {PRIMARY_SOURCE}, {FALLBACK_SOURCE}"
    )


def xy_from_env(env: Any) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    data = getattr(base, "data", None)
    if data is None or len(data.qpos) < 2:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(data.qpos[:2], dtype=np.float64).copy()


class FullIndoorEnv(Stage2B1Env):
    """Strict MuJoCo full-room curriculum with long goals and no planner."""

    def __init__(
        self,
        fixed_task: str | None = None,
        seed: int = 83000,
    ) -> None:
        super().__init__(fixed_task=None, seed=seed)
        self.fixed_full_task = fixed_task
        self.training_phase = 0
        self._full_rng = np.random.default_rng(seed + 1701)
        self._previous_goal_distance = 999.0
        self._previous_xy = np.zeros(2, dtype=np.float64)
        self._idle_steps = 0

    def set_training_phase(self, phase: int) -> None:
        self.training_phase = int(
            np.clip(phase, 0, len(PHASE_PROBABILITIES) - 1)
        )

    def _apply_task(self, task: str) -> None:
        if task not in FULL_TASKS:
            OvernightObstacleEnv._apply_task(self, task)
            return

        config = FULL_TASKS[task]

        # Reuse the proven obstacle/safety configuration first, then replace
        # only the goal geometry and episode length.
        OvernightObstacleEnv._apply_task(
            self,
            str(config["base_task"]),
        )

        self._active_task = task
        self.curriculum_stage = f"stage3_{task}"
        self.goal_distance_min = float(config["goal_distance_min"])
        self.goal_distance_max = float(config["goal_distance_max"])
        self.goal_angle_limit = math.radians(
            float(config["goal_angle_deg"])
        )
        self.goal_clearance_margin = float(
            config["goal_clearance_margin"]
        )
        self.max_steps = int(config["max_steps"])

    def _sample_training_task(self) -> str:
        probabilities = PHASE_PROBABILITIES[self.training_phase]
        tasks = list(probabilities)
        probs = np.asarray(
            [probabilities[task] for task in tasks],
            dtype=np.float64,
        )
        probs /= probs.sum()
        return str(self._full_rng.choice(tasks, p=probs))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        forced_task = None
        if options is not None:
            forced_task = options.get("task_mode")

        if forced_task is not None:
            task = str(forced_task)
            sampling_mode = "forced"
        elif self.fixed_full_task is not None:
            task = str(self.fixed_full_task)
            sampling_mode = "evaluation"
        else:
            task = self._sample_training_task()
            sampling_mode = f"phase_{self.training_phase}"

        actual_seed = int(
            seed
            if seed is not None
            else self._full_rng.integers(300000, 2_000_000_000)
        )

        observation, info = OvernightObstacleEnv.reset(
            self,
            seed=actual_seed,
            options={"task_mode": task},
        )

        self._sampling_mode = sampling_mode
        self._actual_training_seed = actual_seed
        self._last_front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        self._previous_action.fill(0.0)
        self._previous_delta.fill(0.0)
        self._previous_goal_distance = float(
            info.get("goal_dist", 999.0)
        )
        self._previous_xy = xy_from_env(self)
        self._idle_steps = 0

        info["task_mode"] = task
        info["sampling_mode"] = sampling_mode
        info["training_seed"] = actual_seed
        info["training_phase"] = self.training_phase
        info["actual_speed_mps"] = 0.0
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            action
        )

        current_xy = xy_from_env(self)
        displacement = float(
            np.linalg.norm(current_xy - self._previous_xy)
        )
        dt = float(max(getattr(self, "dt", 0.02), 1e-6))
        actual_speed = displacement / dt

        goal_distance = float(info.get("goal_dist", 999.0))
        goal_angle = float(info.get("goal_angle", 0.0))
        front_clearance = float(
            info.get("front_clearance", self.lidar_max_range)
        )
        progress = self._previous_goal_distance - goal_distance
        collision = bool(info.get("collision", False))
        active_task = str(getattr(self, "_active_task", ""))

        stage3_progress_reward = 0.0
        stage3_time_penalty = 0.0
        stage3_idle_penalty = 0.0
        stage3_cruise_bonus = 0.0

        if active_task in FULL_TASKS and not collision:
            # Reward actual world-space progress, not a commanded velocity.
            stage3_progress_reward = 0.012 * float(
                np.clip(progress / 0.004, -1.0, 1.0)
            )
            stage3_time_penalty = -0.0015

            aligned = abs(goal_angle) <= math.radians(50.0)
            safe = front_clearance >= 0.42
            far = goal_distance >= 0.60

            if aligned and safe and far and progress > 0.0:
                stage3_cruise_bonus = 0.0025 * float(
                    np.clip(actual_speed / 0.28, 0.0, 1.0)
                )

            if safe and far and actual_speed < 0.025:
                self._idle_steps += 1
            else:
                self._idle_steps = max(0, self._idle_steps - 2)

            if self._idle_steps >= 25:
                stage3_idle_penalty = -0.004 * float(
                    np.clip(
                        (self._idle_steps - 24) / 30.0,
                        0.0,
                        1.0,
                    )
                )

        reward = float(
            reward
            + stage3_progress_reward
            + stage3_time_penalty
            + stage3_idle_penalty
            + stage3_cruise_bonus
        )

        self._previous_goal_distance = goal_distance
        self._previous_xy = current_xy

        info["task_mode"] = active_task
        info["sampling_mode"] = self._sampling_mode
        info["training_seed"] = self._actual_training_seed
        info["training_phase"] = self.training_phase
        info["actual_speed_mps"] = actual_speed
        info["stage3_progress"] = progress
        info["stage3_progress_reward"] = stage3_progress_reward
        info["stage3_time_penalty"] = stage3_time_penalty
        info["stage3_idle_penalty"] = stage3_idle_penalty
        info["stage3_cruise_bonus"] = stage3_cruise_bonus

        return observation, reward, terminated, truncated, info


def configure_model(model: SAC) -> dict[str, Any]:
    # Preserve the proven LiDAR feature representation. Long-distance
    # navigation is the same local closed-loop problem repeated for longer,
    # so only speed/steering outputs are adapted first.
    for parameter in model.actor.parameters():
        parameter.requires_grad = False

    model.actor.mu.weight.requires_grad = True
    model.actor.mu.bias.requires_grad = True

    if int(model.actor.mu.out_features) != 2:
        raise RuntimeError(
            f"expected speed+steering outputs, got "
            f"{model.actor.mu.out_features}"
        )

    set_learning_rate(model, LEARNING_RATE)

    return {
        "trainable_actor_parameters": [
            name
            for name, parameter in model.actor.named_parameters()
            if parameter.requires_grad
        ],
        "lidar_feature_network_frozen": True,
        "scheduled_learning_rate": LEARNING_RATE,
        "reward_additions": [
            "world_progress",
            "time_efficiency",
            "safe_cruise",
            "safe_idle_penalty",
        ],
    }


def spawn_audit(
    task: str,
    episodes: int = 20,
    seed: int = 930000,
) -> dict[str, Any]:
    env = FullIndoorEnv(fixed_task=task, seed=seed)
    distances: list[float] = []

    try:
        for episode in range(episodes):
            _, info = env.reset(seed=seed + episode)
            distances.append(float(info.get("goal_dist", 999.0)))
    finally:
        env.close()

    result = {
        "task": task,
        "episodes": episodes,
        "minimum": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "maximum": float(np.max(distances)),
        "distances": distances,
    }
    print(f"\n========== SPAWN AUDIT {task.upper()} ==========")
    print("MIN_INITIAL_GOAL_DISTANCE =", result["minimum"])
    print("MEAN_INITIAL_GOAL_DISTANCE =", result["mean"])
    print("MAX_INITIAL_GOAL_DISTANCE =", result["maximum"])
    return result


def evaluate_task(
    model: SAC,
    task: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    env = FullIndoorEnv(fixed_task=task, seed=seed + 9000)
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            actual_seed = seed + episode
            observation, reset_info = env.reset(seed=actual_seed)
            initial_goal_distance = float(
                reset_info.get("goal_dist", 999.0)
            )
            final_info: dict[str, Any] = {}
            path_length = 0.0
            speed_sum = 0.0
            speed_count = 0
            minimum_clearance = float("inf")
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
                path_length += speed * float(
                    max(getattr(env, "dt", 0.02), 1e-6)
                )
                speed_sum += speed
                speed_count += 1
                minimum_clearance = min(
                    minimum_clearance,
                    float(info.get("min_lidar_m", 99.0)),
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
            straight_distance = max(initial_goal_distance, 1e-6)
            path_efficiency = (
                straight_distance / max(path_length, straight_distance)
            )

            rows.append(
                {
                    "seed": actual_seed,
                    "initial_goal_distance": initial_goal_distance,
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "final_goal_distance": float(
                        final_info.get("goal_dist", 999.0)
                    ),
                    "path_length": path_length,
                    "path_efficiency": path_efficiency,
                    "mean_speed_mps": (
                        speed_sum / max(speed_count, 1)
                    ),
                    "minimum_clearance": minimum_clearance,
                }
            )
    finally:
        env.close()

    count = len(rows)
    result = {
        "task": task,
        "episodes": count,
        "success_count": sum(int(row["success"]) for row in rows),
        "collision_count": sum(
            int(row["collision"]) for row in rows
        ),
        "timeout_count": sum(int(row["timeout"]) for row in rows),
        "success_rate": (
            sum(int(row["success"]) for row in rows) / count
        ),
        "collision_rate": (
            sum(int(row["collision"]) for row in rows) / count
        ),
        "timeout_rate": (
            sum(int(row["timeout"]) for row in rows) / count
        ),
        "mean_initial_goal_distance": float(
            np.mean(
                [row["initial_goal_distance"] for row in rows]
            )
        ),
        "mean_final_goal_distance": float(
            np.mean([row["final_goal_distance"] for row in rows])
        ),
        "mean_path_length": float(
            np.mean([row["path_length"] for row in rows])
        ),
        "mean_path_efficiency": float(
            np.mean([row["path_efficiency"] for row in rows])
        ),
        "mean_speed_mps": float(
            np.mean([row["mean_speed_mps"] for row in rows])
        ),
        "mean_minimum_clearance": float(
            np.mean([row["minimum_clearance"] for row in rows])
        ),
        "results": rows,
    }

    print(f"\n========== EVAL {task.upper()} {count} ==========")
    print("SUCCESS_RATE =", result["success_rate"])
    print("COLLISION_RATE =", result["collision_rate"])
    print("TIMEOUT_RATE =", result["timeout_rate"])
    print(
        "MEAN_INITIAL_GOAL_DISTANCE =",
        result["mean_initial_goal_distance"],
    )
    print("MEAN_SPEED_MPS =", result["mean_speed_mps"])
    print(
        "MEAN_PATH_EFFICIENCY =",
        result["mean_path_efficiency"],
    )
    return result


def evaluate_candidate(
    model: SAC,
    label: str,
    seed_offset: int,
    full: bool = False,
) -> dict[str, Any]:
    episodes = (
        {
            "a3": 40,
            "light": 50,
            "indoor_near": 70,
            "indoor_mid": 70,
            "indoor_full": 80,
        }
        if full
        else {
            "a3": 15,
            "light": 20,
            "indoor_near": 20,
            "indoor_mid": 20,
            "indoor_full": 20,
        }
    )
    seeds = {
        "a3": 510000,
        "light": 520000,
        "indoor_near": 530000,
        "indoor_mid": 540000,
        "indoor_full": 550000,
    }

    tasks = {
        task: evaluate_task(
            model,
            task=task,
            episodes=episodes[task],
            seed=seeds[task] + seed_offset,
        )
        for task in (
            "a3",
            "light",
            "indoor_near",
            "indoor_mid",
            "indoor_full",
        )
    }

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

    weighted_success = (
        0.10 * float(tasks["a3"]["success_rate"])
        + 0.10 * float(tasks["light"]["success_rate"])
        + 0.22 * float(tasks["indoor_near"]["success_rate"])
        + 0.26 * float(tasks["indoor_mid"]["success_rate"])
        + 0.32 * float(tasks["indoor_full"]["success_rate"])
    )
    weighted_speed = (
        0.20 * float(tasks["indoor_near"]["mean_speed_mps"])
        + 0.30 * float(tasks["indoor_mid"]["mean_speed_mps"])
        + 0.50 * float(tasks["indoor_full"]["mean_speed_mps"])
    )
    maximum_collision = max(
        float(tasks[name]["collision_rate"])
        for name in tasks
    )

    safety_ok = bool(
        hard_collision_count == 0
        and not bool(focus["collision"])
        and tasks["a3"]["collision_rate"] == 0.0
        and tasks["light"]["collision_rate"] <= 0.05
        and tasks["indoor_near"]["collision_rate"] <= 0.10
        and tasks["indoor_mid"]["collision_rate"] <= 0.12
        and tasks["indoor_full"]["collision_rate"] <= 0.15
    )

    stage3_gate = bool(
        safety_ok
        and tasks["a3"]["success_rate"] >= 0.70
        and tasks["light"]["success_rate"] >= 0.60
        and tasks["indoor_near"]["success_rate"] >= 0.55
        and tasks["indoor_mid"]["success_rate"] >= 0.35
        and tasks["indoor_full"]["success_rate"] >= 0.20
        and tasks["indoor_near"]["mean_speed_mps"] >= 0.10
        and tasks["indoor_mid"]["mean_speed_mps"] >= 0.10
        and tasks["indoor_full"]["mean_speed_mps"] >= 0.09
    )

    result = {
        "label": label,
        "full_evaluation": full,
        "tasks": tasks,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "hard_collision_count": hard_collision_count,
        "weighted_success": weighted_success,
        "weighted_speed": weighted_speed,
        "maximum_collision": maximum_collision,
        "safety_ok": safety_ok,
        "stage3_gate": stage3_gate,
        "selection_key": [
            0 if safety_ok else 1,
            hard_collision_count,
            1 if focus["collision"] else 0,
            maximum_collision,
            -float(tasks["indoor_full"]["success_rate"]),
            -float(tasks["indoor_mid"]["success_rate"]),
            -float(tasks["indoor_near"]["success_rate"]),
            -weighted_success,
            -weighted_speed,
        ],
    }

    print(f"\n========== STAGE3 {label} ==========")
    for task in (
        "a3",
        "light",
        "indoor_near",
        "indoor_mid",
        "indoor_full",
    ):
        print(
            f"{task.upper()}_SUCCESS_RATE =",
            tasks[task]["success_rate"],
        )
        print(
            f"{task.upper()}_COLLISION_RATE =",
            tasks[task]["collision_rate"],
        )
        print(
            f"{task.upper()}_MEAN_SPEED_MPS =",
            tasks[task]["mean_speed_mps"],
        )

    print("HARD_COLLISION_COUNT =", hard_collision_count)
    print("FOCUS_COLLISION =", focus["collision"])
    print("WEIGHTED_SUCCESS =", weighted_success)
    print("WEIGHTED_SPEED =", weighted_speed)
    print("STAGE3_GATE =", "PASS" if stage3_gate else "FAIL")
    return result


def is_better(
    candidate: dict[str, Any],
    best: dict[str, Any],
) -> bool:
    return tuple(candidate["selection_key"]) < tuple(
        best["selection_key"]
    )


def main() -> None:
    source_model = choose_source()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.monotonic()
    training_deadline = start_time + TRAINING_BUDGET_SECONDS

    print("SOURCE_MODEL =", source_model)
    print("DEVICE =", device)
    print("STAGE = STAGE3_FULL_INDOOR_AUTO")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("A_STAR_OR_PLANNER = False")
    print("REAL_MUJOCO_ACTUATORS_ONLY = True")
    print("GOAL_DISTANCE_RANGE = 1.5_TO_5.5_METERS")
    print("GOAL_ANGLE_RANGE = PLUS_MINUS_180_DEGREES")
    print("TRAINING_BUDGET_SECONDS =", TRAINING_BUDGET_SECONDS)
    print("MAX_CHUNKS =", MAX_CHUNKS)
    print("CHUNK_TIMESTEPS =", CHUNK_TIMESTEPS)

    audits = {
        task: spawn_audit(task)
        for task in ("indoor_near", "indoor_mid", "indoor_full")
    }
    save_json(OUTPUT_ROOT / "spawn_audit.json", audits)

    # Refuse to silently train another short-distance environment.
    if audits["indoor_full"]["mean"] < 3.0:
        raise RuntimeError(
            "FULL INDOOR SPAWN AUDIT FAILED: mean indoor_full "
            f"distance is only {audits['indoor_full']['mean']:.3f} m. "
            "The environment is still clamping goals to a short range."
        )

    source = SAC.load(str(source_model), device=device)
    baseline = evaluate_candidate(
        source,
        label="SOURCE_SHORT_RANGE_POLICY",
        seed_offset=0,
        full=False,
    )
    save_json(OUTPUT_ROOT / "baseline_evaluation.json", baseline)

    source.save(str(OUTPUT_ROOT / "best_model"))
    save_json(OUTPUT_ROOT / "best_evaluation.json", baseline)
    save_json(
        OUTPUT_ROOT / "best_metadata.json",
        {
            "source": "SOURCE_SHORT_RANGE_POLICY",
            "chunk": 0,
        },
    )

    base_env = FullIndoorEnv(seed=83000)
    monitor_env = Monitor(
        base_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
    )

    model = SAC.load(
        str(source_model),
        env=monitor_env,
        device=device,
    )
    configure_report = configure_model(model)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.asarray([0.010, 0.018], dtype=np.float32),
    )

    save_json(
        OUTPUT_ROOT / "configure_report.json",
        configure_report,
    )
    print("CONFIGURE_REPORT =", configure_report)
    print("ACTION_NOISE_SIGMA = [0.010, 0.018]")

    best = baseline
    records = [baseline]
    no_improvement = 0

    for chunk in range(1, MAX_CHUNKS + 1):
        if time.monotonic() >= training_deadline:
            print("STAGE3_STOP = TRAINING_TIME_BUDGET_REACHED")
            break

        phase = min(
            (chunk - 1) // max(1, MAX_CHUNKS // 4),
            len(PHASE_PROBABILITIES) - 1,
        )
        base_env.set_training_phase(phase)

        print("\nTRAIN_CHUNK =", chunk, "/", MAX_CHUNKS)
        print("TRAINING_PHASE =", phase)
        print(
            "PHASE_PROBABILITIES =",
            json.dumps(PHASE_PROBABILITIES[phase]),
        )
        print("CHUNK_TIMESTEPS =", CHUNK_TIMESTEPS)

        model.learn(
            total_timesteps=CHUNK_TIMESTEPS,
            reset_num_timesteps=False,
            log_interval=8,
            progress_bar=False,
        )

        record = evaluate_candidate(
            model,
            label=f"CHUNK{chunk}_PHASE{phase}",
            seed_offset=chunk * 10000,
            full=False,
        )
        records.append(record)

        save_json(
            OUTPUT_ROOT / f"chunk_{chunk}_evaluation.json",
            record,
        )
        model.save(str(OUTPUT_ROOT / f"chunk_{chunk}_model"))

        if is_better(record, best):
            best = record
            no_improvement = 0
            model.save(str(OUTPUT_ROOT / "best_model"))
            if model.replay_buffer is not None:
                model.save_replay_buffer(
                    str(OUTPUT_ROOT / "best_replay.pkl")
                )
            save_json(
                OUTPUT_ROOT / "best_evaluation.json",
                record,
            )
            save_json(
                OUTPUT_ROOT / "best_metadata.json",
                {
                    "source": f"CHUNK{chunk}_PHASE{phase}",
                    "chunk": chunk,
                    "phase": phase,
                },
            )
            print("STAGE3_BEST_UPDATED =", chunk)
        else:
            no_improvement += 1

        if (
            record["hard_collision_count"] > 0
            or bool(record["focus"]["collision"])
            or record["tasks"]["a3"]["collision_rate"] > 0.05
            or record["tasks"]["indoor_full"]["collision_rate"] > 0.25
        ):
            print("STAGE3_STOP = SAFETY_DEGRADATION")
            break

        # Do not stop early merely because the first phase passes. Continue
        # until the full-room phase has been trained.
        if no_improvement >= 4 and phase >= 3:
            print("STAGE3_STOP = NO_IMPROVEMENT_IN_FULL_PHASE")
            break

    monitor_env.close()

    print("\n========== FINAL FULL-ROOM EVALUATION ==========")
    final_model = SAC.load(
        str(OUTPUT_ROOT / "best_model.zip"),
        device=device,
    )
    final = evaluate_candidate(
        final_model,
        label="FINAL_BEST_CONFIRM",
        seed_offset=900000,
        full=True,
    )

    save_json(OUTPUT_ROOT / "final_evaluation.json", final)

    if final["safety_ok"]:
        shutil.copy2(
            OUTPUT_ROOT / "best_model.zip",
            CONFIRMED_ROOT / "stage3_full_indoor_best_model.zip",
        )
        shutil.copy2(
            OUTPUT_ROOT / "final_evaluation.json",
            CONFIRMED_ROOT / "stage3_full_indoor_evaluation.json",
        )

    recommendation = (
        "OPEN_MUJOCO_FULL_ROOM_VISUALIZATION"
        if final["stage3_gate"]
        else "KEEP_STAGE3_BEST_AND_CONTINUE_FULL_ROOM_ONLY"
    )

    report = {
        "source_model": str(source_model),
        "full_tasks": FULL_TASKS,
        "spawn_audit": audits,
        "configure_report": configure_report,
        "records": records,
        "final": final,
        "elapsed_seconds": time.monotonic() - start_time,
        "recommendation": recommendation,
    }
    save_json(OUTPUT_ROOT / "final_report.json", report)

    print("\nSTAGE3_FULL_INDOOR = FINISHED")
    print("BEST_MODEL =", OUTPUT_ROOT / "best_model.zip")
    print("FINAL_EVALUATION =", OUTPUT_ROOT / "final_evaluation.json")
    print("FINAL_REPORT =", OUTPUT_ROOT / "final_report.json")
    print(
        "CONFIRMED_MODEL =",
        CONFIRMED_ROOT / "stage3_full_indoor_best_model.zip",
    )
    print("RECOMMENDATION =", recommendation)


if __name__ == "__main__":
    main()
