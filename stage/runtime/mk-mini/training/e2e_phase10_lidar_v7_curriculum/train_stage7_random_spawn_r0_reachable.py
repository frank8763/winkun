from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

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
    DistanceStage,
    atomic_save_json,
    freeze_actor_with_hooks,
    make_fresh_model,
    set_action_noise,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage5_terminal_curriculum_daytime import (
    TerminalCurriculumEnv,
    clamp_entropy,
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage7_random_spawn_r1 import (
    RandomSpawnR1Env,
    evaluate_legacy_task,
    evaluate_random_env,
    resolve_source_model,
    save_model_and_replay,
)


OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_random_spawn_r0_reachable"
)
STATE_PATH = OUTPUT_ROOT / "state.json"
PRECHECK_PATH = OUTPUT_ROOT / "precheck.json"
BEST_MODEL_PATH = OUTPUT_ROOT / "best_model.zip"
BEST_REPLAY_PATH = OUTPUT_ROOT / "best_replay.pkl"
BEST_EVAL_PATH = OUTPUT_ROOT / "best_evaluation.json"
LATEST_EVAL_PATH = OUTPUT_ROOT / "latest_evaluation.json"

CONFIRMED_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/confirmed"
)
CONFIRMED_MODEL = (
    CONFIRMED_ROOT
    / "stage7_R0_reachable_1p20_1p60_angle8_passed_model.zip"
)
CONFIRMED_EVAL = (
    CONFIRMED_ROOT
    / "stage7_R0_reachable_1p20_1p60_angle8_evaluation.json"
)

TURN_RADIUS_M = 4.2
SPAWN_CLEARANCE_MIN_M = 0.90
ARC_CORRIDOR_MARGIN_M = 0.32
ARC_START_CHECK_M = 0.55
KINEMATIC_MARGIN_M = 0.02

R0_STAGE = DistanceStage(
    name="R0_REACHABLE_1p20_1p60_A8_C090",
    goal_min=1.20,
    goal_max=1.60,
    angle_deg=8.0,
    max_steps=1600,
    clearance_margin=0.40,
    stage_budget_minutes=0,
    success_gate=0.70,
    collision_gate=0.05,
    progress_gate=0.70,
    speed_gate=0.030,
    timeout_gate=0.30,
)


R0_BOOTSTRAP_STAGE = DistanceStage(
    name="R0_BOOTSTRAP_F2_FIXED_SPAWN",
    goal_min=0.95,
    goal_max=1.05,
    angle_deg=10.0,
    max_steps=1450,
    clearance_margin=0.47,
    stage_budget_minutes=0,
    success_gate=0.0,
    collision_gate=1.0,
    progress_gate=0.0,
    speed_gate=0.0,
    timeout_gate=1.0,
)

PRECHECK_EPISODES = 240
CURRENT_EVAL_EPISODES = 64
FULL_HOLDOUT_EPISODES = 40
A3_EVAL_EPISODES = 16
LIGHT_EVAL_EPISODES = 20

CRITIC_WARMUP_TIMESTEPS = 10_000
ONE_CHUNK_TIMESTEPS = 5_000
LEARNING_RATE = 5.0e-7
ACTION_NOISE = np.asarray([0.0, 0.0012], dtype=np.float32)
ENTROPY_FLOOR = 0.0030

TASK_PROBABILITIES = {
    R0_STAGE.name: 0.45,
    "a3": 0.25,
    "light": 0.25,
    "caution": 0.05,
}

FULL_RANDOM_SUCCESS_FLOOR = 0.10
FULL_RANDOM_COLLISION_CEILING = 0.12


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class ReachableRandomSpawnR0Env(RandomSpawnR1Env):
    """Random spawn with car-reachable and scene-safe goal sampling."""

    def __init__(self, seed: int) -> None:
        super().__init__(
            seed=seed,
            minimum_spawn_clearance=SPAWN_CLEARANCE_MIN_M,
            goal_min=R0_STAGE.goal_min,
            goal_max=R0_STAGE.goal_max,
            goal_angle_deg=R0_STAGE.angle_deg,
            goal_clearance_margin=R0_STAGE.clearance_margin,
            max_steps=R0_STAGE.max_steps,
        )
        self.stage = R0_STAGE
        self.fixed_curriculum_task = R0_STAGE.name

        # Do not start braking at 0.62 m for a 1.2-1.6 m reachable task.
        self.success_radius = 0.30
        self.terminal_zone = 0.45
        self.terminal_min_speed = 0.018
        self.terminal_entry_speed = 0.110
        self.terminal_overspeed_scale = 0.12
        self.terminal_brake_bonus_scale = 0.008

        self.safe_cruise_scale = 0.010
        self.idle_scale = 0.016
        self.risk_scale = 0.15
        self.speed_risk_scale = 0.18
        self._hard_seed_probability = 0.0

        self.last_effective_radius_m = float("nan")
        self.last_kinematic_margin_m = float("nan")
        self.last_arc_corridor_margin_m = float("nan")
        self.last_arc_checked_points = 0


    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Initialize bookkeeping at the legacy spawn, then randomize spawn.

        The bootstrap reset uses the already verified F2 geometry only to let
        the inherited environment initialize RNG, history and reward state.
        Its goal is discarded immediately. R0 goal sampling is performed only
        after a safe randomized vehicle pose has been applied.
        """
        original_stage = self.stage
        original_fixed_task = self.fixed_curriculum_task

        self.stage = R0_BOOTSTRAP_STAGE
        self.fixed_curriculum_task = R0_BOOTSTRAP_STAGE.name
        self._r0_bootstrap_active = True
        try:
            TerminalCurriculumEnv.reset(
                self,
                seed=seed,
                options={"task_mode": R0_BOOTSTRAP_STAGE.name},
            )
        finally:
            self._r0_bootstrap_active = False
            self.stage = original_stage
            self.fixed_curriculum_task = original_fixed_task

        actual_seed = int(getattr(self, "_actual_seed", seed or 0))
        local_rng = np.random.default_rng(actual_seed + 71_000_003)
        desired_sign = -1 if actual_seed % 2 == 0 else 1
        indices = local_rng.permutation(len(self.random_spawn_pool))
        last_error = "none"

        for raw_index in indices[:96]:
            spawn = self.random_spawn_pool[int(raw_index)]
            self._apply_spawn(spawn)

            try:
                observation, info = self._rebuild_reset_state(
                    actual_seed=actual_seed,
                    spawn=spawn,
                    desired_sign=desired_sign,
                )
                info["spawn_pool_index"] = int(raw_index)
                info["bootstrap_task"] = R0_BOOTSTRAP_STAGE.name
                info["bootstrap_goal_discarded"] = True
                return observation, info
            except RuntimeError as exc:
                last_error = str(exc)

        raise RuntimeError(
            "ReachableRandomSpawnR0Env could not build a balanced, "
            "car-reachable random-spawn episode: " + last_error
        )

    def _arc_corridor_safe(
        self,
        *,
        distance: float,
        relative_angle: float,
        sensor_xy: np.ndarray,
        lidar_m: np.ndarray,
        yaw: float,
    ) -> tuple[bool, float, int]:
        lidar = np.asarray(lidar_m, dtype=np.float64).reshape(-1)
        ray_angles = np.linspace(
            -0.75 * math.pi,
            0.75 * math.pi,
            len(lidar),
            dtype=np.float64,
        )

        alpha = float(relative_angle)
        abs_alpha = abs(alpha)

        if abs_alpha < math.radians(0.25):
            local_points = [
                (float(s), 0.0)
                for s in np.linspace(
                    ARC_START_CHECK_M,
                    float(distance),
                    18,
                )
            ]
            effective_radius = float("inf")
        else:
            effective_radius = float(
                distance / (2.0 * math.sin(abs_alpha))
            )
            if effective_radius + 1e-9 < TURN_RADIUS_M:
                return False, float("-inf"), 0

            sign = 1.0 if alpha > 0.0 else -1.0
            theta = 2.0 * abs_alpha
            local_points = []
            for t in np.linspace(0.0, theta, 24):
                x_local = effective_radius * math.sin(float(t))
                if x_local < ARC_START_CHECK_M:
                    continue
                y_local = (
                    sign
                    * effective_radius
                    * (1.0 - math.cos(float(t)))
                )
                local_points.append((x_local, y_local))

        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        center_xy = np.asarray(self.data.qpos[:2], dtype=np.float64)

        minimum_margin = float("inf")
        checked = 0

        for x_local, y_local in local_points:
            point_world = center_xy + np.asarray(
                [
                    cos_yaw * x_local - sin_yaw * y_local,
                    sin_yaw * x_local + cos_yaw * y_local,
                ],
                dtype=np.float64,
            )
            delta = point_world - np.asarray(
                sensor_xy,
                dtype=np.float64,
            )
            sensor_distance = float(np.linalg.norm(delta))
            if sensor_distance <= 0.05:
                continue

            relative_ray_angle = wrap(
                math.atan2(float(delta[1]), float(delta[0])) - yaw
            )
            if not (
                float(ray_angles[0])
                <= relative_ray_angle
                <= float(ray_angles[-1])
            ):
                return False, float("-inf"), checked

            ray_index = int(
                np.argmin(np.abs(ray_angles - relative_ray_angle))
            )
            lo = max(0, ray_index - self.lidar_sector_half_width)
            hi = min(
                len(lidar),
                ray_index + self.lidar_sector_half_width + 1,
            )
            sector_clearance = float(np.min(lidar[lo:hi]))
            margin = sector_clearance - sensor_distance
            minimum_margin = min(minimum_margin, margin)
            checked += 1

            if margin < ARC_CORRIDOR_MARGIN_M:
                return False, minimum_margin, checked

        if checked == 0:
            return False, float("-inf"), 0
        return True, minimum_margin, checked

    def _candidate_reachable_and_safe(
        self,
        *,
        candidate: np.ndarray,
        distance: float,
        relative_angle: float,
        sensor_xy: np.ndarray,
        lidar_m: np.ndarray,
        yaw: float,
    ) -> bool:
        safe, clearance, sensor_distance, _ = self._candidate_is_safe(
            candidate,
            sensor_xy,
            lidar_m,
            yaw,
        )
        if not safe:
            return False

        required_chord = float(
            2.0
            * TURN_RADIUS_M
            * math.sin(abs(float(relative_angle)))
        )
        kinematic_margin = float(distance - required_chord)
        if kinematic_margin < KINEMATIC_MARGIN_M:
            return False

        arc_safe, arc_margin, checked = self._arc_corridor_safe(
            distance=distance,
            relative_angle=relative_angle,
            sensor_xy=sensor_xy,
            lidar_m=lidar_m,
            yaw=yaw,
        )
        if not arc_safe:
            return False

        if abs(relative_angle) < math.radians(0.25):
            effective_radius = float("inf")
        else:
            effective_radius = float(
                distance
                / (2.0 * math.sin(abs(relative_angle)))
            )

        self.last_goal_sector_clearance = float(clearance)
        self.last_goal_sensor_distance = float(sensor_distance)
        self.last_goal_relative_angle = float(relative_angle)
        self.last_effective_radius_m = effective_radius
        self.last_kinematic_margin_m = kinematic_margin
        self.last_arc_corridor_margin_m = float(arc_margin)
        self.last_arc_checked_points = int(checked)
        return True

    def _sample_stage1a_goal(self) -> np.ndarray:
        # During inherited bootstrap reset, dynamic dispatch would otherwise
        # call this R0 sampler at the legacy fixed spawn. Delegate explicitly
        # to the original Stage1A sampler until random spawn is applied.
        if bool(getattr(self, "_r0_bootstrap_active", False)):
            return super()._sample_stage1a_goal()

        center_xy, sensor_xy, lidar_m, yaw = self._lidar_geometry()

        for _ in range(512):
            distance = float(
                self.np_random.uniform(
                    self.goal_distance_min,
                    self.goal_distance_max,
                )
            )
            relative_angle = float(
                self.np_random.uniform(
                    -self.goal_angle_limit,
                    self.goal_angle_limit,
                )
            )
            world_angle = float(yaw + relative_angle)
            candidate = center_xy + distance * np.asarray(
                [
                    math.cos(world_angle),
                    math.sin(world_angle),
                ],
                dtype=np.float64,
            )

            if self._candidate_reachable_and_safe(
                candidate=candidate,
                distance=distance,
                relative_angle=relative_angle,
                sensor_xy=sensor_xy,
                lidar_m=lidar_m,
                yaw=yaw,
            ):
                return candidate

        candidates: list[
            tuple[float, float, np.ndarray]
        ] = []
        for relative_angle in np.linspace(
            -self.goal_angle_limit,
            self.goal_angle_limit,
            65,
            dtype=np.float64,
        ):
            for distance in np.linspace(
                self.goal_distance_max,
                self.goal_distance_min,
                21,
                dtype=np.float64,
            ):
                world_angle = float(yaw + relative_angle)
                candidate = center_xy + float(distance) * np.asarray(
                    [
                        math.cos(world_angle),
                        math.sin(world_angle),
                    ],
                    dtype=np.float64,
                )
                if self._candidate_reachable_and_safe(
                    candidate=candidate,
                    distance=float(distance),
                    relative_angle=float(relative_angle),
                    sensor_xy=sensor_xy,
                    lidar_m=lidar_m,
                    yaw=yaw,
                ):
                    score = (
                        float(self.last_arc_corridor_margin_m)
                        + 0.25
                        * float(self.last_kinematic_margin_m)
                    )
                    candidates.append(
                        (
                            score,
                            abs(float(relative_angle)),
                            candidate,
                        )
                    )

        if not candidates:
            raise RuntimeError(
                "R0 cannot sample a car-reachable, "
                "arc-clear LiDAR goal"
            )

        candidates.sort(
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        return np.asarray(candidates[0][2], dtype=np.float64)


class MixedReachableR0Env(ReachableRandomSpawnR0Env):
    """R0 plus legacy tasks to prevent actor forgetting."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self._mix_rng = np.random.default_rng(seed + 113_000_003)
        self._mix_tasks = list(TASK_PROBABILITIES)
        self._mix_probabilities = np.asarray(
            [
                TASK_PROBABILITIES[name]
                for name in self._mix_tasks
            ],
            dtype=np.float64,
        )
        self._mix_probabilities /= self._mix_probabilities.sum()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        requested = (
            None if options is None else options.get("task_mode")
        )
        task = (
            str(requested)
            if requested is not None
            else str(
                self._mix_rng.choice(
                    self._mix_tasks,
                    p=self._mix_probabilities,
                )
            )
        )
        actual_seed = int(
            seed
            if seed is not None
            else self._mix_rng.integers(
                80_000_000,
                2_000_000_000,
            )
        )

        if task == R0_STAGE.name:
            self.success_radius = 0.30
            self.terminal_zone = 0.45
            self.terminal_min_speed = 0.018
            self.terminal_entry_speed = 0.110
            output = ReachableRandomSpawnR0Env.reset(
                self,
                seed=actual_seed,
                options={"task_mode": task},
            )
        else:
            self.success_radius = 0.20
            self.terminal_zone = 0.62
            self.terminal_min_speed = 0.022
            self.terminal_entry_speed = 0.105
            output = TerminalCurriculumEnv.reset(
                self,
                seed=actual_seed,
                options={"task_mode": task},
            )

        observation, info = output
        info = dict(info)
        info["training_mix_task"] = task
        return observation, info


def run_precheck() -> dict[str, Any]:
    env = ReachableRandomSpawnR0Env(seed=81_000_000)
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(PRECHECK_EPISODES):
            seed = 81_100_000 + episode
            _, info = env.reset(seed=seed)
            _, distance, angle = env._goal_state()

            rows.append(
                {
                    "seed": seed,
                    "spawn_clearance_m": float(
                        info["spawn_pool_clearance_m"]
                    ),
                    "goal_distance_m": float(distance),
                    "goal_angle_deg": math.degrees(float(angle)),
                    "effective_radius_m": (
                        None
                        if not math.isfinite(
                            env.last_effective_radius_m
                        )
                        else float(env.last_effective_radius_m)
                    ),
                    "kinematic_margin_m": float(
                        env.last_kinematic_margin_m
                    ),
                    "arc_corridor_margin_m": float(
                        env.last_arc_corridor_margin_m
                    ),
                    "arc_checked_points": int(
                        env.last_arc_checked_points
                    ),
                }
            )
    finally:
        env.close()

    finite_radii = [
        float(row["effective_radius_m"])
        for row in rows
        if row["effective_radius_m"] is not None
    ]
    angles = np.asarray(
        [row["goal_angle_deg"] for row in rows],
        dtype=np.float64,
    )

    report = {
        "episodes": len(rows),
        "distance_min_m": float(
            min(row["goal_distance_m"] for row in rows)
        ),
        "distance_max_m": float(
            max(row["goal_distance_m"] for row in rows)
        ),
        "angle_min_deg": float(np.min(angles)),
        "angle_max_deg": float(np.max(angles)),
        "negative_count": int(np.sum(angles < -0.5)),
        "positive_count": int(np.sum(angles > 0.5)),
        "spawn_clearance_min_m": float(
            min(row["spawn_clearance_m"] for row in rows)
        ),
        "effective_radius_min_m": (
            float(min(finite_radii))
            if finite_radii
            else None
        ),
        "kinematic_margin_min_m": float(
            min(row["kinematic_margin_m"] for row in rows)
        ),
        "arc_corridor_margin_min_m": float(
            min(
                row["arc_corridor_margin_m"]
                for row in rows
            )
        ),
        "arc_checked_points_min": int(
            min(row["arc_checked_points"] for row in rows)
        ),
        "rows": rows,
    }
    report["passed"] = bool(
        report["negative_count"] >= 100
        and report["positive_count"] >= 100
        and report["spawn_clearance_min_m"]
        >= SPAWN_CLEARANCE_MIN_M - 1e-6
        and (
            report["effective_radius_min_m"] is None
            or report["effective_radius_min_m"]
            >= TURN_RADIUS_M - 1e-6
        )
        and report["kinematic_margin_min_m"]
        >= KINEMATIC_MARGIN_M - 1e-6
        and report["arc_corridor_margin_min_m"]
        >= ARC_CORRIDOR_MARGIN_M - 1e-6
    )
    atomic_save_json(PRECHECK_PATH, report)

    print("\n========== R0 PRECHECK ==========")
    for key in (
        "episodes",
        "distance_min_m",
        "distance_max_m",
        "angle_min_deg",
        "angle_max_deg",
        "negative_count",
        "positive_count",
        "spawn_clearance_min_m",
        "effective_radius_min_m",
        "kinematic_margin_min_m",
        "arc_corridor_margin_min_m",
        "arc_checked_points_min",
        "passed",
    ):
        print(f"{key.upper()} =", report[key])
    print("PRECHECK =", PRECHECK_PATH)
    return report


def summarize_rows(
    rows: list[dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    count = len(rows)
    result = {
        "task": task,
        "episodes": count,
        "success_rate": float(
            np.mean([row["success"] for row in rows])
        ),
        "collision_rate": float(
            np.mean([row["collision"] for row in rows])
        ),
        "timeout_rate": float(
            np.mean([row["timeout"] for row in rows])
        ),
        "mean_progress_ratio": float(
            np.mean([row["progress_ratio"] for row in rows])
        ),
        "mean_speed_mps": float(
            np.mean([row["mean_speed_mps"] for row in rows])
        ),
        "negative_goal_count": int(
            np.sum(
                [
                    row["initial_angle_deg"] < -0.5
                    for row in rows
                ]
            )
        ),
        "positive_goal_count": int(
            np.sum(
                [
                    row["initial_angle_deg"] > 0.5
                    for row in rows
                ]
            )
        ),
        "collision_seeds": [
            int(row["seed"])
            for row in rows
            if row["collision"]
        ],
        "timeout_seeds": [
            int(row["seed"])
            for row in rows
            if row["timeout"]
        ],
        "results": rows,
    }

    print(f"\n========== EVAL {task} {count} ==========")
    for key in (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_progress_ratio",
        "mean_speed_mps",
        "negative_goal_count",
        "positive_goal_count",
    ):
        print(f"{key.upper()} =", result[key])
    return result


def evaluate_r0(
    model: SAC,
    episodes: int,
    seed_base: int,
) -> dict[str, Any]:
    env = ReachableRandomSpawnR0Env(seed=seed_base + 9000)
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            actual_seed = seed_base + episode
            observation, reset_info = env.reset(seed=actual_seed)
            _, initial_distance, initial_angle = env._goal_state()
            final_info: dict[str, Any] = {}
            speed_sum = 0.0
            steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                (
                    observation,
                    _,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                final_info = dict(info)
                speed_sum += float(
                    info.get("actual_speed_mps", 0.0)
                )
                steps = step + 1
                if terminated or truncated:
                    break

            _, final_distance, final_angle = env._goal_state()
            success = bool(final_info.get("reach_goal", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(
                not success
                and not collision
                and steps >= env.max_steps
            )
            progress = float(
                np.clip(
                    (
                        float(initial_distance)
                        - float(final_distance)
                    )
                    / max(float(initial_distance), 1e-6),
                    -1.0,
                    1.0,
                )
            )

            rows.append(
                {
                    "seed": actual_seed,
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "initial_distance_m": float(
                        initial_distance
                    ),
                    "initial_angle_deg": math.degrees(
                        float(initial_angle)
                    ),
                    "spawn_clearance_m": float(
                        reset_info["spawn_pool_clearance_m"]
                    ),
                    "effective_radius_m": (
                        None
                        if not math.isfinite(
                            env.last_effective_radius_m
                        )
                        else float(env.last_effective_radius_m)
                    ),
                    "progress_ratio": progress,
                    "mean_speed_mps": speed_sum / max(steps, 1),
                    "final_distance_m": float(final_distance),
                    "final_angle_deg": math.degrees(
                        float(final_angle)
                    ),
                }
            )
    finally:
        env.close()

    return summarize_rows(rows, R0_STAGE.name)


def evaluate_all(
    model: SAC,
    seed_offset: int,
) -> dict[str, Any]:
    current = evaluate_r0(
        model,
        CURRENT_EVAL_EPISODES,
        82_000_000 + seed_offset,
    )
    full_random = evaluate_random_env(
        model,
        full_holdout=True,
        episodes=FULL_HOLDOUT_EPISODES,
        seed_base=83_000_000 + seed_offset,
    )
    a3 = evaluate_legacy_task(
        model,
        task="a3",
        episodes=A3_EVAL_EPISODES,
        seed_base=84_000_000 + seed_offset,
    )
    light = evaluate_legacy_task(
        model,
        task="light",
        episodes=LIGHT_EVAL_EPISODES,
        seed_base=85_000_000 + seed_offset,
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

    historical_safety = bool(
        int(hard_caution["collision_count"]) == 0
        and int(hard_light["collision_count"]) == 0
        and not bool(focus["collision"])
    )
    legacy_retention = bool(
        a3["success_rate"] >= 0.65
        and a3["collision_rate"] <= 0.05
        and light["success_rate"] >= 0.80
        and light["collision_rate"] <= 0.05
    )
    full_random_retention = bool(
        full_random["success_rate"]
        >= FULL_RANDOM_SUCCESS_FLOOR
        and full_random["collision_rate"]
        <= FULL_RANDOM_COLLISION_CEILING
    )
    gate_passed = bool(
        current["success_rate"] >= R0_STAGE.success_gate
        and current["collision_rate"]
        <= R0_STAGE.collision_gate
        and current["timeout_rate"]
        <= R0_STAGE.timeout_gate
        and current["mean_progress_ratio"]
        >= R0_STAGE.progress_gate
        and current["mean_speed_mps"]
        >= R0_STAGE.speed_gate
        and historical_safety
        and legacy_retention
        and full_random_retention
    )

    result = {
        "stage": asdict(R0_STAGE),
        "current": current,
        "full_random_holdout": full_random,
        "a3": a3,
        "light": light,
        "hard_caution": hard_caution,
        "hard_light": hard_light,
        "focus": focus,
        "historical_safety": historical_safety,
        "legacy_retention": legacy_retention,
        "full_random_retention": full_random_retention,
        "gate_passed": gate_passed,
        "success_rate": current["success_rate"],
        "collision_rate": current["collision_rate"],
        "timeout_rate": current["timeout_rate"],
        "progress_ratio": current[
            "mean_progress_ratio"
        ],
        "speed_mps": current["mean_speed_mps"],
    }

    print("\n========== R0 SUMMARY ==========")
    print("HISTORICAL_SAFETY =", historical_safety)
    print("LEGACY_RETENTION =", legacy_retention)
    print(
        "FULL_RANDOM_RETENTION =",
        full_random_retention,
    )
    print("GATE_PASSED =", gate_passed)
    return result


def selection_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if result["historical_safety"] else 1,
        0 if result["legacy_retention"] else 1,
        0 if result["full_random_retention"] else 1,
        float(result["collision_rate"]),
        -float(result["success_rate"]),
        float(result["timeout_rate"]),
        -float(result["progress_ratio"]),
        -float(result["speed_mps"]),
    )


def configure_steering_output_only(
    model: SAC,
) -> list[Any]:
    for parameter in model.actor.parameters():
        parameter.requires_grad = False

    if int(model.actor.mu.out_features) != 2:
        raise RuntimeError(
            "Expected action order [speed, steering]"
        )

    model.actor.mu.weight.requires_grad = True
    model.actor.mu.bias.requires_grad = True

    def mask_weight(
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        masked = torch.zeros_like(gradient)
        masked[1:2, :] = gradient[1:2, :]
        return masked

    def mask_bias(
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        masked = torch.zeros_like(gradient)
        masked[1:2] = gradient[1:2]
        return masked

    handles = [
        model.actor.mu.weight.register_hook(mask_weight),
        model.actor.mu.bias.register_hook(mask_bias),
    ]
    print(
        "TRAINABLE_ACTOR_PARAMETERS =",
        ["mu.weight[steering]", "mu.bias[steering]"],
    )
    print("SPEED_OUTPUT_FROZEN = True")
    return handles


def initialize_model(
    source_path: Path,
    monitor: Monitor,
    device: str,
) -> SAC:
    source = SAC.load(str(source_path), device=device)
    model = make_fresh_model(source, monitor, device)

    print("\n========== R0 CRITIC WARMUP ==========")
    print("TIMESTEPS =", CRITIC_WARMUP_TIMESTEPS)
    print("ACTOR_FROZEN = True")
    print("TASK_PROBABILITIES =", TASK_PROBABILITIES)

    handles = freeze_actor_with_hooks(model)
    model.learn(
        total_timesteps=CRITIC_WARMUP_TIMESTEPS,
        reset_num_timesteps=True,
        log_interval=8,
        progress_bar=False,
    )
    for handle in handles:
        handle.remove()

    set_learning_rate(model, LEARNING_RATE)
    set_action_noise(model, ACTION_NOISE)
    clamp_entropy(model, ENTROPY_FLOOR)
    return model


def export_if_confirmed(
    model: SAC,
    chunk_no: int,
) -> bool:
    confirmation = evaluate_all(
        model,
        seed_offset=2_000_000 + chunk_no * 100_000,
    )
    atomic_save_json(
        OUTPUT_ROOT / "confirmation_evaluation.json",
        confirmation,
    )
    if not confirmation["gate_passed"]:
        print("INDEPENDENT_CONFIRMATION = FAILED")
        return False

    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)
    model.save(str(CONFIRMED_MODEL.with_suffix("")))
    atomic_save_json(CONFIRMED_EVAL, confirmation)
    print("INDEPENDENT_CONFIRMATION = PASSED")
    print("CONFIRMED_MODEL =", CONFIRMED_MODEL)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=(
            "precheck",
            "evaluate",
            "train-one-chunk",
        ),
        required=True,
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)

    print("STAGE = STAGE7_RANDOM_SPAWN_R0_REACHABLE")
    print("ACTION =", args.action)
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("PLANNER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("STEP_QPOS_QVEL_OVERRIDE = False")
    print("RESET_INITIAL_STATE_RANDOMIZATION = True")
    print("AUTOMATIC_NEXT_STAGE = False")
    print("TURN_RADIUS_M =", TURN_RADIUS_M)
    print(
        "R0_CONFIG =",
        {
            "spawn_clearance_min_m": SPAWN_CLEARANCE_MIN_M,
            "goal_distance_m": [
                R0_STAGE.goal_min,
                R0_STAGE.goal_max,
            ],
            "goal_angle_deg": R0_STAGE.angle_deg,
            "success_radius_m": 0.30,
            "terminal_zone_m": 0.45,
            "arc_corridor_margin_m": (
                ARC_CORRIDOR_MARGIN_M
            ),
        },
    )

    if args.action == "precheck":
        report = run_precheck()
        if not report["passed"]:
            raise RuntimeError(
                "R0 precheck failed. Do not evaluate or train."
            )
        print("R0_PRECHECK = PASS")
        return

    if not PRECHECK_PATH.is_file():
        raise RuntimeError(
            "Run R0 precheck before evaluation/training."
        )
    precheck = json.loads(
        PRECHECK_PATH.read_text(encoding="utf-8")
    )
    if not bool(precheck.get("passed", False)):
        raise RuntimeError(
            "Saved R0 precheck is not passed."
        )

    source_path = resolve_source_model()
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("SOURCE_MODEL =", source_path)
    print("DEVICE =", device)

    if args.action == "evaluate":
        model_path = (
            BEST_MODEL_PATH
            if BEST_MODEL_PATH.is_file()
            else source_path
        )
        model = SAC.load(str(model_path), device=device)
        evaluation = evaluate_all(model, seed_offset=0)
        atomic_save_json(LATEST_EVAL_PATH, evaluation)
        print("EVALUATED_MODEL =", model_path)
        return

    base_env = MixedReachableR0Env(seed=86_000_000)
    monitor = Monitor(
        base_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
    )

    state: dict[str, Any] = {}
    if STATE_PATH.is_file():
        state = json.loads(
            STATE_PATH.read_text(encoding="utf-8")
        )
    chunk_no = int(state.get("chunk_no", 0))

    if BEST_MODEL_PATH.is_file():
        model = SAC.load(
            str(BEST_MODEL_PATH),
            env=monitor,
            device=device,
        )
        if BEST_REPLAY_PATH.is_file():
            model.load_replay_buffer(
                str(BEST_REPLAY_PATH)
            )
        set_learning_rate(model, LEARNING_RATE)
        set_action_noise(model, ACTION_NOISE)
        clamp_entropy(model, ENTROPY_FLOOR)
    else:
        model = initialize_model(
            source_path,
            monitor,
            device,
        )
        baseline = evaluate_all(model, seed_offset=0)
        save_model_and_replay(
            model,
            BEST_MODEL_PATH,
            BEST_REPLAY_PATH,
        )
        atomic_save_json(BEST_EVAL_PATH, baseline)
        atomic_save_json(LATEST_EVAL_PATH, baseline)
        state = {
            "status": "INITIALIZED",
            "chunk_no": 0,
            "source_model": str(source_path),
            "best_model": str(BEST_MODEL_PATH),
            "best_replay": str(BEST_REPLAY_PATH),
            "best_metrics": {
                "success_rate": baseline["success_rate"],
                "collision_rate": baseline["collision_rate"],
                "timeout_rate": baseline["timeout_rate"],
                "progress_ratio": baseline["progress_ratio"],
                "speed_mps": baseline["speed_mps"],
                "historical_safety": baseline[
                    "historical_safety"
                ],
                "legacy_retention": baseline[
                    "legacy_retention"
                ],
                "full_random_retention": baseline[
                    "full_random_retention"
                ],
                "gate_passed": baseline["gate_passed"],
            },
        }
        atomic_save_json(STATE_PATH, state)

        if baseline["gate_passed"]:
            if export_if_confirmed(model, 0):
                state["status"] = (
                    "PASSED_WITHOUT_ACTOR_TRAINING"
                )
                state["confirmed_model"] = str(
                    CONFIRMED_MODEL
                )
                atomic_save_json(STATE_PATH, state)
                monitor.close()
                return

    chunk_no += 1
    print("\n========== TRAIN ONE R0 CHUNK ==========")
    print("CHUNK_NO =", chunk_no)
    print("TIMESTEPS =", ONE_CHUNK_TIMESTEPS)
    print("LEARNING_RATE =", LEARNING_RATE)
    print("ACTION_NOISE =", ACTION_NOISE.tolist())

    gradient_handles = configure_steering_output_only(
        model
    )
    model.learn(
        total_timesteps=ONE_CHUNK_TIMESTEPS,
        reset_num_timesteps=False,
        log_interval=8,
        progress_bar=False,
    )
    for handle in gradient_handles:
        handle.remove()
    clamp_entropy(model, ENTROPY_FLOOR)

    candidate_model = (
        OUTPUT_ROOT / f"chunk_{chunk_no:03d}_candidate.zip"
    )
    candidate_replay = (
        OUTPUT_ROOT / f"chunk_{chunk_no:03d}_replay.pkl"
    )
    save_model_and_replay(
        model,
        candidate_model,
        candidate_replay,
    )

    candidate = evaluate_all(
        model,
        seed_offset=chunk_no * 100_000,
    )
    candidate["chunk_no"] = chunk_no
    candidate_eval_path = (
        OUTPUT_ROOT / f"chunk_{chunk_no:03d}_evaluation.json"
    )
    atomic_save_json(candidate_eval_path, candidate)
    atomic_save_json(LATEST_EVAL_PATH, candidate)

    best = json.loads(
        BEST_EVAL_PATH.read_text(encoding="utf-8")
    )
    promotion_allowed = bool(
        candidate["historical_safety"]
        and candidate["legacy_retention"]
        and candidate["full_random_retention"]
    )
    promoted = bool(
        promotion_allowed
        and selection_key(candidate)
        < selection_key(best)
    )

    print("PROMOTION_ALLOWED =", promotion_allowed)
    print("PROMOTED =", promoted)

    if promoted:
        shutil.copy2(candidate_model, BEST_MODEL_PATH)
        if candidate_replay.is_file():
            shutil.copy2(
                candidate_replay,
                BEST_REPLAY_PATH,
            )
        atomic_save_json(BEST_EVAL_PATH, candidate)
        best = candidate

    passed = False
    if candidate["gate_passed"]:
        passed = export_if_confirmed(
            model,
            chunk_no,
        )

    state = {
        "status": (
            "PASSED"
            if passed
            else "WAITING_FOR_USER_DECISION"
        ),
        "chunk_no": chunk_no,
        "source_model": str(source_path),
        "best_model": str(BEST_MODEL_PATH),
        "best_replay": str(BEST_REPLAY_PATH),
        "candidate_model": str(candidate_model),
        "candidate_replay": str(candidate_replay),
        "candidate_evaluation": str(
            candidate_eval_path
        ),
        "promotion_allowed": promotion_allowed,
        "promoted": promoted,
        "latest_metrics": {
            "success_rate": candidate["success_rate"],
            "collision_rate": candidate["collision_rate"],
            "timeout_rate": candidate["timeout_rate"],
            "progress_ratio": candidate["progress_ratio"],
            "speed_mps": candidate["speed_mps"],
            "historical_safety": candidate[
                "historical_safety"
            ],
            "legacy_retention": candidate[
                "legacy_retention"
            ],
            "full_random_retention": candidate[
                "full_random_retention"
            ],
            "gate_passed": candidate["gate_passed"],
        },
        "best_metrics": {
            "success_rate": best["success_rate"],
            "collision_rate": best["collision_rate"],
            "timeout_rate": best["timeout_rate"],
            "progress_ratio": best["progress_ratio"],
            "speed_mps": best["speed_mps"],
            "historical_safety": best[
                "historical_safety"
            ],
            "legacy_retention": best[
                "legacy_retention"
            ],
            "full_random_retention": best[
                "full_random_retention"
            ],
            "gate_passed": best["gate_passed"],
        },
        "confirmed_model": (
            str(CONFIRMED_MODEL)
            if passed
            else None
        ),
    }
    atomic_save_json(STATE_PATH, state)

    print("\nSTAGE7_R0_ONE_CHUNK = FINISHED")
    print("STATUS =", state["status"])
    print("STATE =", STATE_PATH)
    monitor.close()


if __name__ == "__main__":
    main()
