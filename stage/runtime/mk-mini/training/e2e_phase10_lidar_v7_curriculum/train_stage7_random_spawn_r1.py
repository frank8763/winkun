from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from torch import nn
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


OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_random_spawn_r1"
)
STATE_PATH = OUTPUT_ROOT / "state.json"
BEST_MODEL_PATH = OUTPUT_ROOT / "best_model.zip"
BEST_REPLAY_PATH = OUTPUT_ROOT / "best_replay.pkl"
BEST_EVAL_PATH = OUTPUT_ROOT / "best_evaluation.json"
LATEST_EVAL_PATH = OUTPUT_ROOT / "latest_evaluation.json"

POOL_PATH = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_random_spawn_audit/safe_spawn_pool.json"
)
CONFIRMED_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/confirmed"
)
CONFIRMED_MODEL = (
    CONFIRMED_ROOT
    / "stage7_R1_random_spawn_0p80_0p95_angle15_passed_model.zip"
)
CONFIRMED_EVAL = (
    CONFIRMED_ROOT
    / "stage7_R1_random_spawn_0p80_0p95_angle15_evaluation.json"
)

SOURCE_CANDIDATES = (
    CONFIRMED_ROOT / "stage7_F2_0p95_1p05_angle10_passed_model.zip",
    CONFIRMED_ROOT / "stage5_B0_0p75_1p05_terminal_passed_model.zip",
)

R1_STAGE = DistanceStage(
    name="R1_RANDOM_SPAWN_0p80_0p95_A15_C070",
    goal_min=0.80,
    goal_max=0.95,
    angle_deg=15.0,
    max_steps=1450,
    clearance_margin=0.40,
    stage_budget_minutes=0,
    success_gate=0.70,
    collision_gate=0.06,
    progress_gate=0.68,
    speed_gate=0.030,
    timeout_gate=0.30,
)

TRAINING_SPAWN_CLEARANCE_M = 0.70
FULL_HOLDOUT_SPAWN_CLEARANCE_M = 0.55

CRITIC_WARMUP_TIMESTEPS = 12_000
ONE_CHUNK_TIMESTEPS = 20_000
LEARNING_RATE = 4.0e-6
ACTION_NOISE = np.asarray([0.0015, 0.0030], dtype=np.float32)
ENTROPY_FLOOR = 0.0040

CURRENT_EVAL_EPISODES = 48
FULL_HOLDOUT_EPISODES = 40
A3_EVAL_EPISODES = 16
LIGHT_EVAL_EPISODES = 20

FULL_HOLDOUT_SUCCESS_FLOOR = 0.18
FULL_HOLDOUT_COLLISION_CEILING = 0.12


def resolve_source_model() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No source model found:\n"
        + "\n".join(str(path) for path in SOURCE_CANDIDATES)
    )


def load_spawn_pool(minimum_clearance: float) -> list[dict[str, Any]]:
    if not POOL_PATH.is_file():
        raise FileNotFoundError(POOL_PATH)

    payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    spawns = [
        dict(item)
        for item in payload["spawns"]
        if float(item["spawn_min_lidar_m"]) >= float(minimum_clearance)
    ]
    if len(spawns) < 80:
        raise RuntimeError(
            f"Filtered spawn pool too small: {len(spawns)} "
            f"for clearance >= {minimum_clearance}"
        )
    return spawns


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        [
            math.cos(0.5 * float(yaw)),
            0.0,
            0.0,
            math.sin(0.5 * float(yaw)),
        ],
        dtype=np.float64,
    )


class RandomSpawnR1Env(TerminalCurriculumEnv):
    """Random reset pose + balanced safe goal.

    qpos/qvel are assigned only during reset. step() remains the inherited
    actuator-only MuJoCo dynamics path.
    """

    def __init__(
        self,
        *,
        seed: int,
        minimum_spawn_clearance: float,
        goal_min: float,
        goal_max: float,
        goal_angle_deg: float,
        goal_clearance_margin: float,
        max_steps: int,
    ) -> None:
        stage = DistanceStage(
            name=R1_STAGE.name,
            goal_min=float(goal_min),
            goal_max=float(goal_max),
            angle_deg=float(goal_angle_deg),
            max_steps=int(max_steps),
            clearance_margin=float(goal_clearance_margin),
            stage_budget_minutes=0,
            success_gate=R1_STAGE.success_gate,
            collision_gate=R1_STAGE.collision_gate,
            progress_gate=R1_STAGE.progress_gate,
            speed_gate=R1_STAGE.speed_gate,
            timeout_gate=R1_STAGE.timeout_gate,
        )
        super().__init__(
            stage=stage,
            previous_stage=None,
            fixed_task=stage.name,
            seed=seed,
        )

        self.random_spawn_pool = load_spawn_pool(minimum_spawn_clearance)
        self.minimum_spawn_clearance = float(minimum_spawn_clearance)
        self.random_goal_min = float(goal_min)
        self.random_goal_max = float(goal_max)
        self.random_goal_angle_deg = float(goal_angle_deg)
        self.random_goal_clearance_margin = float(goal_clearance_margin)

        # Baseline failure is mainly timeout/stall, not collision.
        self.safe_cruise_scale = 0.010
        self.idle_scale = 0.018
        self.risk_scale = 0.15
        self.speed_risk_scale = 0.18
        self._hard_seed_probability = 0.0

    def _apply_spawn(self, spawn: dict[str, Any]) -> None:
        qpos = np.asarray(self.initial_qpos, dtype=np.float64).copy()
        qvel = np.asarray(self.initial_qvel, dtype=np.float64).copy()
        ctrl = np.asarray(self.initial_ctrl, dtype=np.float64).copy()

        qpos[0] = float(spawn["spawn_x"])
        qpos[1] = float(spawn["spawn_y"])
        qpos[2] = float(spawn["spawn_z"])
        qpos[3:7] = yaw_to_quaternion(float(spawn["spawn_yaw_rad"]))

        qvel[:] = 0.0
        ctrl[:] = self.initial_ctrl

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def _true_goal_state(self) -> tuple[float, float]:
        result = self._goal_state()
        return float(result[-2]), float(result[-1])

    def _rebuild_reset_state(
        self,
        actual_seed: int,
        spawn: dict[str, Any],
        desired_sign: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.goal_distance_min = self.random_goal_min
        self.goal_distance_max = self.random_goal_max
        self.goal_angle_limit = math.radians(self.random_goal_angle_deg)
        self.goal_clearance_margin = self.random_goal_clearance_margin
        self.max_steps = int(self.stage.max_steps)

        accepted_goal: np.ndarray | None = None
        accepted_distance = float("nan")
        accepted_angle = float("nan")

        for _ in range(40):
            candidate = np.asarray(
                self._sample_stage1a_goal(),
                dtype=np.float64,
            )
            self.goal_xy = candidate
            distance, angle = self._true_goal_state()

            sign_matches = bool(
                (desired_sign < 0 and angle < math.radians(-0.5))
                or (desired_sign > 0 and angle > math.radians(0.5))
            )
            if sign_matches:
                accepted_goal = candidate
                accepted_distance = distance
                accepted_angle = angle
                break

        if accepted_goal is None:
            raise RuntimeError("spawn cannot provide the requested goal sign")

        self.goal_xy = accepted_goal
        self.prev_action[:] = 0.0
        self.history.clear()

        if hasattr(self, "_previous_action"):
            self._previous_action.fill(0.0)
        if hasattr(self, "_previous_delta"):
            self._previous_delta.fill(0.0)

        self.prev_goal_dist = accepted_distance
        self._initial_goal_distance = accepted_distance
        self._previous_goal_distance = accepted_distance
        self._previous_goal_angle = accepted_angle
        self._previous_xy = np.asarray(
            self.data.qpos[:2],
            dtype=np.float64,
        ).copy()
        self._previous_stage4_action.fill(0.0)
        self._idle_steps = 0

        _, _, lidar_m, _ = self._lidar_geometry()
        minimum_lidar = float(np.min(np.asarray(lidar_m, dtype=np.float64)))
        self._last_front_clearance = minimum_lidar

        observation = np.asarray(
            self._observation(update_history=False),
            dtype=np.float32,
        )
        frame = self._history_frame(
            observation[:48],
            accepted_distance,
            accepted_angle,
        )
        for _ in range(self.HISTORY_FRAMES):
            self.history.append(frame.copy())
        observation = np.asarray(self._observation(False), dtype=np.float32)

        x, y, yaw, *_ = self._pose()
        self._actual_seed = int(actual_seed)
        self._sampling_mode = "random_spawn_balanced_goal"
        self._active_task = self.stage.name
        self.curriculum_stage = self.stage.name

        info = {
            "task_mode": self.stage.name,
            "sampling_mode": self._sampling_mode,
            "training_seed": int(actual_seed),
            "spawn_x": float(x),
            "spawn_y": float(y),
            "spawn_yaw_deg": math.degrees(float(yaw)),
            "spawn_min_lidar_m": minimum_lidar,
            "spawn_pool_clearance_m": float(spawn["spawn_min_lidar_m"]),
            "goal_xy": self.goal_xy.astype(float).tolist(),
            "goal_dist": accepted_distance,
            "goal_angle": accepted_angle,
            "goal_angle_deg": math.degrees(accepted_angle),
            "desired_goal_sign": int(desired_sign),
            "actual_speed_mps": 0.0,
            "progress_ratio": 0.0,
            "planner_used": False,
            "teacher_used": False,
            "camera_used": False,
            "safety_shield_used": False,
            "manual_kinematics": False,
            "step_qpos_qvel_override": False,
            "reset_initial_state_randomization": True,
        }
        return observation, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        # Parent reset initializes Gymnasium RNG and all reward bookkeeping.
        super().reset(seed=seed, options={"task_mode": self.stage.name})

        actual_seed = int(getattr(self, "_actual_seed", seed or 0))
        local_rng = np.random.default_rng(actual_seed + 71_000_003)
        desired_sign = -1 if actual_seed % 2 == 0 else 1

        indices = local_rng.permutation(len(self.random_spawn_pool))
        last_error = "none"

        for raw_index in indices[:64]:
            spawn = self.random_spawn_pool[int(raw_index)]
            self._apply_spawn(spawn)

            try:
                observation, info = self._rebuild_reset_state(
                    actual_seed=actual_seed,
                    spawn=spawn,
                    desired_sign=desired_sign,
                )
                info["spawn_pool_index"] = int(raw_index)
                return observation, info
            except RuntimeError as exc:
                last_error = str(exc)

        raise RuntimeError(
            "RandomSpawnR1Env could not sample a balanced safe episode: "
            + last_error
        )


def configure_actor(model: SAC) -> list[str]:
    # Keep early LiDAR representation fixed. Adapt the last latent layer and
    # action head to random spawn/heading conditions.
    for parameter in model.actor.parameters():
        parameter.requires_grad = False

    model.actor.mu.weight.requires_grad = True
    model.actor.mu.bias.requires_grad = True

    linear_layers = [
        module
        for module in model.actor.latent_pi.modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("actor.latent_pi has no Linear layer")

    for parameter in linear_layers[-1].parameters():
        parameter.requires_grad = True

    return [
        name
        for name, parameter in model.actor.named_parameters()
        if parameter.requires_grad
    ]


def make_training_env(seed: int) -> RandomSpawnR1Env:
    return RandomSpawnR1Env(
        seed=seed,
        minimum_spawn_clearance=TRAINING_SPAWN_CLEARANCE_M,
        goal_min=R1_STAGE.goal_min,
        goal_max=R1_STAGE.goal_max,
        goal_angle_deg=R1_STAGE.angle_deg,
        goal_clearance_margin=R1_STAGE.clearance_margin,
        max_steps=R1_STAGE.max_steps,
    )


def make_full_holdout_env(seed: int) -> RandomSpawnR1Env:
    return RandomSpawnR1Env(
        seed=seed,
        minimum_spawn_clearance=FULL_HOLDOUT_SPAWN_CLEARANCE_M,
        goal_min=0.80,
        goal_max=1.20,
        goal_angle_deg=35.0,
        goal_clearance_margin=0.40,
        max_steps=1600,
    )


def evaluate_random_env(
    model: SAC,
    *,
    full_holdout: bool,
    episodes: int,
    seed_base: int,
) -> dict[str, Any]:
    env = (
        make_full_holdout_env(seed_base + 9000)
        if full_holdout
        else make_training_env(seed_base + 9000)
    )
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            actual_seed = seed_base + episode
            observation, reset_info = env.reset(seed=actual_seed)
            initial_distance = float(reset_info["goal_dist"])
            initial_angle = float(reset_info["goal_angle"])
            final_info: dict[str, Any] = {}
            speed_sum = 0.0
            steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                speed_sum += float(info.get("actual_speed_mps", 0.0))
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
                    (initial_distance - float(final_distance))
                    / max(initial_distance, 1e-6),
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
                    "initial_distance_m": initial_distance,
                    "initial_angle_deg": math.degrees(initial_angle),
                    "spawn_clearance_m": float(
                        reset_info["spawn_pool_clearance_m"]
                    ),
                    "progress_ratio": progress,
                    "mean_speed_mps": speed_sum / max(steps, 1),
                    "final_distance_m": float(final_distance),
                    "final_angle_deg": math.degrees(float(final_angle)),
                }
            )
    finally:
        env.close()

    return summarize_rows(
        rows,
        task="full_random_holdout" if full_holdout else R1_STAGE.name,
    )


def evaluate_legacy_task(
    model: SAC,
    task: str,
    episodes: int,
    seed_base: int,
) -> dict[str, Any]:
    env = TerminalCurriculumEnv(
        stage=R1_STAGE,
        previous_stage=None,
        fixed_task=task,
        seed=seed_base + 9000,
    )
    rows: list[dict[str, Any]] = []

    try:
        for episode in range(episodes):
            actual_seed = seed_base + episode
            observation, _ = env.reset(seed=actual_seed)
            _, initial_distance, initial_angle = env._goal_state()
            final_info: dict[str, Any] = {}
            speed_sum = 0.0
            steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                speed_sum += float(info.get("actual_speed_mps", 0.0))
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
                    (float(initial_distance) - float(final_distance))
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
                    "initial_distance_m": float(initial_distance),
                    "initial_angle_deg": math.degrees(float(initial_angle)),
                    "progress_ratio": progress,
                    "mean_speed_mps": speed_sum / max(steps, 1),
                    "final_distance_m": float(final_distance),
                    "final_angle_deg": math.degrees(float(final_angle)),
                }
            )
    finally:
        env.close()

    return summarize_rows(rows, task=task)


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    task: str,
) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise RuntimeError(f"No evaluation rows for {task}")

    result = {
        "task": task,
        "episodes": count,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(
            np.mean([row["collision"] for row in rows])
        ),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "mean_progress_ratio": float(
            np.mean([row["progress_ratio"] for row in rows])
        ),
        "mean_speed_mps": float(
            np.mean([row["mean_speed_mps"] for row in rows])
        ),
        "negative_goal_count": int(
            np.sum([row["initial_angle_deg"] < -0.5 for row in rows])
        ),
        "positive_goal_count": int(
            np.sum([row["initial_angle_deg"] > 0.5 for row in rows])
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


def evaluate_all(
    model: SAC,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    current = evaluate_random_env(
        model,
        full_holdout=False,
        episodes=CURRENT_EVAL_EPISODES,
        seed_base=51_000_000 + seed_offset,
    )
    full_holdout = evaluate_random_env(
        model,
        full_holdout=True,
        episodes=FULL_HOLDOUT_EPISODES,
        seed_base=52_000_000 + seed_offset,
    )
    a3 = evaluate_legacy_task(
        model,
        task="a3",
        episodes=A3_EVAL_EPISODES,
        seed_base=53_000_000 + seed_offset,
    )
    light = evaluate_legacy_task(
        model,
        task="light",
        episodes=LIGHT_EVAL_EPISODES,
        seed_base=54_000_000 + seed_offset,
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
        a3["collision_rate"] <= 0.05
        and light["collision_rate"] <= 0.05
        and a3["success_rate"] >= 0.55
        and light["success_rate"] >= 0.55
    )
    full_random_retention = bool(
        full_holdout["success_rate"] >= FULL_HOLDOUT_SUCCESS_FLOOR
        and full_holdout["collision_rate"]
        <= FULL_HOLDOUT_COLLISION_CEILING
    )
    gate_passed = bool(
        current["success_rate"] >= R1_STAGE.success_gate
        and current["collision_rate"] <= R1_STAGE.collision_gate
        and current["timeout_rate"] <= R1_STAGE.timeout_gate
        and current["mean_progress_ratio"] >= R1_STAGE.progress_gate
        and current["mean_speed_mps"] >= R1_STAGE.speed_gate
        and historical_safety
        and legacy_retention
        and full_random_retention
    )

    result = {
        "stage": R1_STAGE.__dict__,
        "current": current,
        "full_random_holdout": full_holdout,
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
        "progress_ratio": current["mean_progress_ratio"],
        "speed_mps": current["mean_speed_mps"],
    }

    print("\n========== R1 SUMMARY ==========")
    print("HISTORICAL_SAFETY =", historical_safety)
    print("LEGACY_RETENTION =", legacy_retention)
    print("FULL_RANDOM_RETENTION =", full_random_retention)
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


def save_model_and_replay(
    model: SAC,
    model_path: Path,
    replay_path: Path,
) -> None:
    model.save(str(model_path.with_suffix("")))
    if model.replay_buffer is not None:
        model.save_replay_buffer(str(replay_path))


def load_best_model(
    env: Monitor,
    device: str,
) -> SAC:
    model = SAC.load(str(BEST_MODEL_PATH), env=env, device=device)
    if BEST_REPLAY_PATH.is_file():
        model.load_replay_buffer(str(BEST_REPLAY_PATH))
    print("TRAINABLE_ACTOR_PARAMETERS =", configure_actor(model))
    set_learning_rate(model, LEARNING_RATE)
    set_action_noise(model, ACTION_NOISE)
    clamp_entropy(model, ENTROPY_FLOOR)
    return model


def initialize_model(
    source_path: Path,
    env: Monitor,
    device: str,
) -> SAC:
    source = SAC.load(str(source_path), device=device)
    model = make_fresh_model(source, env, device)

    print("\n========== INITIAL CRITIC WARMUP ==========")
    print("TIMESTEPS =", CRITIC_WARMUP_TIMESTEPS)
    print("ACTOR_FROZEN = True")

    handles = freeze_actor_with_hooks(model)
    model.learn(
        total_timesteps=CRITIC_WARMUP_TIMESTEPS,
        reset_num_timesteps=True,
        log_interval=8,
        progress_bar=False,
    )
    for handle in handles:
        handle.remove()

    print("TRAINABLE_ACTOR_PARAMETERS =", configure_actor(model))
    set_learning_rate(model, LEARNING_RATE)
    set_action_noise(model, ACTION_NOISE)
    clamp_entropy(model, ENTROPY_FLOOR)
    return model


def confirm_and_export(
    model: SAC,
    *,
    chunk_no: int,
) -> bool:
    confirmation = evaluate_all(
        model,
        seed_offset=1_000_000 + chunk_no * 100_000,
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
        choices=("train-one-chunk", "evaluate"),
        default="train-one-chunk",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIRMED_ROOT.mkdir(parents=True, exist_ok=True)

    source_path = resolve_source_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("STAGE = STAGE7_RANDOM_SPAWN_R1")
    print("ACTION =", args.action)
    print("SOURCE_MODEL =", source_path)
    print("DEVICE =", device)
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("PLANNER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("STEP_QPOS_QVEL_OVERRIDE = False")
    print("RESET_INITIAL_STATE_RANDOMIZATION = True")
    print("AUTOMATIC_NEXT_STAGE = False")
    print(
        "R1_CONFIG =",
        {
            "spawn_clearance_min_m": TRAINING_SPAWN_CLEARANCE_M,
            "goal_distance_m": [R1_STAGE.goal_min, R1_STAGE.goal_max],
            "goal_angle_deg": R1_STAGE.angle_deg,
            "goal_clearance_margin_m": R1_STAGE.clearance_margin,
            "max_steps": R1_STAGE.max_steps,
        },
    )

    base_env = make_training_env(seed=61_000_000)
    monitor = Monitor(
        base_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
    )

    state: dict[str, Any] = {}
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    chunk_no = int(state.get("chunk_no", 0))

    if args.action == "evaluate":
        model_path = (
            BEST_MODEL_PATH if BEST_MODEL_PATH.is_file() else source_path
        )
        model = SAC.load(str(model_path), env=monitor, device=device)
        evaluation = evaluate_all(
            model,
            seed_offset=chunk_no * 100_000,
        )
        atomic_save_json(LATEST_EVAL_PATH, evaluation)
        print("EVALUATED_MODEL =", model_path)
        monitor.close()
        return

    if not BEST_MODEL_PATH.is_file():
        model = initialize_model(source_path, monitor, device)
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
                "historical_safety": baseline["historical_safety"],
                "legacy_retention": baseline["legacy_retention"],
                "full_random_retention": baseline[
                    "full_random_retention"
                ],
                "gate_passed": baseline["gate_passed"],
            },
        }
        atomic_save_json(STATE_PATH, state)

        if baseline["gate_passed"] and confirm_and_export(
            model,
            chunk_no=0,
        ):
            state["status"] = "PASSED_WITHOUT_ACTOR_TRAINING"
            state["confirmed_model"] = str(CONFIRMED_MODEL)
            atomic_save_json(STATE_PATH, state)
            monitor.close()
            return
    else:
        model = load_best_model(monitor, device)

    chunk_no += 1
    print("\n========== TRAIN ONE R1 CHUNK ==========")
    print("CHUNK_NO =", chunk_no)
    print("TIMESTEPS =", ONE_CHUNK_TIMESTEPS)
    print("LEARNING_RATE =", LEARNING_RATE)
    print("ACTION_NOISE =", ACTION_NOISE.tolist())
    print("ENTROPY_FLOOR =", ENTROPY_FLOOR)

    model.learn(
        total_timesteps=ONE_CHUNK_TIMESTEPS,
        reset_num_timesteps=False,
        log_interval=8,
        progress_bar=False,
    )
    clamp_entropy(model, ENTROPY_FLOOR)

    candidate_model = OUTPUT_ROOT / f"chunk_{chunk_no:03d}_candidate.zip"
    candidate_replay = OUTPUT_ROOT / f"chunk_{chunk_no:03d}_replay.pkl"
    save_model_and_replay(model, candidate_model, candidate_replay)

    candidate = evaluate_all(
        model,
        seed_offset=chunk_no * 100_000,
    )
    candidate["chunk_no"] = chunk_no
    candidate_eval = (
        OUTPUT_ROOT / f"chunk_{chunk_no:03d}_evaluation.json"
    )
    atomic_save_json(candidate_eval, candidate)
    atomic_save_json(LATEST_EVAL_PATH, candidate)

    best = json.loads(BEST_EVAL_PATH.read_text(encoding="utf-8"))
    promotion_allowed = bool(
        candidate["historical_safety"]
        and candidate["legacy_retention"]
        and candidate["full_random_retention"]
    )
    promoted = bool(
        promotion_allowed
        and selection_key(candidate) < selection_key(best)
    )

    print("PROMOTION_ALLOWED =", promotion_allowed)
    print("PROMOTED =", promoted)

    if promoted:
        shutil.copy2(candidate_model, BEST_MODEL_PATH)
        if candidate_replay.is_file():
            shutil.copy2(candidate_replay, BEST_REPLAY_PATH)
        atomic_save_json(BEST_EVAL_PATH, candidate)
        best = candidate

    passed = False
    if candidate["gate_passed"]:
        passed = confirm_and_export(model, chunk_no=chunk_no)

    state = {
        "status": "PASSED" if passed else "WAITING_FOR_USER_DECISION",
        "chunk_no": chunk_no,
        "source_model": str(source_path),
        "best_model": str(BEST_MODEL_PATH),
        "best_replay": str(BEST_REPLAY_PATH),
        "candidate_model": str(candidate_model),
        "candidate_replay": str(candidate_replay),
        "candidate_evaluation": str(candidate_eval),
        "promotion_allowed": promotion_allowed,
        "promoted": promoted,
        "latest_metrics": {
            "success_rate": candidate["success_rate"],
            "collision_rate": candidate["collision_rate"],
            "timeout_rate": candidate["timeout_rate"],
            "progress_ratio": candidate["progress_ratio"],
            "speed_mps": candidate["speed_mps"],
            "historical_safety": candidate["historical_safety"],
            "legacy_retention": candidate["legacy_retention"],
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
            "historical_safety": best["historical_safety"],
            "legacy_retention": best["legacy_retention"],
            "full_random_retention": best[
                "full_random_retention"
            ],
            "gate_passed": best["gate_passed"],
        },
        "confirmed_model": str(CONFIRMED_MODEL) if passed else None,
    }
    atomic_save_json(STATE_PATH, state)

    print("\nSTAGE7_R1_ONE_CHUNK = FINISHED")
    print("STATUS =", state["status"])
    print("STATE =", STATE_PATH)
    monitor.close()


if __name__ == "__main__":
    main()
