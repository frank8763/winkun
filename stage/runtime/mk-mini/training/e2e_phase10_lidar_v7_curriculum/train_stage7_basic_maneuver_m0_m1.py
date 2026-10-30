from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from training.e2e_phase10_lidar_v7_curriculum.train_stage7_random_spawn_r0_reachable import (
    ReachableRandomSpawnR0Env,
)
from training.e2e_phase10_lidar_v7_curriculum.test_stage7_full_random_v2_reachable_fix import (
    FullRandomV2ReachableEnv,
)

STAGE = "STAGE7_BASIC_MANEUVER_M0_M1"

SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_r1_reachable_transition/best_model.zip"
)

OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_basic_maneuver_m0_m1"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BASELINE_REPORT = OUTPUT_ROOT / "baseline.json"
CANDIDATE_REPORT = OUTPUT_ROOT / "candidate.json"
STATE_PATH = OUTPUT_ROOT / "state.json"

CANDIDATE_MODEL = OUTPUT_ROOT / "candidate_model.zip"
CANDIDATE_REPLAY = OUTPUT_ROOT / "candidate_replay.pkl"
BEST_MODEL = OUTPUT_ROOT / "best_model.zip"
BEST_REPLAY = OUTPUT_ROOT / "best_replay.pkl"
BEST_REPORT = OUTPUT_ROOT / "best_evaluation.json"

# ---------------------------------------------------------------------------
# Foundation curriculum
# ---------------------------------------------------------------------------

SKILL_CONFIG = {
    "straight": {
        "probability": 0.40,
        "distance_min": 1.20,
        "distance_max": 1.55,
        "angle_min_deg": 0.60,
        "angle_max_deg": 2.00,
        "max_steps": 1500,
    },
    "left": {
        "probability": 0.30,
        "distance_min": 1.55,
        "distance_max": 1.90,
        "angle_min_deg": 4.0,
        "angle_max_deg": 8.0,
        "max_steps": 1600,
    },
    "right": {
        "probability": 0.30,
        "distance_min": 1.55,
        "distance_max": 1.90,
        "angle_min_deg": 4.0,
        "angle_max_deg": 8.0,
        "max_steps": 1600,
    },
}

TRAIN_STEPS = 15_000
LEARNING_RATE = 1.2e-7
ACTION_NOISE = np.asarray([0.0010, 0.0016], dtype=np.float32)

SPEED_GRAD_SCALE = 0.45
STEER_GRAD_SCALE = 0.35
ENTROPY_FLOOR = 0.003

SKILL_EVAL_EPISODES = 60
R0_RETENTION_EPISODES = 64
FULL_RETENTION_EPISODES = 80

SKILL_SEED_BASE = {
    "straight": 270_000_000,
    "left": 271_000_000,
    "right": 272_000_000,
}
R0_SEED_BASE = 273_000_000
FULL_SEED_BASE = 274_000_000

# Candidate promotion = useful skill improvement + no safety/retention break.
MAX_SKILL_COLLISION = 0.05
MAX_R0_COLLISION = 0.0625
MAX_FULL_COLLISION = 0.05
MAX_SKILL_REGRESSION = 0.03
MAX_R0_SUCCESS_REGRESSION = 0.05
MAX_FULL_SUCCESS_REGRESSION = 0.05

MIN_AVG_SKILL_SUCCESS_GAIN = 0.05
MIN_AVG_TIMEOUT_REDUCTION = 0.05
MIN_AVG_PROGRESS_GAIN = 0.02

# Foundation completion target. One chunk does not have to reach this.
FOUNDATION_STRAIGHT_SUCCESS = 0.80
FOUNDATION_LEFT_SUCCESS = 0.70
FOUNDATION_RIGHT_SUCCESS = 0.70


def save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def force_constant_lr(model: SAC, lr: float) -> None:
    lr = float(lr)
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr

    for optimizer in (
        getattr(model.actor, "optimizer", None),
        getattr(model.critic, "optimizer", None),
        getattr(model, "ent_coef_optimizer", None),
    ):
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            group["lr"] = lr

    print("LR_SCHEDULE_FORCED_CONSTANT = True")
    print("LR_SCHEDULE_TEST =", float(model.lr_schedule(0.5)))


def clamp_entropy(model: SAC) -> None:
    log_ent = getattr(model, "log_ent_coef", None)
    if log_ent is None:
        return

    with torch.no_grad():
        log_ent.data.clamp_(min=math.log(ENTROPY_FLOOR))


def configure_actor_output_only(model: SAC):
    for p in model.actor.parameters():
        p.requires_grad_(False)

    mu = model.actor.mu
    if int(mu.out_features) != 2:
        raise RuntimeError(
            f"Expected two actions (speed + steering), got {mu.out_features}"
        )

    mu.weight.requires_grad_(True)
    mu.bias.requires_grad_(True)

    def weight_hook(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[0, :] *= SPEED_GRAD_SCALE
        out[1, :] *= STEER_GRAD_SCALE
        return out

    def bias_hook(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[0] *= SPEED_GRAD_SCALE
        out[1] *= STEER_GRAD_SCALE
        return out

    handles = [
        mu.weight.register_hook(weight_hook),
        mu.bias.register_hook(bias_hook),
    ]

    print("TRAINABLE_ACTOR = speed_output + steering_output")
    print("ACTOR_HIDDEN_FROZEN = True")
    print("LIDAR_REPRESENTATION_FROZEN = True")
    print("SPEED_GRAD_SCALE =", SPEED_GRAD_SCALE)
    print("STEER_GRAD_SCALE =", STEER_GRAD_SCALE)

    return handles


class BasicManeuverEnv(ReachableRandomSpawnR0Env):
    """
    M0/M1 foundation environment.

    It deliberately trains local motion primitives rather than long-distance
    navigation:
      straight : local target within +/-2 deg
      left     : local target +4..+8 deg
      right    : local target -4..-8 deg

    Robot x/y and yaw still come from the validated random-spawn pool.
    Every local target still passes the 4.2 m turn-radius / LiDAR arc guard.
    """

    def __init__(
        self,
        seed: int,
        fixed_skill: str | None = None,
    ) -> None:
        super().__init__(seed=seed)

        if fixed_skill is not None and fixed_skill not in SKILL_CONFIG:
            raise ValueError(f"Unknown skill: {fixed_skill}")

        self.fixed_skill = fixed_skill
        self._skill_rng = np.random.default_rng(seed + 301_000_003)
        self._current_skill = fixed_skill or "straight"
        self._current_sign = 1
        self._maneuver_prev_abs_angle = 0.0

        self.success_radius = 0.30
        self.terminal_zone = 0.45

    def _choose_skill(self) -> str:
        if self.fixed_skill is not None:
            return self.fixed_skill

        names = tuple(SKILL_CONFIG)
        probs = np.asarray(
            [SKILL_CONFIG[name]["probability"] for name in names],
            dtype=np.float64,
        )
        probs /= probs.sum()
        return str(self._skill_rng.choice(names, p=probs))

    def _prepare_seed(self, raw_seed: int, skill: str) -> int:
        # Parent RandomSpawnR1 reset enforces desired goal sign using seed parity:
        # even -> negative/right, odd -> positive/left.
        seed = int(raw_seed)

        if skill == "left" and seed % 2 == 0:
            seed += 1
        elif skill == "right" and seed % 2 != 0:
            seed += 1

        self._current_sign = -1 if seed % 2 == 0 else 1
        return seed

    def _sample_stage1a_goal(self) -> np.ndarray:
        # Preserve the already-validated bootstrap behavior.
        if bool(getattr(self, "_r0_bootstrap_active", False)):
            return super()._sample_stage1a_goal()

        cfg = SKILL_CONFIG[self._current_skill]
        center_xy, sensor_xy, lidar_m, yaw = self._lidar_geometry()

        if self._current_skill == "straight":
            # Still satisfy parent's balanced-sign check, but remain almost
            # straight. This prevents the reset loop from fighting a zero-angle
            # target.
            sign = float(self._current_sign)
            angle_low = math.radians(float(cfg["angle_min_deg"]))
            angle_high = math.radians(float(cfg["angle_max_deg"]))
        elif self._current_skill == "left":
            sign = 1.0
            angle_low = math.radians(float(cfg["angle_min_deg"]))
            angle_high = math.radians(float(cfg["angle_max_deg"]))
        else:
            sign = -1.0
            angle_low = math.radians(float(cfg["angle_min_deg"]))
            angle_high = math.radians(float(cfg["angle_max_deg"]))

        for _ in range(512):
            distance = float(
                self.np_random.uniform(
                    float(cfg["distance_min"]),
                    float(cfg["distance_max"]),
                )
            )
            relative_angle = sign * float(
                self.np_random.uniform(angle_low, angle_high)
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

        # Deterministic fallback, widest distance first for turn feasibility.
        for distance in np.linspace(
            float(cfg["distance_max"]),
            float(cfg["distance_min"]),
            24,
        ):
            for magnitude in np.linspace(
                angle_low,
                angle_high,
                17,
            ):
                relative_angle = sign * float(magnitude)
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
                    relative_angle=relative_angle,
                    sensor_xy=sensor_xy,
                    lidar_m=lidar_m,
                    yaw=yaw,
                ):
                    return candidate

        raise RuntimeError(
            f"M0/M1 cannot sample reachable {self._current_skill} target"
        )

    def reset(self, seed=None, options=None):
        raw_seed = int(
            seed
            if seed is not None
            else self._skill_rng.integers(
                400_000_000,
                2_000_000_000,
            )
        )

        self._current_skill = self._choose_skill()
        actual_seed = self._prepare_seed(raw_seed, self._current_skill)

        # Parent's _rebuild_reset_state needs broad limits, but actual angle
        # sampling is fully controlled by _sample_stage1a_goal above.
        cfg = SKILL_CONFIG[self._current_skill]
        self.random_goal_min = float(cfg["distance_min"])
        self.random_goal_max = float(cfg["distance_max"])
        self.random_goal_angle_deg = float(cfg["angle_max_deg"])

        obs, info = super().reset(seed=actual_seed, options=options)
        self.max_steps = int(cfg["max_steps"])

        current_angle = float(info["goal_angle"])
        self._maneuver_prev_abs_angle = abs(current_angle)

        info = dict(info)
        info["task_mode"] = f"BASIC_{self._current_skill.upper()}"
        info["maneuver_skill"] = self._current_skill
        info["maneuver_raw_seed"] = raw_seed
        info["maneuver_actual_seed"] = actual_seed

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        info = dict(info)
        current_angle = float(info.get("goal_angle", 0.0))
        current_abs = abs(current_angle)
        angle_improvement = (
            self._maneuver_prev_abs_angle - current_abs
        )
        progress = float(info.get("goal_progress", 0.0))

        progress_bonus = 0.010 * float(
            np.clip(progress / 0.004, -1.0, 1.0)
        )

        if self._current_skill == "straight":
            alignment = float(
                np.clip(
                    1.0 - current_abs / math.radians(3.0),
                    0.0,
                    1.0,
                )
            )
            alignment_bonus = 0.012 * alignment
            heading_improvement_bonus = 0.006 * float(
                np.clip(
                    angle_improvement / math.radians(0.15),
                    -1.0,
                    1.0,
                )
            )
        else:
            alignment_bonus = 0.006 * float(
                np.clip(
                    1.0 - current_abs / math.radians(5.0),
                    0.0,
                    1.0,
                )
            )
            heading_improvement_bonus = 0.016 * float(
                np.clip(
                    angle_improvement / math.radians(0.15),
                    -1.0,
                    1.0,
                )
            )

        maneuver_reward = (
            progress_bonus
            + alignment_bonus
            + heading_improvement_bonus
        )
        reward = float(reward + maneuver_reward)

        self._maneuver_prev_abs_angle = current_abs

        info["maneuver_skill"] = self._current_skill
        info["maneuver_progress_bonus"] = progress_bonus
        info["maneuver_alignment_bonus"] = alignment_bonus
        info["maneuver_heading_improvement_bonus"] = (
            heading_improvement_bonus
        )
        info["maneuver_reward"] = maneuver_reward

        return obs, reward, terminated, truncated, info


def evaluate(
    model: SAC,
    env,
    episodes: int,
    seed_base: int,
    name: str,
) -> dict[str, Any]:
    rows = []

    try:
        for ep in range(episodes):
            seed = seed_base + ep * 2
            obs, reset_info = env.reset(seed=seed)
            initial_dist = float(reset_info["goal_dist"])
            initial_angle = float(reset_info["goal_angle"])

            final_info = {}
            speed_sum = 0.0
            steps = 0

            for step in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                speed_sum += float(
                    info.get("actual_speed_mps", 0.0)
                )
                steps = step + 1

                if terminated or truncated:
                    break

            _, final_dist, final_angle = env._goal_state()

            success = bool(final_info.get("reach_goal", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(
                not success
                and not collision
                and steps >= env.max_steps
            )

            progress = float(
                np.clip(
                    (initial_dist - float(final_dist))
                    / max(initial_dist, 1e-6),
                    -1.0,
                    1.0,
                )
            )

            rows.append(
                {
                    "seed": seed,
                    "success": success,
                    "collision": collision,
                    "timeout": timeout,
                    "steps": steps,
                    "initial_distance_m": initial_dist,
                    "initial_angle_deg": math.degrees(initial_angle),
                    "final_distance_m": float(final_dist),
                    "final_angle_deg": math.degrees(float(final_angle)),
                    "progress_ratio": progress,
                    "speed_mps": speed_sum / max(steps, 1),
                }
            )
    finally:
        env.close()

    n = len(rows)
    report = {
        "name": name,
        "episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "timeout_rate": sum(r["timeout"] for r in rows) / n,
        "progress_ratio": float(
            np.mean([r["progress_ratio"] for r in rows])
        ),
        "speed_mps": float(
            np.mean([r["speed_mps"] for r in rows])
        ),
        "rows": rows,
    }

    print(f"\n========== {name} ==========")
    for key in (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "progress_ratio",
        "speed_mps",
    ):
        print(key.upper(), "=", report[key])

    return report


def evaluate_skills(model: SAC) -> dict[str, Any]:
    out = {}
    for skill in ("straight", "left", "right"):
        out[skill] = evaluate(
            model,
            BasicManeuverEnv(
                seed=SKILL_SEED_BASE[skill] + 9999,
                fixed_skill=skill,
            ),
            SKILL_EVAL_EPISODES,
            SKILL_SEED_BASE[skill],
            f"SKILL_{skill.upper()}_{SKILL_EVAL_EPISODES}",
        )
    return out


def evaluate_retention(model: SAC) -> dict[str, Any]:
    r0 = evaluate(
        model,
        ReachableRandomSpawnR0Env(seed=R0_SEED_BASE + 9999),
        R0_RETENTION_EPISODES,
        R0_SEED_BASE,
        f"R0_RETENTION_{R0_RETENTION_EPISODES}",
    )
    full = evaluate(
        model,
        FullRandomV2ReachableEnv(seed=FULL_SEED_BASE + 9999),
        FULL_RETENTION_EPISODES,
        FULL_SEED_BASE,
        f"FULL_V2_RETENTION_{FULL_RETENTION_EPISODES}",
    )
    return {"r0": r0, "full_v2": full}


def average_metric(block: dict[str, Any], key: str) -> float:
    return float(
        np.mean(
            [
                block["straight"][key],
                block["left"][key],
                block["right"][key],
            ]
        )
    )


def decide(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    bs = baseline["skills"]
    cs = candidate["skills"]
    br = baseline["retention"]
    cr = candidate["retention"]

    skill_deltas = {
        skill: {
            "success_delta": (
                cs[skill]["success_rate"]
                - bs[skill]["success_rate"]
            ),
            "collision_delta": (
                cs[skill]["collision_rate"]
                - bs[skill]["collision_rate"]
            ),
            "timeout_delta": (
                cs[skill]["timeout_rate"]
                - bs[skill]["timeout_rate"]
            ),
            "progress_delta": (
                cs[skill]["progress_ratio"]
                - bs[skill]["progress_ratio"]
            ),
        }
        for skill in ("straight", "left", "right")
    }

    avg_success_gain = (
        average_metric(cs, "success_rate")
        - average_metric(bs, "success_rate")
    )
    avg_timeout_reduction = (
        average_metric(bs, "timeout_rate")
        - average_metric(cs, "timeout_rate")
    )
    avg_progress_gain = (
        average_metric(cs, "progress_ratio")
        - average_metric(bs, "progress_ratio")
    )

    useful_gain = bool(
        avg_success_gain >= MIN_AVG_SKILL_SUCCESS_GAIN
        or avg_timeout_reduction >= MIN_AVG_TIMEOUT_REDUCTION
        or avg_progress_gain >= MIN_AVG_PROGRESS_GAIN
    )

    checks = {
        "straight_collision_safe": (
            cs["straight"]["collision_rate"] <= MAX_SKILL_COLLISION
        ),
        "left_collision_safe": (
            cs["left"]["collision_rate"] <= MAX_SKILL_COLLISION
        ),
        "right_collision_safe": (
            cs["right"]["collision_rate"] <= MAX_SKILL_COLLISION
        ),
        "straight_not_regressed": (
            cs["straight"]["success_rate"]
            >= bs["straight"]["success_rate"] - MAX_SKILL_REGRESSION
        ),
        "left_not_regressed": (
            cs["left"]["success_rate"]
            >= bs["left"]["success_rate"] - MAX_SKILL_REGRESSION
        ),
        "right_not_regressed": (
            cs["right"]["success_rate"]
            >= bs["right"]["success_rate"] - MAX_SKILL_REGRESSION
        ),
        "r0_retained": (
            cr["r0"]["success_rate"]
            >= br["r0"]["success_rate"] - MAX_R0_SUCCESS_REGRESSION
            and cr["r0"]["collision_rate"] <= MAX_R0_COLLISION
        ),
        "full_v2_retained": (
            cr["full_v2"]["success_rate"]
            >= br["full_v2"]["success_rate"]
            - MAX_FULL_SUCCESS_REGRESSION
            and cr["full_v2"]["collision_rate"]
            <= MAX_FULL_COLLISION
        ),
        "useful_skill_gain": useful_gain,
    }

    foundation_complete = bool(
        cs["straight"]["success_rate"] >= FOUNDATION_STRAIGHT_SUCCESS
        and cs["left"]["success_rate"] >= FOUNDATION_LEFT_SUCCESS
        and cs["right"]["success_rate"] >= FOUNDATION_RIGHT_SUCCESS
        and cs["straight"]["collision_rate"] <= MAX_SKILL_COLLISION
        and cs["left"]["collision_rate"] <= MAX_SKILL_COLLISION
        and cs["right"]["collision_rate"] <= MAX_SKILL_COLLISION
    )

    return {
        "passed": bool(all(checks.values())),
        "foundation_complete": foundation_complete,
        "checks": checks,
        "skill_deltas": skill_deltas,
        "average_success_gain": avg_success_gain,
        "average_timeout_reduction": avg_timeout_reduction,
        "average_progress_gain": avg_progress_gain,
    }


def main() -> None:
    print("STAGE =", STAGE)
    print("TRAINING = True")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("PLANNER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("STEP_QPOS_QVEL_OVERRIDE = False")
    print("RESET_RANDOM_XY_YAW = True")
    print("SOURCE_MODEL =", SOURCE_MODEL)
    print("SOURCE_IS_70_30_PROMOTED_BEST = True")
    print("FAILED_LAST_HIDDEN_MODEL_USED = False")
    print("SKILL_CONFIG =", SKILL_CONFIG)
    print("TRAIN_STEPS =", TRAIN_STEPS)
    print("LEARNING_RATE =", LEARNING_RATE)
    print("ACTION_NOISE =", ACTION_NOISE.tolist())
    print("FRESH_REPLAY = True")
    print("SIGNED_REVERSE_TRAINING_THIS_STAGE = False")
    print("NEXT_AFTER_M0_M1 = M2_turn_then_straight")
    print("REVERSE_STAGE_AFTER_M2 = M3_signed_reverse_recovery")
    print("AUTOMATIC_NEXT_STAGE = False")

    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", device)

    print("\n========== BASELINE SKILLS ==========")
    baseline_model = SAC.load(str(SOURCE_MODEL), device=device)

    baseline = {
        "skills": evaluate_skills(baseline_model),
        "retention": evaluate_retention(baseline_model),
    }
    save_json(BASELINE_REPORT, baseline)

    print("\n========== TRAIN BASIC MANEUVERS ==========")
    train_env = BasicManeuverEnv(seed=275_000_000)
    monitor = Monitor(
        train_env,
        filename=str(OUTPUT_ROOT / "monitor.csv"),
    )

    model = SAC.load(
        str(SOURCE_MODEL),
        env=monitor,
        device=device,
    )

    print(
        "REPLAY_INITIAL_SIZE =",
        0 if model.replay_buffer is None else model.replay_buffer.size(),
    )

    force_constant_lr(model, LEARNING_RATE)
    model.action_noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=ACTION_NOISE.copy(),
    )
    clamp_entropy(model)
    handles = configure_actor_output_only(model)

    try:
        model.learn(
            total_timesteps=TRAIN_STEPS,
            reset_num_timesteps=False,
            log_interval=8,
            progress_bar=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    clamp_entropy(model)

    print(
        "ACTUAL_ACTOR_LR_AFTER_TRAIN =",
        float(model.actor.optimizer.param_groups[0]["lr"]),
    )
    print(
        "ACTUAL_CRITIC_LR_AFTER_TRAIN =",
        float(model.critic.optimizer.param_groups[0]["lr"]),
    )
    print(
        "REPLAY_FINAL_SIZE =",
        model.replay_buffer.size()
        if model.replay_buffer is not None
        else -1,
    )

    model.save(str(CANDIDATE_MODEL))
    model.save_replay_buffer(str(CANDIDATE_REPLAY))
    monitor.close()

    print("\n========== CANDIDATE SKILLS ==========")
    candidate = {
        "skills": evaluate_skills(model),
        "retention": evaluate_retention(model),
    }
    save_json(CANDIDATE_REPORT, candidate)

    decision = decide(baseline, candidate)

    print("\n========== M0/M1 DECISION ==========")
    print("DECISION =", decision)

    promoted = bool(decision["passed"])

    if promoted:
        shutil.copy2(CANDIDATE_MODEL, BEST_MODEL)
        shutil.copy2(CANDIDATE_REPLAY, BEST_REPLAY)
        save_json(BEST_REPORT, candidate)

    state = {
        "status": "PROMOTED" if promoted else "REJECTED",
        "foundation_complete": bool(decision["foundation_complete"]),
        "source_model": str(SOURCE_MODEL),
        "candidate_model": str(CANDIDATE_MODEL),
        "best_model": str(BEST_MODEL) if promoted else None,
        "decision": decision,
        "automatic_next_stage": False,
        "next_stage_if_foundation_complete": "M2_turn_then_straight",
        "reverse_stage": "M3_signed_reverse_recovery",
    }
    save_json(STATE_PATH, state)

    print("M0_M1_CANDIDATE_PROMOTED =", promoted)
    print(
        "BASIC_MANEUVER_FOUNDATION_COMPLETE =",
        decision["foundation_complete"],
    )
    print("SOURCE_70_30_BEST_PRESERVED = True")
    print("FAILED_LAST_HIDDEN_MODEL_IGNORED = True")
    print("AUTOMATIC_NEXT_STAGE = False")
    print("STATE =", STATE_PATH)
    print("DONE = True")


if __name__ == "__main__":
    main()
