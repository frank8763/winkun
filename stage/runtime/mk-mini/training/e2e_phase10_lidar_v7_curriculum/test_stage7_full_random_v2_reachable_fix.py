from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from training.e2e_phase10_lidar_v7_curriculum.train_stage7_random_spawn_r0_reachable import (
    ReachableRandomSpawnR0Env,
    TURN_RADIUS_M,
    KINEMATIC_MARGIN_M,
    ARC_CORRIDOR_MARGIN_M,
    SPAWN_CLEARANCE_MIN_M,
)

OUTPUT_ROOT = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_full_random_v2_reachable"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

AUDIT_PATH = OUTPUT_ROOT / "reset_audit_300.json"
EVAL_PATH = OUTPUT_ROOT / "joint_candidate_eval_80.json"

SOURCE_MODEL = Path(
    "experiments/e2e_phase10_lidar_v7_curriculum/"
    "stage7_r0_joint_adapt/joint_chunk001_candidate.zip"
)

AUDIT_EPISODES = 300
AUDIT_SEED_BASE = 240_000_000

# Matched to the old 80-episode Full Random evaluation seed family.
EVAL_EPISODES = 80
EVAL_SEED_BASE = 156_000_000

# Full Random V2:
# Keep broad random goals, but reject car-infeasible combinations.
GOAL_MIN_M = 1.20
GOAL_MAX_M = 5.00
GOAL_ANGLE_DEG = 35.0
MAX_STEPS = 1600
SUCCESS_RADIUS_M = 0.30

OLD_MATCHED_BASELINE = {
    "success_rate": 0.4625,
    "collision_rate": 0.1625,
    "timeout_rate": 0.375,
    "progress_ratio": 0.6571489300880279,
}


def atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


class FullRandomV2ReachableEnv(ReachableRandomSpawnR0Env):
    """
    Broad random spawn/goal environment with the already validated R0
    bootstrap + 4.2 m kinematic reachability + LiDAR arc-corridor guard.

    Randomness is preserved:
      - spawn x/y random from safe pool
      - spawn yaw random from safe pool
      - goal distance random in [1.20, 5.00] m
      - relative goal angle random in [-35, +35] deg
      - left/right balanced by seed parity

    The only rejected samples are combinations that are unsafe/unreachable.
    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)

        self.random_goal_min = GOAL_MIN_M
        self.random_goal_max = GOAL_MAX_M
        self.random_goal_angle_deg = GOAL_ANGLE_DEG
        self.random_goal_clearance_margin = 0.40

        self.success_radius = SUCCESS_RADIUS_M
        self.terminal_zone = 0.45
        self.max_steps = MAX_STEPS

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["task_mode"] = "FULL_RANDOM_V2_REACHABLE"
        info["sampling_mode"] = "random_spawn_balanced_reachable_goal"
        info["full_random_v2"] = True
        info["goal_config_m"] = [GOAL_MIN_M, GOAL_MAX_M]
        info["goal_angle_limit_deg"] = GOAL_ANGLE_DEG
        return obs, info


def yaw_bin(yaw_deg: float) -> int:
    wrapped = (float(yaw_deg) + 180.0) % 360.0
    return int(np.clip(math.floor(wrapped / 30.0), 0, 11))


def audit_resets() -> dict[str, Any]:
    env = FullRandomV2ReachableEnv(seed=AUDIT_SEED_BASE)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        for ep in range(AUDIT_EPISODES):
            seed = AUDIT_SEED_BASE + ep
            try:
                _, info = env.reset(seed=seed)
            except Exception as exc:
                failures.append({"seed": seed, "error": repr(exc)})
                print(f"AUDIT_RESET_FAIL seed={seed} error={exc!r}")
                continue

            radius = (
                None
                if not math.isfinite(env.last_effective_radius_m)
                else float(env.last_effective_radius_m)
            )

            row = {
                "seed": seed,
                "spawn_x": float(info["spawn_x"]),
                "spawn_y": float(info["spawn_y"]),
                "spawn_yaw_deg": float(info["spawn_yaw_deg"]),
                "spawn_clearance_m": float(
                    info["spawn_pool_clearance_m"]
                ),
                "goal_distance_m": float(info["goal_dist"]),
                "goal_angle_deg": float(info["goal_angle_deg"]),
                "desired_goal_sign": int(info["desired_goal_sign"]),
                "effective_radius_m": radius,
                "kinematic_margin_m": float(
                    env.last_kinematic_margin_m
                ),
                "arc_corridor_margin_m": float(
                    env.last_arc_corridor_margin_m
                ),
                "arc_checked_points": int(
                    env.last_arc_checked_points
                ),
                "bootstrap_goal_discarded": bool(
                    info.get("bootstrap_goal_discarded", False)
                ),
            }
            rows.append(row)

            if (ep + 1) % 50 == 0:
                print("AUDIT_RESETS_DONE =", ep + 1)

    finally:
        env.close()

    if not rows:
        report = {
            "passed": False,
            "accepted": 0,
            "failures": failures,
        }
        atomic_json(AUDIT_PATH, report)
        return report

    xs = np.asarray([r["spawn_x"] for r in rows], dtype=float)
    ys = np.asarray([r["spawn_y"] for r in rows], dtype=float)
    yaws = np.asarray([r["spawn_yaw_deg"] for r in rows], dtype=float)
    angles = np.asarray([r["goal_angle_deg"] for r in rows], dtype=float)
    dists = np.asarray([r["goal_distance_m"] for r in rows], dtype=float)
    spawn_clear = np.asarray(
        [r["spawn_clearance_m"] for r in rows],
        dtype=float,
    )
    kin_margin = np.asarray(
        [r["kinematic_margin_m"] for r in rows],
        dtype=float,
    )
    arc_margin = np.asarray(
        [r["arc_corridor_margin_m"] for r in rows],
        dtype=float,
    )
    finite_radii = np.asarray(
        [
            r["effective_radius_m"]
            for r in rows
            if r["effective_radius_m"] is not None
        ],
        dtype=float,
    )

    neg = int(np.sum(angles < -0.5))
    pos = int(np.sum(angles > 0.5))
    yaw_bins = sorted(set(yaw_bin(v) for v in yaws))

    x_edges = np.linspace(float(xs.min()), float(xs.max()) + 1e-9, 9)
    y_edges = np.linspace(float(ys.min()), float(ys.max()) + 1e-9, 9)
    xb = np.clip(np.digitize(xs, x_edges) - 1, 0, 7)
    yb = np.clip(np.digitize(ys, y_edges) - 1, 0, 7)
    spatial_cells = len(set(zip(xb.tolist(), yb.tolist())))

    checks = {
        "all_300_reset_ok": len(rows) == AUDIT_EPISODES,
        "zero_reset_failure": len(failures) == 0,
        "negative_goals_present": neg > 0,
        "positive_goals_present": pos > 0,
        "balanced_signs": abs(neg - pos) <= 10,
        "full_360_yaw_bins": len(yaw_bins) == 12,
        "spawn_clearance_ok": bool(
            np.min(spawn_clear) >= SPAWN_CLEARANCE_MIN_M - 1e-6
        ),
        "turn_radius_ok": bool(
            finite_radii.size == 0
            or np.min(finite_radii) >= TURN_RADIUS_M - 1e-6
        ),
        "kinematic_margin_ok": bool(
            np.min(kin_margin) >= KINEMATIC_MARGIN_M - 1e-6
        ),
        "arc_corridor_ok": bool(
            np.min(arc_margin) >= ARC_CORRIDOR_MARGIN_M - 1e-6
        ),
        "bootstrap_discard_ok": all(
            r["bootstrap_goal_discarded"] for r in rows
        ),
    }

    report = {
        "passed": bool(all(checks.values())),
        "config": {
            "goal_min_m": GOAL_MIN_M,
            "goal_max_m": GOAL_MAX_M,
            "goal_angle_deg": GOAL_ANGLE_DEG,
            "turn_radius_m": TURN_RADIUS_M,
            "kinematic_margin_m": KINEMATIC_MARGIN_M,
            "arc_corridor_margin_m": ARC_CORRIDOR_MARGIN_M,
            "spawn_clearance_min_m": SPAWN_CLEARANCE_MIN_M,
        },
        "accepted": len(rows),
        "reset_failures": len(failures),
        "negative_goal_count": neg,
        "positive_goal_count": pos,
        "yaw_min_deg": float(np.min(yaws)),
        "yaw_max_deg": float(np.max(yaws)),
        "occupied_30deg_yaw_bins": yaw_bins,
        "spatial_cells_8x8": int(spatial_cells),
        "spawn_x_range": [float(np.min(xs)), float(np.max(xs))],
        "spawn_y_range": [float(np.min(ys)), float(np.max(ys))],
        "spawn_clearance_min_m": float(np.min(spawn_clear)),
        "goal_distance_range_m": [
            float(np.min(dists)),
            float(np.max(dists)),
        ],
        "goal_angle_range_deg": [
            float(np.min(angles)),
            float(np.max(angles)),
        ],
        "effective_radius_min_m": (
            None
            if finite_radii.size == 0
            else float(np.min(finite_radii))
        ),
        "kinematic_margin_min_m": float(np.min(kin_margin)),
        "arc_corridor_margin_min_m": float(np.min(arc_margin)),
        "arc_checked_points_min": int(
            min(r["arc_checked_points"] for r in rows)
        ),
        "checks": checks,
        "failures": failures,
        "rows": rows,
    }

    atomic_json(AUDIT_PATH, report)
    return report


def evaluate_joint_candidate() -> dict[str, Any]:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC.load(str(SOURCE_MODEL), device=device)
    env = FullRandomV2ReachableEnv(seed=EVAL_SEED_BASE + 9999)

    rows: list[dict[str, Any]] = []

    try:
        for ep in range(EVAL_EPISODES):
            seed = EVAL_SEED_BASE + ep
            obs, reset_info = env.reset(seed=seed)

            initial_distance = float(reset_info["goal_dist"])
            initial_angle = float(reset_info["goal_angle"])
            final_info: dict[str, Any] = {}
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

            row = {
                "seed": seed,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": steps,
                "initial_distance_m": initial_distance,
                "initial_angle_deg": math.degrees(initial_angle),
                "final_distance_m": float(final_distance),
                "final_angle_deg": math.degrees(float(final_angle)),
                "progress_ratio": progress,
                "mean_speed_mps": speed_sum / max(steps, 1),
                "spawn_clearance_m": float(
                    reset_info["spawn_pool_clearance_m"]
                ),
                "effective_radius_m": (
                    None
                    if not math.isfinite(env.last_effective_radius_m)
                    else float(env.last_effective_radius_m)
                ),
                "kinematic_margin_m": float(
                    env.last_kinematic_margin_m
                ),
                "arc_corridor_margin_m": float(
                    env.last_arc_corridor_margin_m
                ),
            }
            rows.append(row)

            if (ep + 1) % 10 == 0:
                s = sum(r["success"] for r in rows)
                c = sum(r["collision"] for r in rows)
                t = sum(r["timeout"] for r in rows)
                n = len(rows)
                print(
                    f"EVAL_DONE={n} "
                    f"SUCCESS={s/n:.4f} "
                    f"COLLISION={c/n:.4f} "
                    f"TIMEOUT={t/n:.4f}"
                )

    finally:
        env.close()

    n = len(rows)
    report = {
        "episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "timeout_rate": sum(r["timeout"] for r in rows) / n,
        "progress_ratio": float(
            np.mean([r["progress_ratio"] for r in rows])
        ),
        "speed_mps": float(
            np.mean([r["mean_speed_mps"] for r in rows])
        ),
        "negative_goal_count": int(
            sum(r["initial_angle_deg"] < -0.5 for r in rows)
        ),
        "positive_goal_count": int(
            sum(r["initial_angle_deg"] > 0.5 for r in rows)
        ),
        "old_matched_baseline": OLD_MATCHED_BASELINE,
        "delta_vs_old": {},
        "rows": rows,
    }

    for key in (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "progress_ratio",
    ):
        report["delta_vs_old"][key] = (
            report[key] - OLD_MATCHED_BASELINE[key]
        )

    atomic_json(EVAL_PATH, report)
    return report


def main() -> None:
    print("STAGE = STAGE7_FULL_RANDOM_V2_REACHABLE_FIX")
    print("TRAINING = False")
    print("PURE_LIDAR_END_TO_END_SAC = True")
    print("PLANNER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("STEP_QPOS_QVEL_OVERRIDE = False")
    print("RESET_INITIAL_STATE_RANDOMIZATION = True")
    print("CAR_POSITION_RANDOM = True")
    print("CAR_YAW_RANDOM = True")
    print("GOAL_RANDOM = True")
    print("BOOTSTRAP_FIX = ReachableRandomSpawnR0Env")
    print(
        "FULL_RANDOM_V2_CONFIG =",
        {
            "goal_distance_m": [GOAL_MIN_M, GOAL_MAX_M],
            "goal_angle_deg": GOAL_ANGLE_DEG,
            "turn_radius_m": TURN_RADIUS_M,
            "kinematic_margin_m": KINEMATIC_MARGIN_M,
            "arc_corridor_margin_m": ARC_CORRIDOR_MARGIN_M,
            "spawn_clearance_min_m": SPAWN_CLEARANCE_MIN_M,
        },
    )

    print("\n========== 300 RESET AUDIT ==========")
    audit = audit_resets()

    print("\n========== RESET AUDIT SUMMARY ==========")
    for key in (
        "passed",
        "accepted",
        "reset_failures",
        "negative_goal_count",
        "positive_goal_count",
        "yaw_min_deg",
        "yaw_max_deg",
        "occupied_30deg_yaw_bins",
        "spatial_cells_8x8",
        "spawn_clearance_min_m",
        "goal_distance_range_m",
        "goal_angle_range_deg",
        "effective_radius_min_m",
        "kinematic_margin_min_m",
        "arc_corridor_margin_min_m",
        "arc_checked_points_min",
        "checks",
    ):
        if key in audit:
            print(key.upper(), "=", audit[key])

    print("AUDIT_REPORT =", AUDIT_PATH)

    if not audit.get("passed", False):
        print("FULL_RANDOM_V2_RESET_AUDIT = FAIL")
        print("EVALUATION_SKIPPED = True")
        return

    print("FULL_RANDOM_V2_RESET_AUDIT = PASS")

    print("\n========== JOINT CANDIDATE MATCHED-80 EVAL ==========")
    result = evaluate_joint_candidate()

    print("\n========== FULL RANDOM V2 RESULT ==========")
    for key in (
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "progress_ratio",
        "speed_mps",
        "negative_goal_count",
        "positive_goal_count",
        "old_matched_baseline",
        "delta_vs_old",
    ):
        print(key.upper(), "=", result[key])

    print("EVAL_REPORT =", EVAL_PATH)
    print("TRAINING = False")
    print("DONE = True")


if __name__ == "__main__":
    main()
