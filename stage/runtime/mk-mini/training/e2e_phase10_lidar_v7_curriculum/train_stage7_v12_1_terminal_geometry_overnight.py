from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from training.e2e_phase10_lidar_v7_curriculum.train_stage7_basic_maneuver_m0_m1 import (
    BasicManeuverEnv,
    evaluate,
)
# ReachableRandomSpawnR0Env was introduced in a later Stage7 module.
# Do not hard-code a stale module name: resolve the actual local source once.
import importlib

def _resolve_reachable_r0_class():
    package_name = "training.e2e_phase10_lidar_v7_curriculum"
    package = importlib.import_module(package_name)
    package_dir = Path(package.__file__).resolve().parent

    matches = []
    for path in sorted(package_dir.glob("train_stage7*.py")):
        if path.name.startswith("train_stage7_v12"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "class ReachableRandomSpawnR0Env" in source:
            matches.append(path)

    if not matches:
        raise ImportError(
            "Cannot locate class ReachableRandomSpawnR0Env under "
            f"{package_dir}. Run: grep -R "
            "'class ReachableRandomSpawnR0Env' -n "
            "training/e2e_phase10_lidar_v7_curriculum"
        )

    errors = []
    for path in matches:
        module_name = f"{package_name}.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, "ReachableRandomSpawnR0Env", None)
            if cls is not None:
                return cls, module_name, str(path)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")

    raise ImportError(
        "Found ReachableRandomSpawnR0Env source file(s) but import failed: "
        + " | ".join(errors)
    )

ReachableRandomSpawnR0Env, REACHABLE_R0_MODULE, REACHABLE_R0_FILE = (
    _resolve_reachable_r0_class()
)
from training.e2e_phase10_lidar_v7_curriculum.train_stage7_v5_closed_loop_residual_sac_to_m2 import (
    ResidualEnv,
    Composite,
)

ROOT = Path("experiments/e2e_phase10_lidar_v7_curriculum")
BASE_MODEL = ROOT / "stage7_basic_maneuver_m0_m1/candidate_model.zip"
V5_MODEL = ROOT / "stage7_v5_closed_loop_residual_sac_to_m2/best_residual_sac.zip"
V10_R4 = ROOT / "stage7_v10_mixed_foundation_retrain/round_04_candidate.zip"

OUT = ROOT / "stage7_v12_terminal_geometry_overnight"
OUT.mkdir(parents=True, exist_ok=True)

STATE = OUT / "state.json"
HB = OUT / "heartbeat.json"
PRECHECK = OUT / "precheck.json"
BEST_REPORT = OUT / "best_report.json"
FINAL_REPORT = OUT / "final_report.json"

BEST_MODEL = OUT / "best_residual_sac.zip"
BEST_REPLAY = OUT / "best_residual_replay.pkl"

# Main fixes from visual inspection:
# 1. Center-only 0.30 m was too strict for "vehicle reached goal zone".
#    0.45 m center tolerance approximates the vehicle envelope entering a
#    small target neighborhood without requiring the chassis center to sit on it.
SUCCESS_RADIUS_M = 0.45
TERMINAL_ZONE_M = 0.68

# Goal center must not be generated right next to walls / static obstacles.
GOAL_STATIC_CLEARANCE_M = 0.60
MAX_SAFE_RESET_TRIES = 64

# Foundation training distribution.
TASK_LAYOUT = (
    "left",
    "right",
    "left",
    "right",
    "straight",
    "r0",
)

TRAIN_LR = 1.0e-5
BATCH_SIZE = 512
GRADIENT_STEPS = 1

A = {
    "straight": 2_610_000_000,
    "left": 2_611_000_000,
    "right": 2_612_000_000,
    "r0": 2_613_000_000,
}
B = {
    "straight": 2_620_000_000,
    "left": 2_621_000_000,
    "right": 2_622_000_000,
    "r0": 2_623_000_000,
}
F = {
    "straight": 2_630_000_000,
    "left": 2_631_000_000,
    "right": 2_632_000_000,
    "r0": 2_633_000_000,
}

QUICK_TURN = 32
QUICK_KEEP = 24


def savej(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def heartbeat(phase: str, **kwargs) -> None:
    savej(
        HB,
        {
            "wall_time": time.time(),
            "phase": phase,
            **kwargs,
        },
    )


def robot_body_ids(model) -> set[int]:
    roots = set()

    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            roots.add(int(model.jnt_bodyid[jid]))

    if not roots:
        return set()

    bodies = set(roots)
    changed = True

    while changed:
        changed = False
        for bid in range(1, model.nbody):
            if bid in bodies:
                continue
            parent = int(model.body_parentid[bid])
            if parent in bodies:
                bodies.add(bid)
                changed = True

    return bodies


def point_geom_clearance_xy(model, data, gid: int, point_xy) -> float:
    gtype = int(model.geom_type[gid])

    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return float("inf")

    center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)

    point3 = np.asarray(
        [float(point_xy[0]), float(point_xy[1]), center[2]],
        dtype=np.float64,
    )
    local = R.T @ (point3 - center)

    size = np.asarray(model.geom_size[gid], dtype=np.float64)

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        dx = max(abs(float(local[0])) - float(size[0]), 0.0)
        dy = max(abs(float(local[1])) - float(size[1]), 0.0)
        return float(math.hypot(dx, dy))

    if gtype in (
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    ):
        radial = float(np.linalg.norm(local[:2]))
        return max(0.0, radial - float(size[0]))

    # Conservative fallback for mesh/other static collision geoms.
    rbound = float(model.geom_rbound[gid])
    d = float(np.linalg.norm(np.asarray(point_xy) - center[:2]))
    return max(0.0, d - rbound)


def static_goal_clearance(env, goal_xy) -> float:
    model = env.model
    data = env.data

    robot_bodies = robot_body_ids(model)

    best = float("inf")

    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])

        if bid in robot_bodies:
            continue

        if (
            int(model.geom_contype[gid]) == 0
            and int(model.geom_conaffinity[gid]) == 0
        ):
            continue

        d = point_geom_clearance_xy(
            model,
            data,
            gid,
            goal_xy,
        )

        best = min(best, d)

    return float(best)


def extract_goal_xy(env, info):
    for name in (
        "goal_xy",
        "goal",
        "target_xy",
        "_goal_xy",
    ):
        value = getattr(env, name, None)
        if value is not None:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size >= 2:
                return arr[:2].copy()

    if isinstance(info, dict):
        value = info.get("goal_xy")
        if value is not None:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size >= 2:
                return arr[:2].copy()

    raise RuntimeError("Cannot locate goal_xy after reset")


class GeometryFixMixin:
    def _apply_terminal_geometry(self):
        self.success_radius = SUCCESS_RADIUS_M
        self.terminal_zone = max(
            TERMINAL_ZONE_M,
            SUCCESS_RADIUS_M + 0.12,
        )

        if hasattr(self, "goal_clearance_margin"):
            self.goal_clearance_margin = max(
                float(self.goal_clearance_margin),
                GOAL_STATIC_CLEARANCE_M,
            )

        if hasattr(self, "random_goal_clearance_margin"):
            self.random_goal_clearance_margin = max(
                float(self.random_goal_clearance_margin),
                GOAL_STATIC_CLEARANCE_M,
            )

    def _safe_reset(self, parent_reset, seed=None, options=None):
        self._apply_terminal_geometry()

        base_seed = (
            int(seed)
            if seed is not None
            else int(
                self._v12_safe_rng.integers(
                    100_000_000,
                    2_000_000_000,
                )
            )
        )

        last = None

        for attempt in range(MAX_SAFE_RESET_TRIES):
            # Large odd stride avoids repeatedly sampling nearly identical RNG states.
            actual_seed = int(base_seed + 104729 * attempt)

            self._apply_terminal_geometry()

            obs, info = parent_reset(
                seed=actual_seed,
                options=options,
            )

            self._apply_terminal_geometry()

            goal = extract_goal_xy(self, info)
            clearance = static_goal_clearance(
                self,
                goal,
            )

            last = (
                obs,
                info,
                goal,
                clearance,
                actual_seed,
            )

            if clearance >= GOAL_STATIC_CLEARANCE_M - 1e-9:
                info = dict(info)
                info["v12_goal_static_clearance_m"] = clearance
                info["v12_success_radius_m"] = SUCCESS_RADIUS_M
                info["v12_terminal_zone_m"] = self.terminal_zone
                info["v12_reset_attempts"] = attempt + 1
                info["v12_actual_seed"] = actual_seed
                return obs, info

        _, _, goal, clearance, actual_seed = last
        raise RuntimeError(
            "V12 could not obtain a safe goal after "
            f"{MAX_SAFE_RESET_TRIES} tries; "
            f"last_goal={goal.tolist()} "
            f"last_clearance={clearance:.3f} "
            f"last_seed={actual_seed}"
        )


class V12BasicEnv(GeometryFixMixin, BasicManeuverEnv):
    def __init__(self, *, fixed_skill: str, seed: int):
        super().__init__(
            fixed_skill=fixed_skill,
            seed=seed,
        )
        self._v12_safe_rng = np.random.default_rng(seed + 91_700_003)
        self._apply_terminal_geometry()

    def reset(self, *, seed=None, options=None):
        return self._safe_reset(
            super().reset,
            seed=seed,
            options=options,
        )


class V12R0Env(GeometryFixMixin, ReachableRandomSpawnR0Env):
    def __init__(self, seed: int):
        super().__init__(seed=seed)
        self._v12_safe_rng = np.random.default_rng(seed + 91_700_003)
        self._apply_terminal_geometry()

    def reset(self, *, seed=None, options=None):
        return self._safe_reset(
            super().reset,
            seed=seed,
            options=options,
        )


def v12_task(name: str, seed: int):
    if name == "r0":
        return V12R0Env(seed=seed)

    if name not in ("straight", "left", "right"):
        raise ValueError(name)

    return V12BasicEnv(
        fixed_skill=name,
        seed=seed,
    )


def make_factory(name: str, seed: int, base: SAC):
    def _factory():
        return Monitor(
            ResidualEnv(
                v12_task(name, seed),
                base,
            )
        )
    return _factory


def make_vec(seed_base: int, base: SAC):
    return DummyVecEnv(
        [
            make_factory(
                name,
                seed_base + i * 100_000,
                base,
            )
            for i, name in enumerate(TASK_LAYOUT)
        ]
    )


def set_lr(model: SAC, lr: float) -> None:
    def schedule(_progress_remaining):
        return float(lr)

    model.learning_rate = float(lr)
    model.lr_schedule = schedule

    for opt in (
        getattr(model.actor, "optimizer", None),
        getattr(model.critic, "optimizer", None),
        getattr(model, "ent_coef_optimizer", None),
    ):
        if opt is None:
            continue
        for pg in opt.param_groups:
            pg["lr"] = float(lr)


def suite(pol, seeds, tag, turn_ep=QUICK_TURN, keep_ep=QUICK_KEEP):
    heartbeat("eval", tag=tag)

    return {
        "straight": evaluate(
            pol,
            v12_task("straight", seeds["straight"] + 9999),
            keep_ep,
            seeds["straight"],
            f"{tag}_STRAIGHT_{keep_ep}",
        ),
        "left": evaluate(
            pol,
            v12_task("left", seeds["left"] + 9999),
            turn_ep,
            seeds["left"],
            f"{tag}_LEFT_{turn_ep}",
        ),
        "right": evaluate(
            pol,
            v12_task("right", seeds["right"] + 9999),
            turn_ep,
            seeds["right"],
            f"{tag}_RIGHT_{turn_ep}",
        ),
        "r0": evaluate(
            pol,
            v12_task("r0", seeds["r0"] + 9999),
            keep_ep,
            seeds["r0"],
            f"{tag}_R0_{keep_ep}",
        ),
    }


def turn_floor(r):
    return min(
        r["left"]["success_rate"],
        r["right"]["success_rate"],
    )


def turn_mean(r):
    return 0.5 * (
        r["left"]["success_rate"]
        + r["right"]["success_rate"]
    )


def score(r):
    return float(
        4.0 * turn_floor(r)
        + 2.0 * turn_mean(r)
        + 0.8 * r["straight"]["success_rate"]
        + 0.8 * r["r0"]["success_rate"]
        - 2.0 * (
            r["left"]["collision_rate"]
            + r["right"]["collision_rate"]
        )
    )


def choose_source(base: SAC, device: str):
    candidates = [
        ("V5_ACCEPTED", V5_MODEL),
    ]

    if V10_R4.is_file():
        candidates.append(("V10_R4", V10_R4))

    rows = []

    for name, path in candidates:
        residual = SAC.load(str(path), device=device)
        pol = Composite(base, residual)

        report = suite(
            pol,
            A,
            f"V12_SOURCE_{name}",
            turn_ep=32,
            keep_ep=24,
        )

        row = {
            "name": name,
            "path": str(path),
            "report": report,
            "turn_floor": turn_floor(report),
            "turn_mean": turn_mean(report),
            "score": score(report),
        }
        rows.append(row)

        print(
            "SOURCE_CANDIDATE =",
            name,
            "FLOOR =",
            row["turn_floor"],
            "MEAN =",
            row["turn_mean"],
            "SCORE =",
            row["score"],
        )

    rows.sort(
        key=lambda x: (
            x["turn_floor"],
            x["score"],
            x["turn_mean"],
        ),
        reverse=True,
    )

    selected = rows[0]
    print("SELECTED_SOURCE =", selected["name"])
    print("SELECTED_SOURCE_PATH =", selected["path"])

    return selected, rows


def reset_precheck(samples_per_skill: int = 24):
    result = {}

    for k, skill in enumerate(("straight", "left", "right", "r0")):
        env = v12_task(
            skill,
            2_600_000_000 + k * 1_000_000,
        )

        clearances = []
        attempts = []

        try:
            for ep in range(samples_per_skill):
                _, info = env.reset(
                    seed=2_600_100_000
                    + k * 1_000_000
                    + ep * 2
                )
                clearances.append(
                    float(info["v12_goal_static_clearance_m"])
                )
                attempts.append(
                    int(info["v12_reset_attempts"])
                )
        finally:
            env.close()

        result[skill] = {
            "samples": samples_per_skill,
            "min_goal_static_clearance_m": float(np.min(clearances)),
            "mean_goal_static_clearance_m": float(np.mean(clearances)),
            "max_reset_attempts": int(np.max(attempts)),
            "mean_reset_attempts": float(np.mean(attempts)),
        }

        print(
            "PRECHECK",
            skill.upper(),
            result[skill],
        )

    return result


def gate(current_a, cand_a):
    one_turn = 1.0 / QUICK_TURN
    one_keep = 1.0 / QUICK_KEEP

    checks = {
        "turn_floor_not_worse": (
            turn_floor(cand_a) >= turn_floor(current_a) - one_turn - 1e-12
        ),
        "turn_mean_not_worse": (
            turn_mean(cand_a) >= turn_mean(current_a) - one_turn / 2.0 - 1e-12
        ),
        "straight_retained": (
            cand_a["straight"]["success_rate"]
            >= current_a["straight"]["success_rate"] - one_keep - 1e-12
        ),
        "r0_retained": (
            cand_a["r0"]["success_rate"]
            >= current_a["r0"]["success_rate"] - one_keep - 1e-12
        ),
        "left_collision_ok": cand_a["left"]["collision_rate"] <= 0.0625,
        "right_collision_ok": cand_a["right"]["collision_rate"] <= 0.0625,
        "straight_collision_ok": cand_a["straight"]["collision_rate"] <= 0.0834,
        "r0_collision_ok": cand_a["r0"]["collision_rate"] <= 0.0834,
    }

    useful = bool(
        turn_floor(cand_a) >= turn_floor(current_a) + one_turn - 1e-12
        or score(cand_a) >= score(current_a) + 0.10
    )

    return {
        "passed": bool(all(checks.values()) and useful),
        "useful": useful,
        "checks": checks,
        "old_floor": turn_floor(current_a),
        "new_floor": turn_floor(cand_a),
        "old_mean": turn_mean(current_a),
        "new_mean": turn_mean(cand_a),
        "old_score": score(current_a),
        "new_score": score(cand_a),
    }


def confirm_b(current_b, cand_b):
    one_turn = 1.0 / QUICK_TURN
    one_keep = 1.0 / QUICK_KEEP

    checks = {
        "floor_not_worse": (
            turn_floor(cand_b) >= turn_floor(current_b) - one_turn - 1e-12
        ),
        "mean_not_worse": (
            turn_mean(cand_b) >= turn_mean(current_b) - one_turn / 2.0 - 1e-12
        ),
        "straight_retained": (
            cand_b["straight"]["success_rate"]
            >= current_b["straight"]["success_rate"] - one_keep - 1e-12
        ),
        "r0_retained": (
            cand_b["r0"]["success_rate"]
            >= current_b["r0"]["success_rate"] - one_keep - 1e-12
        ),
        "left_collision_ok": cand_b["left"]["collision_rate"] <= 0.0625,
        "right_collision_ok": cand_b["right"]["collision_rate"] <= 0.0625,
    }

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "old_floor": turn_floor(current_b),
        "new_floor": turn_floor(cand_b),
        "old_score": score(current_b),
        "new_score": score(cand_b),
    }


def formal(base: SAC, residual: SAC):
    pol = Composite(base, residual)

    r = {
        "straight": evaluate(
            pol,
            v12_task("straight", F["straight"] + 9999),
            80,
            F["straight"],
            "V12_FORMAL_STRAIGHT_80",
        ),
        "left": evaluate(
            pol,
            v12_task("left", F["left"] + 9999),
            80,
            F["left"],
            "V12_FORMAL_LEFT_80",
        ),
        "right": evaluate(
            pol,
            v12_task("right", F["right"] + 9999),
            80,
            F["right"],
            "V12_FORMAL_RIGHT_80",
        ),
        "r0": evaluate(
            pol,
            v12_task("r0", F["r0"] + 9999),
            80,
            F["r0"],
            "V12_FORMAL_R0_80",
        ),
    }

    r["formal_pass"] = bool(
        r["straight"]["success_rate"] >= 0.85
        and r["left"]["success_rate"] >= 0.75
        and r["right"]["success_rate"] >= 0.75
        and r["r0"]["success_rate"] >= 0.80
        and r["straight"]["collision_rate"] <= 0.05
        and r["left"]["collision_rate"] <= 0.05
        and r["right"]["collision_rate"] <= 0.05
        and r["r0"]["collision_rate"] <= 0.0625
    )

    return r


def prefill(base: SAC, source: Path, device: str, transitions: int):
    vec = make_vec(
        2_640_000_000,
        base,
    )

    model = SAC.load(
        str(source),
        env=vec,
        device=device,
    )

    # Fresh replay, source weights only.
    model.learning_starts = 0
    model.gradient_steps = 0

    heartbeat(
        "prefill",
        transitions=transitions,
    )

    model.learn(
        total_timesteps=int(transitions),
        reset_num_timesteps=False,
        progress_bar=False,
        log_interval=30,
    )

    model.save(str(BEST_MODEL))
    model.save_replay_buffer(str(BEST_REPLAY))
    vec.close()


def train_one_round(base: SAC, device: str, rnd: int, transitions: int):
    vec = make_vec(
        2_650_000_000 + rnd * 1_000_000,
        base,
    )

    model = SAC.load(
        str(BEST_MODEL),
        env=vec,
        device=device,
    )
    model.load_replay_buffer(str(BEST_REPLAY))

    model.learning_starts = 0
    model.batch_size = BATCH_SIZE
    model.gradient_steps = GRADIENT_STEPS
    set_lr(model, TRAIN_LR)

    heartbeat(
        "train",
        round=rnd,
        transitions=transitions,
    )

    try:
        model.learn(
            total_timesteps=int(transitions),
            reset_num_timesteps=False,
            progress_bar=False,
            log_interval=30,
        )
    finally:
        vec.close()

    cmodel = OUT / f"round_{rnd:02d}_candidate.zip"
    creplay = OUT / f"round_{rnd:02d}_candidate_replay.pkl"

    model.save(str(cmodel))
    model.save_replay_buffer(str(creplay))

    return model, cmodel, creplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--prefill", type=int, default=24000)
    ap.add_argument("--transitions", type=int, default=12000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("STAGE = STAGE7_V12_1_TERMINAL_GEOMETRY_OVERNIGHT")
    print("REACHABLE_R0_MODULE =", REACHABLE_R0_MODULE)
    print("REACHABLE_R0_FILE =", REACHABLE_R0_FILE)
    print("DEVICE =", device)
    print("VIDEO_DRIVEN_FIX = True")
    print("SUCCESS_RADIUS_M =", SUCCESS_RADIUS_M)
    print("TERMINAL_ZONE_M =", TERMINAL_ZONE_M)
    print("GOAL_STATIC_CLEARANCE_M =", GOAL_STATIC_CLEARANCE_M)
    print("CENTER_MUST_REACH_EXACT_TARGET = False")
    print("SAFE_GOAL_REJECTION = True")
    print("BASE_SAC_FROZEN = True")
    print("TRAINABLE = RESIDUAL_SAC")
    print("TASK_LAYOUT =", TASK_LAYOUT)
    print("TRAIN_LR =", TRAIN_LR)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("GRADIENT_STEPS =", GRADIENT_STEPS)
    print("ROUNDS =", args.rounds)
    print("PREFILL =", args.prefill)
    print("TRANSITIONS_PER_ROUND =", args.transitions)
    print("AUTO_M2 = False")

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(BASE_MODEL)
    if not V5_MODEL.is_file():
        raise FileNotFoundError(V5_MODEL)

    pre = reset_precheck(samples_per_skill=24)
    savej(PRECHECK, pre)

    for skill, row in pre.items():
        if row["min_goal_static_clearance_m"] < GOAL_STATIC_CLEARANCE_M - 1e-6:
            raise RuntimeError(
                f"Unsafe goal precheck for {skill}: {row}"
            )

    base = SAC.load(str(BASE_MODEL), device=device)
    for p in base.policy.parameters():
        p.requires_grad_(False)

    selected, source_rows = choose_source(
        base,
        device,
    )
    source_path = Path(selected["path"])

    source_res = SAC.load(
        str(source_path),
        device=device,
    )
    source_pol = Composite(base, source_res)

    current_a = selected["report"]
    current_b = suite(
        source_pol,
        B,
        "V12_SOURCE_SELECTED_B",
    )

    # Fresh corrected replay.
    prefill(
        base,
        source_path,
        device,
        args.prefill,
    )

    state = {
        "status": "RUNNING",
        "source": selected["name"],
        "source_path": str(source_path),
        "promotions": 0,
        "rounds_completed": 0,
        "success_radius_m": SUCCESS_RADIUS_M,
        "goal_static_clearance_m": GOAL_STATIC_CLEARANCE_M,
        "source_comparison": source_rows,
    }
    savej(STATE, state)

    for rnd in range(1, args.rounds + 1):
        print(f"\n========== V12 ROUND {rnd:02d} ==========")

        model, cmodel, creplay = train_one_round(
            base,
            device,
            rnd,
            args.transitions,
        )

        pol = Composite(base, model)

        cand_a = suite(
            pol,
            A,
            f"V12_R{rnd:02d}_A",
        )

        g_a = gate(
            current_a,
            cand_a,
        )
        print("GATE_A =", g_a)

        promoted = False
        cand_b = None
        g_b = None

        if g_a["passed"]:
            cand_b = suite(
                pol,
                B,
                f"V12_R{rnd:02d}_B",
            )

            g_b = confirm_b(
                current_b,
                cand_b,
            )
            print("GATE_B =", g_b)

            if g_b["passed"]:
                # Combined confirmation:
                combined_old = (
                    turn_floor(current_a)
                    + turn_floor(current_b)
                )
                combined_new = (
                    turn_floor(cand_a)
                    + turn_floor(cand_b)
                )

                promoted = bool(
                    combined_new
                    >= combined_old + 1.0 / QUICK_TURN - 1e-12
                    or (
                        combined_new >= combined_old - 1e-12
                        and score(cand_a) + score(cand_b)
                        >= score(current_a) + score(current_b) + 0.15
                    )
                )

        if promoted:
            shutil.copy2(cmodel, BEST_MODEL)
            shutil.copy2(creplay, BEST_REPLAY)

            current_a = cand_a
            current_b = cand_b

            state["promotions"] += 1
            state["last_promotion_round"] = rnd

            savej(
                BEST_REPORT,
                {
                    "round": rnd,
                    "A": current_a,
                    "B": current_b,
                    "gate_A": g_a,
                    "gate_B": g_b,
                },
            )

            print("V12_PROMOTED = True")
        else:
            print("V12_PROMOTED = False")

        state["rounds_completed"] = rnd
        state["A_straight"] = current_a["straight"]["success_rate"]
        state["A_left"] = current_a["left"]["success_rate"]
        state["A_right"] = current_a["right"]["success_rate"]
        state["A_r0"] = current_a["r0"]["success_rate"]
        state["B_straight"] = current_b["straight"]["success_rate"]
        state["B_left"] = current_b["left"]["success_rate"]
        state["B_right"] = current_b["right"]["success_rate"]
        state["B_r0"] = current_b["r0"]["success_rate"]

        savej(STATE, state)
        heartbeat(
            "round_done",
            round=rnd,
            promotions=state["promotions"],
        )

    best = SAC.load(
        str(BEST_MODEL),
        device=device,
    )

    f = formal(
        base,
        best,
    )

    result = {
        "status": (
            "V12_FOUNDATION_PASS"
            if f["formal_pass"]
            else (
                "V12_FOUNDATION_PROGRESS"
                if state["promotions"] > 0
                else "V12_NO_PROMOTION"
            )
        ),
        "promotions": state["promotions"],
        "source": state["source"],
        "formal": f,
        "success_radius_m": SUCCESS_RADIUS_M,
        "terminal_zone_m": TERMINAL_ZONE_M,
        "goal_static_clearance_m": GOAL_STATIC_CLEARANCE_M,
        "protected_v5_untouched": True,
        "auto_m2": False,
    }

    savej(FINAL_REPORT, result)

    state["status"] = result["status"]
    state["formal_pass"] = f["formal_pass"]
    savej(STATE, state)

    print("\n========== V12 FINAL ==========")
    print("STATUS =", result["status"])
    print("SOURCE =", result["source"])
    print("PROMOTIONS =", result["promotions"])
    print("FORMAL_STRAIGHT =", f["straight"]["success_rate"])
    print("FORMAL_LEFT =", f["left"]["success_rate"])
    print("FORMAL_RIGHT =", f["right"]["success_rate"])
    print("FORMAL_R0 =", f["r0"]["success_rate"])
    print("FORMAL_STRAIGHT_COLLISION =", f["straight"]["collision_rate"])
    print("FORMAL_LEFT_COLLISION =", f["left"]["collision_rate"])
    print("FORMAL_RIGHT_COLLISION =", f["right"]["collision_rate"])
    print("FORMAL_R0_COLLISION =", f["r0"]["collision_rate"])
    print("FORMAL_PASS =", f["formal_pass"])
    print("AUTO_M2 = False")
    print("DONE = True")


if __name__ == "__main__":
    main()
