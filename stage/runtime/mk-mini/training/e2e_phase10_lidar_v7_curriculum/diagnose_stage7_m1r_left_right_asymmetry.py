
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC

from training.e2e_phase10_lidar_v7_curriculum.train_stage7_basic_maneuver_m0_m1 import BasicManeuverEnv

STAGE = "STAGE7_M1R_LEFT_RIGHT_ASYMMETRY_DIAGNOSTIC"
MODEL_PATH = Path("experiments/e2e_phase10_lidar_v7_curriculum/stage7_basic_maneuver_m0_m1/candidate_model.zip")
OUT = Path("experiments/e2e_phase10_lidar_v7_curriculum/stage7_m1r_left_right_diagnostic")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "diagnostic_report.json"

PHYSICAL_PAIRS = 10
PHYSICAL_STEPS = 180
SPEED_ACTION = 0.45
STEER_ACTION = 0.55
MIN_CLEARANCE = 1.10

POLICY_PAIRS = 30
GOAL_DIST = 1.75
GOAL_ANGLE_DEG = 6.0
ROLLOUT_STEPS = 220


def wrap(a):
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def pose(env):
    p = env._pose()
    return float(p[0]), float(p[1]), float(p[2])


def joint_qpos_from_actuator(env, aid):
    jid = int(env.model.actuator_trnid[int(aid), 0])
    if jid < 0:
        return None
    adr = int(env.model.jnt_qposadr[jid])
    return float(env.data.qpos[adr])


def steer_state(env):
    return {
        "lf_joint": joint_qpos_from_actuator(env, env.id_lf_steer),
        "rf_joint": joint_qpos_from_actuator(env, env.id_rf_steer),
        "lf_ctrl": float(env.data.ctrl[env.id_lf_steer]),
        "rf_ctrl": float(env.data.ctrl[env.id_rf_steer]),
        "lr_ctrl": float(env.data.ctrl[env.id_lr_wheel]),
        "rr_ctrl": float(env.data.ctrl[env.id_rr_wheel]),
    }


def physical_run(env, seed, steer):
    _, info0 = env.reset(seed=seed)
    x0, y0, yaw0 = pose(env)
    px, py, pyaw = x0, y0, yaw0
    yaw_sum = 0.0
    path = 0.0
    lf, rf, speeds = [], [], []
    collision = False
    action = np.array([SPEED_ACTION, steer], dtype=np.float32)

    for i in range(PHYSICAL_STEPS):
        _, _, term, trunc, info = env.step(action)
        x, y, yaw = pose(env)
        path += math.hypot(x-px, y-py)
        yaw_sum += wrap(yaw-pyaw)
        px, py, pyaw = x, y, yaw
        ss = steer_state(env)
        if ss["lf_joint"] is not None:
            lf.append(ss["lf_joint"])
        if ss["rf_joint"] is not None:
            rf.append(ss["rf_joint"])
        speeds.append(float(info.get("actual_speed_mps", 0.0)))
        collision |= bool(info.get("collision", False))
        if term or trunc:
            break

    return {
        "seed": seed,
        "steer_action": float(steer),
        "steps": i+1,
        "spawn_clearance": float(info0.get("spawn_pool_clearance_m", float("nan"))),
        "yaw_deg": math.degrees(yaw_sum),
        "path_m": path,
        "speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "lf_joint_mean": float(np.mean(lf)) if lf else None,
        "rf_joint_mean": float(np.mean(rf)) if rf else None,
        "collision": collision,
        "final_ctrl": steer_state(env),
    }


def find_seeds(env, start, count, clearance):
    out = []
    s = start
    while len(out) < count and s < start + 10000:
        _, info = env.reset(seed=s)
        if float(info.get("spawn_pool_clearance_m", 0.0)) >= clearance:
            out.append(s)
        s += 2
    if len(out) < count:
        raise RuntimeError(f"not enough safe seeds: {len(out)}")
    return out


def ratio(a, b):
    lo = max(min(abs(a), abs(b)), 1e-9)
    return max(abs(a), abs(b)) / lo


def physical_diag():
    env = BasicManeuverEnv(seed=310_000_000, fixed_skill="straight")
    pairs = []
    try:
        seeds = find_seeds(env, 310_000_000, PHYSICAL_PAIRS, MIN_CLEARANCE)
        for idx, seed in enumerate(seeds, 1):
            pos = physical_run(env, seed, +STEER_ACTION)
            neg = physical_run(env, seed, -STEER_ACTION)
            pairs.append({"seed": seed, "positive": pos, "negative": neg})
            print(f"PHY {idx:02d} seed={seed} +yaw={pos['yaw_deg']:+.3f} -yaw={neg['yaw_deg']:+.3f} +path={pos['path_m']:.3f} -path={neg['path_m']:.3f} +col={pos['collision']} -col={neg['collision']}")
    finally:
        env.close()

    valid = [p for p in pairs if not p["positive"]["collision"] and not p["negative"]["collision"]]
    if not valid:
        raise RuntimeError("no collision-free physical pairs")

    py = np.array([abs(p["positive"]["yaw_deg"]) for p in valid])
    ny = np.array([abs(p["negative"]["yaw_deg"]) for p in valid])
    pp = np.array([p["positive"]["path_m"] for p in valid])
    npth = np.array([p["negative"]["path_m"] for p in valid])

    ps = float(np.mean([p["positive"]["yaw_deg"] for p in valid]))
    ns = float(np.mean([p["negative"]["yaw_deg"] for p in valid]))

    pos_lf = [abs(p["positive"]["lf_joint_mean"]) for p in valid if p["positive"]["lf_joint_mean"] is not None]
    neg_lf = [abs(p["negative"]["lf_joint_mean"]) for p in valid if p["negative"]["lf_joint_mean"] is not None]
    pos_rf = [abs(p["positive"]["rf_joint_mean"]) for p in valid if p["positive"]["rf_joint_mean"] is not None]
    neg_rf = [abs(p["negative"]["rf_joint_mean"]) for p in valid if p["negative"]["rf_joint_mean"] is not None]

    steer_ratios = []
    if pos_lf and neg_lf:
        steer_ratios.append(ratio(np.mean(pos_lf), np.mean(neg_lf)))
    if pos_rf and neg_rf:
        steer_ratios.append(ratio(np.mean(pos_rf), np.mean(neg_rf)))
    steer_ratio = max(steer_ratios) if steer_ratios else float("nan")

    yaw_ratio = ratio(float(np.mean(py)), float(np.mean(ny)))
    warning = bool(yaw_ratio > 1.25 or (math.isfinite(steer_ratio) and steer_ratio > 1.20))

    return {
        "valid_pairs": len(valid),
        "positive_mean_abs_yaw_deg": float(np.mean(py)),
        "negative_mean_abs_yaw_deg": float(np.mean(ny)),
        "yaw_abs_ratio": float(yaw_ratio),
        "positive_mean_path_m": float(np.mean(pp)),
        "negative_mean_path_m": float(np.mean(npth)),
        "path_ratio": float(ratio(np.mean(pp), np.mean(npth))),
        "positive_mean_signed_yaw_deg": ps,
        "negative_mean_signed_yaw_deg": ns,
        "positive_steer_causes_left": bool(ps > ns),
        "steering_joint_abs_ratio": float(steer_ratio),
        "physical_asymmetry_warning": warning,
        "pairs": pairs,
    }


def install_goal(env, dist, angle_deg):
    center, sensor, lidar, yaw = env._lidar_geometry()
    a = math.radians(angle_deg)
    wa = yaw + a
    cand = center + dist * np.array([math.cos(wa), math.sin(wa)], dtype=np.float64)

    if not env._candidate_reachable_and_safe(
        candidate=cand,
        distance=float(dist),
        relative_angle=float(a),
        sensor_xy=sensor,
        lidar_m=lidar,
        yaw=float(yaw),
    ):
        return None

    env.goal_xy = cand
    env.prev_action[:] = 0.0
    env.history.clear()
    if hasattr(env, "_previous_action"):
        env._previous_action.fill(0.0)
    if hasattr(env, "_previous_delta"):
        env._previous_delta.fill(0.0)

    env.prev_goal_dist = float(dist)
    env._initial_goal_distance = float(dist)
    env._previous_goal_distance = float(dist)
    env._previous_goal_angle = float(a)
    env._previous_xy = np.asarray(env.data.qpos[:2], dtype=np.float64).copy()
    if hasattr(env, "_previous_stage4_action"):
        env._previous_stage4_action.fill(0.0)
    if hasattr(env, "_idle_steps"):
        env._idle_steps = 0

    obs0 = np.asarray(env._observation(update_history=False), dtype=np.float32)
    frame = env._history_frame(obs0[:48], float(dist), float(a))
    for _ in range(env.HISTORY_FRAMES):
        env.history.append(frame.copy())
    return np.asarray(env._observation(False), dtype=np.float32)


def find_policy_seeds(env):
    out = []
    s = 320_000_000
    while len(out) < POLICY_PAIRS and s < 320_010_000:
        _, info = env.reset(seed=s)
        if float(info.get("spawn_pool_clearance_m", 0.0)) < 1.0:
            s += 2
            continue
        left = install_goal(env, GOAL_DIST, +GOAL_ANGLE_DEG)
        env.reset(seed=s)
        right = install_goal(env, GOAL_DIST, -GOAL_ANGLE_DEG)
        if left is not None and right is not None:
            out.append(s)
        s += 2
    if len(out) < POLICY_PAIRS:
        raise RuntimeError(f"not enough paired policy seeds: {len(out)}")
    return out


def policy_branch(model, env, seed, angle_deg):
    env.reset(seed=seed)
    obs = install_goal(env, GOAL_DIST, angle_deg)
    if obs is None:
        raise RuntimeError("paired goal unexpectedly unsafe")

    a0, _ = model.predict(obs, deterministic=True)
    a0 = np.asarray(a0, dtype=np.float64).reshape(2)

    speeds, steers = [], []
    yaw_sum = 0.0
    pyaw = pose(env)[2]
    collision = False
    initial_dist = GOAL_DIST

    for i in range(ROLLOUT_STEPS):
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float64).reshape(2)
        speeds.append(float(a[0]))
        steers.append(float(a[1]))
        obs, _, term, trunc, info = env.step(a.astype(np.float32))
        yaw = pose(env)[2]
        yaw_sum += wrap(yaw-pyaw)
        pyaw = yaw
        collision |= bool(info.get("collision", False))
        if term or trunc:
            break

    _, final_dist, _ = env._goal_state()
    prog = float(np.clip((initial_dist-float(final_dist))/initial_dist, -1.0, 1.0))

    return {
        "initial_speed": float(a0[0]),
        "initial_steer": float(a0[1]),
        "mean_speed": float(np.mean(speeds)),
        "mean_steer": float(np.mean(steers)),
        "mean_abs_steer": float(np.mean(np.abs(steers))),
        "yaw_deg": math.degrees(yaw_sum),
        "progress": prog,
        "collision": collision,
        "steps": i+1,
    }


def policy_diag(model, positive_causes_left):
    env = BasicManeuverEnv(seed=320_000_000, fixed_skill="straight")
    pairs = []
    try:
        seeds = find_policy_seeds(env)
        for idx, seed in enumerate(seeds, 1):
            left = policy_branch(model, env, seed, +GOAL_ANGLE_DEG)
            right = policy_branch(model, env, seed, -GOAL_ANGLE_DEG)
            pairs.append({"seed": seed, "left": left, "right": right})
            print(f"POL {idx:02d} seed={seed} L[v={left['initial_speed']:+.4f},s={left['initial_steer']:+.4f},p={left['progress']:.3f},c={left['collision']}] R[v={right['initial_speed']:+.4f},s={right['initial_steer']:+.4f},p={right['progress']:.3f},c={right['collision']}]")
    finally:
        env.close()

    li = np.array([p["left"]["initial_steer"] for p in pairs])
    ri = np.array([p["right"]["initial_steer"] for p in pairs])
    lv = np.array([p["left"]["initial_speed"] for p in pairs])
    rv = np.array([p["right"]["initial_speed"] for p in pairs])

    left_abs = float(np.mean(np.abs(li)))
    right_abs = float(np.mean(np.abs(ri)))
    expected_left = 1.0 if positive_causes_left else -1.0
    expected_right = -expected_left

    left_sign = float(np.mean(np.sign(li) == expected_left))
    right_sign = float(np.mean(np.sign(ri) == expected_right))
    lp = float(np.mean([p["left"]["progress"] for p in pairs]))
    rp = float(np.mean([p["right"]["progress"] for p in pairs]))
    lc = float(np.mean([p["left"]["collision"] for p in pairs]))
    rc = float(np.mean([p["right"]["collision"] for p in pairs]))

    mag_ratio = ratio(left_abs, right_abs)
    warning = bool(
        mag_ratio > 1.35
        or left_sign < 0.75
        or right_sign < 0.75
        or abs(lp-rp) > 0.08
    )

    return {
        "pairs": len(pairs),
        "left_initial_mean_speed": float(np.mean(lv)),
        "right_initial_mean_speed": float(np.mean(rv)),
        "speed_mirror_abs_error": float(np.mean(np.abs(lv-rv))),
        "left_initial_mean_steer": float(np.mean(li)),
        "right_initial_mean_steer": float(np.mean(ri)),
        "left_initial_mean_abs_steer": left_abs,
        "right_initial_mean_abs_steer": right_abs,
        "steer_magnitude_ratio": float(mag_ratio),
        "steer_mirror_abs_error": float(np.mean(np.abs(li+ri))),
        "left_expected_sign_rate": left_sign,
        "right_expected_sign_rate": right_sign,
        "left_short_progress": lp,
        "right_short_progress": rp,
        "left_collision_rate": lc,
        "right_collision_rate": rc,
        "policy_asymmetry_warning": warning,
        "raw_pairs": pairs,
    }


def root_cause(phys, pol):
    p = phys["physical_asymmetry_warning"]
    s = pol["policy_asymmetry_warning"]
    if p and s:
        return "BOTH_PHYSICAL_AND_POLICY_ASYMMETRY"
    if p:
        return "PRIMARY_PHYSICAL_ASYMMETRY"
    if s:
        return "PRIMARY_POLICY_ASYMMETRY"
    return "NO_LARGE_LEFT_RIGHT_ASYMMETRY_FOUND"


def main():
    print("STAGE =", STAGE)
    print("TRAINING = False")
    print("MODEL =", MODEL_PATH)
    print("ACTION_0 = rear-wheel speed")
    print("ACTION_1 = front steering")
    print("PLANNER = False")
    print("CAMERA = False")
    print("SAFETY_SHIELD = False")
    print("STEP_QPOS_QVEL_OVERRIDE = False")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", device)
    model = SAC.load(str(MODEL_PATH), device=device)

    print("\n========== PHYSICAL SYMMETRY ==========")
    phys = physical_diag()
    for k, v in phys.items():
        if k != "pairs":
            print(k.upper(), "=", v)

    print("\n========== POLICY MIRROR SYMMETRY ==========")
    pol = policy_diag(model, phys["positive_steer_causes_left"])
    for k, v in pol.items():
        if k != "raw_pairs":
            print(k.upper(), "=", v)

    diagnosis = root_cause(phys, pol)

    payload = {
        "stage": STAGE,
        "training": False,
        "model": str(MODEL_PATH),
        "physical": phys,
        "policy": pol,
        "diagnosis": diagnosis,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== FINAL DIAGNOSIS ==========")
    print("DIAGNOSIS =", diagnosis)
    print("REPORT =", REPORT)
    print("DONE = True")


if __name__ == "__main__":
    main()
