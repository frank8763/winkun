#!/usr/bin/env python3

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback

from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn


HOME = Path.home()

G16 = HOME / (
    "mkmini_stage_g16_p3_beta_sweep_20261017_v2/runner.py"
)

U4 = HOME / (
    "mkmini_stage_g17_u4_unknown_layout_eval_20261017_v1/"
    "summary.json"
)

P0C = HOME / (
    "mkmini_stage_g19_p0c_depth_gate_"
    "20261018_210551_340032366/summary.json"
)

P1A = HOME / (
    "mkmini_stage_g19_p1a_fusion_"
    "20261018_211302_776316757/summary.json"
)

CUT = datetime(
    2026, 10, 19, 8, 30,
    tzinfo=timezone(timedelta(hours=8))
).timestamp()

FID = 227
DIM = 2 * 30 * 40
MAXMEM = 1500

TARGET = "real_rect_obs_7"
CAM = "d435i_depth_camera"

FROZEN_AGENT = None
TRAIN_COUNTER = {"frames": 0}


def write_json(path, data):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ) + "\n"
    )

    os.replace(tmp, p)


def save_pt(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".partial")

    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Vision(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1),
            nn.SiLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),

            nn.Conv2d(32, 32, 3, 2, 1),
            nn.SiLU(),

            nn.Flatten(),
            nn.Linear(32 * 4 * 5, 64),
            nn.SiLU()
        )

        self.proj = nn.Linear(64, FID)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, image):
        return self.proj(
            self.encoder(image)
        )


class VisualPolicy:

    def __init__(self, C, agent, device):
        self.C = C
        self.agent = agent
        self.device = device

        self.vision = Vision().to(device)
        self.vision_target = copy.deepcopy(
            self.vision
        )

        for p in self.vision_target.parameters():
            p.requires_grad_(False)

        # 视觉参数同时参与真实 SAC 的
        # Critic 和 Actor 更新。
        self.agent.cp = torch.optim.Adam(
            list(agent.q1.parameters())
            + list(agent.q2.parameters())
            + list(self.vision.parameters()),
            lr=5e-5
        )

        self.agent.ap = torch.optim.Adam(
            list(agent.actor.parameters())
            + list(self.vision.parameters()),
            lr=2e-6
        )

        self.grad_actor = 0.0
        self.grad_critic = 0.0
        self.grad_encoder = 0.0

    def pack(self, feature, depth):
        x = np.concatenate((
            np.asarray(
                feature, np.float32
            ).reshape(FID),

            np.asarray(
                depth, np.float32
            ).reshape(DIM)
        ))

        if not np.isfinite(x).all():
            raise RuntimeError(
                "FUSED_FEATURE_NONFINITE"
            )

        return x

    def fused_state(self, x, target=False):
        t = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=self.device
        )

        if t.ndim == 1:
            t = t[None, :]

        if t.shape[1] != FID + DIM:
            raise RuntimeError(
                "FUSED_REPLAY_DIM_INVALID"
            )

        base = self.agent.state(
            t[:, :FID]
        )

        image = t[:, FID:].reshape(
            -1, 2, 30, 40
        )

        encoder = (
            self.vision_target
            if target
            else self.vision
        )

        return base + encoder(image)

    def choose(
        self,
        feature,
        depth,
        stochastic=False
    ):
        with torch.no_grad():
            state = self.fused_state(
                self.pack(feature, depth)
            )

            actions, _, probabilities, _ = (
                self.agent.actor.sample_all(
                    state,
                    deterministic=not stochastic
                )
            )

            probs = probabilities[0]

            if stochastic and random.random() < 0.20:
                index = random.randrange(3)

            elif stochastic:
                index = int(
                    torch.multinomial(
                        probs, 1
                    ).item()
                )

            else:
                index = int(
                    probs.argmax().item()
                )

            action = actions[0, index].cpu().numpy()
            action = action.astype(np.float32)

            if not np.isfinite(action).all():
                raise RuntimeError(
                    "POLICY_ACTION_NONFINITE"
                )

            return action, index

    def update(self, batch):
        sx, ix, act, ret, nx, discount, terminal = batch

        ag = self.agent

        s = self.fused_state(sx)
        n = self.fused_state(
            nx,
            target=True
        )

        action = torch.as_tensor(
            act, dtype=torch.float32,
            device=self.device
        )

        index = torch.as_tensor(
            ix, dtype=torch.long,
            device=self.device
        )

        reward = torch.as_tensor(
            ret, dtype=torch.float32,
            device=self.device
        )

        gamma = torch.as_tensor(
            discount, dtype=torch.float32,
            device=self.device
        )

        done = torch.as_tensor(
            terminal, dtype=torch.float32,
            device=self.device
        )

        # 真实 SAC Critic 更新
        with torch.no_grad():
            target = (
                reward
                + gamma * (1.0 - done)
                * ag._value(
                    n, (ag.t1, ag.t2)
                )
            )

        loss_q = (
            (ag.q1(s, action, index) - target)
            .square().mean()
            +
            (ag.q2(s, action, index) - target)
            .square().mean()
        )

        ag.cp.zero_grad()
        loss_q.backward()

        self.grad_critic = max(
            self.grad_critic,
            sum(
                float(
                    p.grad.detach().abs().sum().item()
                )
                for p in self.vision.parameters()
                if p.grad is not None
            )
        )

        torch.nn.utils.clip_grad_norm_(
            list(ag.q1.parameters())
            + list(ag.q2.parameters())
            + list(self.vision.parameters()),
            5.0
        )

        ag.cp.step()

        # Critic 更新后重新生成融合状态，
        # 避免同一计算图的原地参数版本冲突。
        for q in (ag.q1, ag.q2):
            for p in q.parameters():
                p.requires_grad_(False)

        try:
            s2 = self.fused_state(sx)

            aa, logpa, probabilities, logpd = (
                ag.actor.sample_all(s2)
            )

            terms = []

            for k in range(len(self.C.DURATIONS)):
                ids = torch.full(
                    (s2.shape[0],),
                    k,
                    device=self.device,
                    dtype=torch.long
                )

                qmin = torch.minimum(
                    ag.q1(s2, aa[:, k], ids),
                    ag.q2(s2, aa[:, k], ids)
                )

                terms.append(
                    ag.alpha_c * logpa[:, k]
                    + ag.alpha_d * logpd[:, k]
                    - qmin
                )

            loss_actor = (
                probabilities
                * torch.stack(
                    terms,
                    dim=-1
                )
            ).sum(-1).mean()

            ag.ap.zero_grad()
            loss_actor.backward()

            self.grad_actor = max(
                self.grad_actor,
                sum(
                    float(
                        p.grad.detach().abs().sum().item()
                    )
                    for p in self.vision.parameters()
                    if p.grad is not None
                )
            )

            first_grad = (
                self.vision.encoder[0]
                .weight.grad
            )

            if first_grad is not None:
                self.grad_encoder = max(
                    self.grad_encoder,
                    float(
                        first_grad.detach()
                        .abs().sum().item()
                    )
                )

            torch.nn.utils.clip_grad_norm_(
                list(ag.actor.parameters())
                + list(self.vision.parameters()),
                5.0
            )

            ag.ap.step()

        finally:
            for q in (ag.q1, ag.q2):
                for p in q.parameters():
                    p.requires_grad_(True)

        with torch.no_grad():
            for source, target_net in (
                (ag.q1, ag.t1),
                (ag.q2, ag.t2)
            ):
                for p, tp in zip(
                    source.parameters(),
                    target_net.parameters()
                ):
                    tp.lerp_(p, ag.tau)

            for p, tp in zip(
                self.vision.parameters(),
                self.vision_target.parameters()
            ):
                tp.lerp_(p, ag.tau)

        ag.updates += 1

        if not all(
            math.isfinite(float(v))
            for v in (
                loss_q.item(),
                loss_actor.item()
            )
        ):
            raise RuntimeError(
                "NONFINITE_REAL_SAC_LOSS"
            )

        return (
            float(loss_q.item()),
            float(loss_actor.item())
        )


def source_gate():
    u4 = json.loads(U4.read_text())
    p0c = json.loads(P0C.read_text())
    p1a = json.loads(P1A.read_text())

    if (
        p0c.get("gate")
        != "G19_P0C_DEPTH_COMPLEMENT_FOR_ENCODER_PASS"
    ):
        raise RuntimeError(
            "P0C_GATE_NOT_PASS"
        )

    if (
        p1a.get("gate")
        != "G19_P1A_FUSION_GRADIENT_GATE_PASS"
    ):
        raise RuntimeError(
            "P1A_GATE_NOT_PASS"
        )

    if sha(G16) != u4["runner_sha256"]:
        raise RuntimeError(
            "G16_SOURCE_CHANGED"
        )

    sys.path[:0] = [
        str(HOME / "catkin_ws/src/mk-mini"),
        "/tmp/stage_daad_g5b_selector_recovery_v1",
        str(
            HOME
            / "stage_daad_g4h_multibranch_night_v1/runner"
        )
    ]

    spec = importlib.util.spec_from_file_location(
        "g19_p1b_original",
        G16
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    prep = mod.do_prepare()
    mod.source_gate(prep)

    (
        g5b, H, C, N,
        proof, frozen, snap,
        d1, protected, device, ck
    ) = prep

    if sha(mod.FROZEN) != u4["frozen_sha256"]:
        raise RuntimeError(
            "FROZEN_CHANGED"
        )

    if (
        tuple(C.DURATIONS) != (1, 8, 24)
        or int(C.FEAT_N) != FID
    ):
        raise RuntimeError(
            "DAAD_CONTRACT_CHANGED"
        )

    paths = [
        G16,
        Path(mod.FROZEN),
        Path(C.__file__),
        Path(N.__file__),
        Path(g5b.__file__)
    ]

    hashes = {
        str(p): sha(p)
        for p in paths
    }

    return mod, prep, hashes


def check_sources(mod, prep, hashes):
    (
        g5b, H, C, N,
        proof, frozen, snap,
        d1, protected, device, ck
    ) = prep

    mod.assert_unchanged(
        g5b, H, proof, frozen, protected
    )

    for name, original in hashes.items():
        if sha(name) != original:
            raise RuntimeError(
                "PROTECTED_SOURCE_CHANGED"
            )


def new_model(mod, prep):
    (
        g5b, H, C, N,
        proof, frozen, snap,
        d1, protected, device, ck
    ) = prep

    agent = mod.agent_new(
        g5b, H, C, ck,
        snap, d1, device, 202710700
    )

    if int(agent.mean.numel()) != FID:
        raise RuntimeError(
            "NORMALIZER_DIM_INVALID"
        )

    if agent.mode != "adaptive":
        raise RuntimeError(
            "DAAD_ADAPTIVE_MODE_REQUIRED"
        )

    return VisualPolicy(C, agent, device)


def depth_frame(renderer, env):
    renderer.enable_depth_rendering()

    try:
        renderer.update_scene(
            env.data,
            camera=CAM
        )

        raw = np.asarray(
            renderer.render(),
            np.float32
        ).copy()

    finally:
        renderer.disable_depth_rendering()

    if (
        raw.shape != (120, 160)
        or not np.isfinite(raw).all()
    ):
        raise RuntimeError(
            "DEPTH_FRAME_INVALID"
        )

    valid = (
        (raw > 0.05)
        &
        (raw < 10.0)
    )

    normalized = np.where(
        valid,
        np.minimum(raw, 10.0) / 10.0,
        1.0
    ).astype(np.float32)

    # 120x160 -> 30x40
    fused = np.stack((
        normalized,
        valid.astype(np.float32)
    ), axis=0)

    return fused.reshape(
        2, 30, 4, 40, 4
    ).mean(axis=(2, 4)).astype(np.float32)


def create_nearfield(
    mj, N, project, task, seed, variant
):
    env = None
    renderer = None

    try:
        env, obs, info, goal = N.create(
            project,
            "reference",
            task,
            int(seed)
        )

        gid = mj.mj_name2id(
            env.model,
            mj.mjtObj.mjOBJ_GEOM,
            TARGET
        )

        if (
            gid < 0
            or int(env.model.geom_bodyid[gid]) != 0
        ):
            raise RuntimeError(
                "NEARFIELD_TARGET_INVALID"
            )

        start = np.asarray(
            env.data.qpos[:2],
            np.float64
        ).copy()

        goal = np.asarray(
            goal,
            np.float64
        ).reshape(2)

        direction = goal - start
        distance = float(
            np.linalg.norm(direction)
        )

        if distance < 0.6:
            raise RuntimeError(
                "GOAL_TOO_NEAR_FOR_OBSTACLE"
            )

        perpendicular = np.array([
            -direction[1],
            direction[0]
        ]) / distance

        sign = (
            1.0
            if variant == "left"
            else -1.0
        )

        obstacle_xy = (
            start
            + 0.54 * direction
            + sign * 0.16 * perpendicular
        )

        # 修改当前 env 内存中的几何。
        # 不修改 MJB 文件。
        env.model.geom_pos[gid, :2] = obstacle_xy
        env.model.geom_size[gid, :2] = [0.11, 0.13]

        mj.mj_forward(
            env.model,
            env.data
        )

        # N.create() has already reset the environment.
        # Keep its original robot state and sampled goal.
        # A second reset can resample the goal after geometry moves.

        actual_goal = np.asarray(
            env.goal_xy,
            np.float64
        ).reshape(2)

        if not np.allclose(
            actual_goal,
            goal,
            atol=1e-8,
            rtol=0
        ):
            raise RuntimeError(
                "GOAL_CHANGED_WITHOUT_RESET"
            )

        # MuJoCo geometry has already been updated with mj_forward().
        # Synchronize the native LiDAR against the modified geometry.
        obs = np.asarray(
            obs,
            dtype=np.float32
        ).reshape(-1).copy()

        if obs.shape != (112,):
            raise RuntimeError(
                "NATIVE_OBSERVATION_DIM_CHANGED"
            )

        raw_lidar = np.asarray(
            env._lidar48(),
            dtype=np.float32
        ).reshape(-1)

        if raw_lidar.shape != (48,) or not np.isfinite(raw_lidar).all():
            raise RuntimeError(
                "MODIFIED_GEOMETRY_LIDAR_INVALID"
            )

        # Prefer the environment's own complete observation builder.
        native_full_obs = False
        native_method = None

        # G19-P1C verified that the native reset uses:
        # self._observation(update_history=False)
        # Rebuild all 112 observation entries after geom change,
        # without resampling goal or modifying history.
        if not callable(getattr(env, "_observation", None)):
            raise RuntimeError(
                "G19_V3_NATIVE_OBSERVATION_METHOD_MISSING"
            )

        candidate = np.asarray(
            env._observation(update_history=False),
            dtype=np.float32
        ).reshape(-1)

        if (
            candidate.shape != (112,)
            or not np.isfinite(candidate).all()
        ):
            raise RuntimeError(
                "G19_V3_FULL_OBSERVATION_INVALID"
            )

        if not np.allclose(
            candidate[:48],
            raw_lidar,
            atol=1e-5,
            rtol=0
        ):
            raise RuntimeError(
                "G19_V3_FULL_OBSERVATION_LIDAR_MISMATCH"
            )

        obs = candidate.copy()
        native_full_obs = True
        native_method = "_observation(update_history=False)"

        for method_name in (
            "_get_obs",
            "_get_observation",
            "_get_observations",
            "_build_obs",
            "_make_obs",
            "get_obs",
            "get_observation",
        ):
            method = getattr(env, method_name, None)

            if not callable(method):
                continue

            try:
                candidate = np.asarray(
                    method(),
                    dtype=np.float32
                ).reshape(-1)

            except (TypeError, AttributeError, NotImplementedError):
                continue

            if (
                candidate.shape == (112,)
                and np.isfinite(candidate).all()
                and np.allclose(
                    candidate[:48],
                    raw_lidar,
                    atol=1e-5
                )
            ):
                obs = candidate.copy()
                native_full_obs = True
                native_method = method_name
                break

        if not native_full_obs:
            # The verified LiDAR component is refreshed.
            # Other fields retain N.create() values; this is
            # acceptable only for the foreground diagnostic,
            # not as proof of full observation synchronization.
            obs[:48] = raw_lidar

        if not np.allclose(
            obs[:48],
            env._lidar48(),
            atol=1e-5
        ):
            raise RuntimeError(
                "NEARFIELD_LIDAR_OBSERVATION_DESYNC"
            )

        if not np.isfinite(obs).all():
            raise RuntimeError(
                "NEARFIELD_OBSERVATION_NONFINITE"
            )

        if not getattr(create_nearfield, "_sync_reported", False):
            print(
                "G19_P1B_GOALFIX=PASS",
                "SECOND_RESET=FALSE",
                "FULL_NATIVE_OBS=", native_full_obs,
                "OBS_METHOD=", native_method,
                flush=True
            )
            create_nearfield._sync_reported = True

        if native_full_obs:
            create_nearfield._native_full_obs_verified = True
        else:
            create_nearfield._native_full_obs_verified = False

        if bool(env._collision()):
            raise RuntimeError(
                "INITIAL_COLLISION"
            )

        lidar = np.asarray(
            env._lidar48(),
            np.float32
        )

        obs = np.asarray(
            obs,
            np.float32
        )

        if not np.allclose(
            lidar,
            obs[:48],
            atol=1e-5
        ):
            raise RuntimeError(
                "LIDAR_RESET_NOT_SYNCHRONIZED"
            )

        renderer = mj.Renderer(
            env.model,
            height=120,
            width=160
        )

        depth = depth_frame(
            renderer, env
        )

        return (
            env,
            obs,
            actual_goal,
            renderer,
            depth,
            obstacle_xy.tolist()
        )

    except Exception:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()

        raise


def nearfield_gate(mod, prep):
    import mujoco as mj

    N = prep[3]
    records = []

    for variant in ("left", "right"):
        env, obs, goal, renderer, depth, xy = (
            create_nearfield(
                mj,
                N,
                mod.PR,
                "medium",
                202879001,
                variant
            )
        )

        records.append((
            depth.copy(),
            obs[:48].copy(),
            xy
        ))

        renderer.close()
        env.close()

    a, b = records

    changed = int(
        np.sum(
            np.abs(a[0] - b[0]) > 0.01
        )
    )

    lidar_delta = float(
        np.max(
            np.abs(a[1] - b[1])
        )
    )

    print(
        "G19_P1B_NEARFIELD_SENSOR",
        json.dumps({
            "depth_feature_cells_changed": changed,
            "lidar_max_diff": lidar_delta,
            "obstacle_positions": [
                a[2], b[2]
            ]
        }),
        flush=True
    )

    if changed < 4:
        raise RuntimeError(
            "NEARFIELD_VISUAL_INFORMATION_NOT_ESTABLISHED"
        )


def episode(
    mod, prep, policy,
    variant, seed,
    training=False,
    replay=None,
    frame_cap=None
):
    import mujoco as mj

    C = prep[2]
    N = prep[3]

    env = None
    renderer = None

    try:
        (
            env, obs, goal,
            renderer, depth, xy
        ) = create_nearfield(
            mj, N, mod.PR,
            "medium", seed, variant
        )

        prev = obs.copy()
        prev_action = np.zeros(
            2, np.float32
        )

        prev_len = 1
        steps = 0
        contact = False
        reached = False

        start_xy = np.asarray(
            env.data.qpos[:2],
            np.float64
        ).copy()

        max_steps = min(
            1100,
            frame_cap
            if frame_cap is not None
            else 1100
        )

        while steps < max_steps:
            feature = C.feat(
                obs,
                prev,
                prev_action,
                prev_len
            )

            if policy is None:
                with torch.no_grad():
                    state = FROZEN_AGENT.state(
                        feature
                    ).reshape(1, -1)

                    actions, _, probs, _ = (
                        FROZEN_AGENT.actor.sample_all(
                            state,
                            deterministic=True
                        )
                    )

                    idx = int(
                        probs[0].argmax().item()
                    )

                    action = (
                        actions[0, idx]
                        .cpu().numpy()
                        .astype(np.float32)
                    )

            else:
                action, idx = policy.choose(
                    feature,
                    depth,
                    stochastic=training
                )

            before = obs.copy()

            duration = min(
                int(C.DURATIONS[idx]),
                max_steps - steps
            )

            reward_total = 0.0
            actual = 0
            done = False
            truncated = False

            for j in range(duration):
                (
                    next_obs,
                    reward,
                    done,
                    truncated,
                    info
                ) = N.step_checked(
                    env,
                    action
                )

                gamma = (
                    policy.agent.gamma
                    if policy is not None
                    else 0.99
                )

                reward_total += (
                    gamma ** j
                ) * float(reward)

                actual += 1
                steps += 1

                if training:
                    TRAIN_COUNTER["frames"] += 1

                contact |= bool(
                    info.get("collision", False)
                )

                reached |= bool(
                    info.get("reach_goal", False)
                )

                if done or truncated or contact:
                    break

            if actual < 1:
                raise RuntimeError(
                    "NO_PHYSICS_STEP"
                )

            next_obs = np.asarray(
                next_obs,
                np.float32
            )

            next_feature = C.feat(
                next_obs,
                before,
                action,
                actual
            )

            next_depth = depth_frame(
                renderer,
                env
            )

            if training:
                replay.add(
                    policy.pack(feature, depth),
                    idx,
                    action,
                    reward_total,
                    policy.pack(
                        next_feature,
                        next_depth
                    ),
                    policy.agent.gamma ** actual,
                    float(done or contact)
                )

                if len(replay) >= 128:
                    while (
                        policy.agent.updates
                        <
                        TRAIN_COUNTER["frames"] // 16
                    ):
                        policy.update(
                            replay.sample(64)
                        )

            prev = before
            obs = next_obs

            prev_action = action.copy()
            prev_len = actual
            depth = next_depth

            if done or truncated or contact:
                break

        final_xy = np.asarray(
            env.data.qpos[:2],
            np.float64
        )

        progress = float(
            np.linalg.norm(goal - start_xy)
            -
            np.linalg.norm(goal - final_xy)
        )

        return {
            "safe": int(
                reached and not contact
            ),
            "contact": int(contact),
            "steps": steps,
            "progress_m": round(
                progress, 6
            )
        }

    finally:
        if renderer is not None:
            renderer.close()

        if env is not None:
            env.close()


def evaluate(
    mod, prep, policy,
    seeds=(202879101, 202879102, 202879103)
):
    results = {}

    for variant in ("left", "right"):
        rows = []

        for seed in seeds:
            seed_all(seed + 7)

            rows.append(
                episode(
                    mod, prep,
                    policy,
                    variant,
                    seed
                )
            )

        results[variant] = {
            "n": len(rows),
            "safe": sum(
                r["safe"]
                for r in rows
            ),
            "contacts": sum(
                r["contact"]
                for r in rows
            ),
            "progress": round(
                sum(
                    r["progress_m"]
                    for r in rows
                ) / len(rows),
                6
            ),
            "rows": rows
        }

    return results


def safety_gate(candidate, baseline):
    """
    Development safety gate.
    No loss of safe arrivals, no additional contacts,
    and no more than 3 percent progress regression.

    This is an exploratory training admission test,
    not proof of statistically significant improvement.
    """
    for name in ("left", "right"):
        c = candidate[name]
        b = baseline[name]

        if c["contacts"] > b["contacts"]:
            return False

        if c["safe"] < b["safe"]:
            return False

        baseline_progress = float(b["progress"])
        current_progress = float(c["progress"])

        tolerance = max(
            0.005,
            0.03 * abs(baseline_progress)
        )

        if current_progress < baseline_progress - tolerance:
            return False

    return True

def train_chunk(
    mod, prep, policy,
    replay, episodes, budget
):
    start = TRAIN_COUNTER["frames"]

    while (
        TRAIN_COUNTER["frames"] - start
        <
        budget
    ):
        remaining = (
            budget
            - (
                TRAIN_COUNTER["frames"]
                - start
            )
        )

        variant = (
            "left"
            if episodes % 2 == 0
            else "right"
        )

        seed = 202871000 + episodes

        episode(
            mod,
            prep,
            policy,
            variant,
            seed,
            training=True,
            replay=replay,
            frame_cap=min(
                1100, remaining
            )
        )

        episodes += 1

        if episodes % 4 == 0:
            print(
                "G19_P1B_TRAIN_PROGRESS",
                "FRAMES=",
                TRAIN_COUNTER["frames"],
                "UPDATES=",
                policy.agent.updates,
                "REPLAY=",
                len(replay),
                flush=True
            )

    gained = (
        TRAIN_COUNTER["frames"] - start
    )

    if gained != budget:
        raise RuntimeError(
            "INCREMENTAL_FRAME_BUDGET_MISMATCH"
        )

    return episodes


def checkpoint(policy, replay, episodes):
    agent = policy.agent

    networks = {}

    for key in (
        "actor",
        "q1",
        "q2",
        "t1",
        "t2"
    ):
        networks[key] = {
            name: value.detach().cpu()
            for name, value
            in getattr(
                agent, key
            ).state_dict().items()
        }

    return {
        "format":
            "G19_P1B_MULTIMODAL_EPISODE_BOUNDARY_V1",

        "frames": int(
            TRAIN_COUNTER["frames"]
        ),

        "episodes": int(episodes),
        "net": networks,

        "vision": {
            name: value.detach().cpu()
            for name, value
            in policy.vision.state_dict().items()
        },

        "vision_target": {
            name: value.detach().cpu()
            for name, value
            in policy.vision_target.state_dict().items()
        },

        "ap": agent.ap.state_dict(),
        "cp": agent.cp.state_dict(),

        "replay": list(replay.data),

        "updates": int(agent.updates),

        "mean": agent.mean.detach().cpu(),
        "std": agent.std.detach().cpu(),

        "alpha_c": float(agent.alpha_c),
        "alpha_d": float(agent.alpha_d),

        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),

        "rng_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),

        "promotion": False,
        "phase_planner": False,

        "camera": CAM,
        "depth_shape": [2, 30, 40],

        "restart_semantics": "episode_boundary"
    }


def restore(policy, replay, data):
    if (
        data.get("format")
        != "G19_P1B_MULTIMODAL_EPISODE_BOUNDARY_V1"
        or data.get("promotion") is not False
        or data.get("depth_shape") != [2, 30, 40]
    ):
        raise RuntimeError(
            "RESUME_PROVENANCE_INVALID"
        )

    agent = policy.agent

    for name, weights in data["net"].items():
        getattr(
            agent, name
        ).load_state_dict(
            weights,
            strict=True
        )

    policy.vision.load_state_dict(
        data["vision"],
        strict=True
    )

    policy.vision_target.load_state_dict(
        data["vision_target"],
        strict=True
    )

    agent.ap.load_state_dict(
        data["ap"]
    )

    agent.cp.load_state_dict(
        data["cp"]
    )

    for optimizer in (
        agent.ap,
        agent.cp
    ):
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(
                        agent.device
                    )

    replay.data = deque(
        data["replay"],
        maxlen=MAXMEM
    )

    agent.updates = int(
        data["updates"]
    )

    agent.alpha_c = float(
        data["alpha_c"]
    )

    agent.alpha_d = float(
        data["alpha_d"]
    )

    agent.mean = data["mean"].to(
        agent.device
    )

    agent.std = data["std"].to(
        agent.device
    )

    random.setstate(
        data["rng_python"]
    )

    np.random.set_state(
        data["rng_numpy"]
    )

    torch.set_rng_state(
        data["rng_torch"]
    )

    if (
        torch.cuda.is_available()
        and data["rng_cuda"]
    ):
        torch.cuda.set_rng_state_all(
            data["rng_cuda"]
        )

    TRAIN_COUNTER["frames"] = int(
        data["frames"]
    )

    return int(
        data["episodes"]
    )


def pilot(mod, prep, hashes, out):
    global FROZEN_AGENT

    (
        g5b, H, C, N,
        proof, frozen, snap,
        d1, protected, device, ck
    ) = prep

    nearfield_gate(mod, prep)

    FROZEN_AGENT = mod.agent_new(
        g5b, H, C, ck,
        snap, d1, device, 202710700
    )

    FROZEN_AGENT.actor.eval()

    baseline = evaluate(
        mod, prep, None
    )

    print(
        "G19_P1B_FROZEN_DEV",
        json.dumps(baseline),
        flush=True
    )

    policy = new_model(
        mod, prep
    )

    replay = C.Replay(MAXMEM)

    # 初始化时融合网络必须完全保持冻结策略。
    x = np.zeros(
        FID, np.float32
    )

    image = np.zeros(
        DIM, np.float32
    )

    with torch.no_grad():
        fused = policy.fused_state(
            policy.pack(x, image)
        )

        original = policy.agent.state(
            x
        ).reshape(1, -1)

        delta = float(
            (
                policy.agent.actor.forward(fused)[0]
                -
                policy.agent.actor.forward(original)[0]
            ).abs().max().item()
        )

    if delta > 1e-7:
        raise RuntimeError(
            "ZERO_INIT_NOT_IDENTICAL"
        )

    episodes = train_chunk(
        mod, prep,
        policy,
        replay,
        0,
        3200
    )

    if not all((
        policy.agent.updates >= 10,
        policy.grad_actor > 0,
        policy.grad_critic > 0,
        policy.grad_encoder > 0
    )):
        raise RuntimeError(
            "REAL_SAC_VISION_GRADIENT_GATE_FAILED"
        )

    candidate = evaluate(
        mod, prep, policy
    )

    # 验证训练后动作确实依赖深度输入。
    import mujoco as mj

    (
        env, obs, goal,
        renderer, depth, xy
    ) = create_nearfield(
        mj,
        N,
        mod.PR,
        "medium",
        202879101,
        "left"
    )

    try:
        feature = C.feat(
            obs,
            obs,
            np.zeros(2, np.float32),
            1
        )

        with torch.no_grad():
            on = policy.fused_state(
                policy.pack(
                    feature, depth
                )
            )

            off = policy.fused_state(
                policy.pack(
                    feature,
                    np.zeros_like(depth)
                )
            )

            visual_action_delta = float(
                (
                    policy.agent.actor.forward(on)[0]
                    -
                    policy.agent.actor.forward(off)[0]
                ).abs().max().item()
            )

    finally:
        renderer.close()
        env.close()

    allowed = (
        safety_gate(
            candidate, baseline
        )
        and visual_action_delta > 1e-8
        and bool(
            getattr(
                create_nearfield,
                "_native_full_obs_verified",
                False
            )
        )
    )

    # 先做实际检查点恢复，再决定是否准入。
    save_pt(
        out / "pilot_resume.pt",
        checkpoint(
            policy,
            replay,
            episodes
        )
    )

    restored = new_model(
        mod, prep
    )

    restored_replay = C.Replay(
        MAXMEM
    )

    restored_episodes = restore(
        restored,
        restored_replay,
        torch.load(
            out / "pilot_resume.pt",
            map_location="cpu",
            weights_only=False
        )
    )

    check_input = (
        np.ones(
            FID + DIM,
            np.float32
        ) * 0.2
    )

    with torch.no_grad():
        restore_error = float(
            (
                policy.fused_state(check_input)
                -
                restored.fused_state(check_input)
            ).abs().max().item()
        )

    if not all((
        restore_error <= 1e-7,
        restored.agent.updates
        == policy.agent.updates,
        restored_episodes == episodes,
        len(restored_replay) == len(replay)
    )):
        raise RuntimeError(
            "CHECKPOINT_RELOAD_FAILED"
        )

    check_sources(
        mod, prep, hashes
    )

    report = {
        "gate": (
            "G19_P1B_READY_FOR_GUARDED_NIGHT"
            if allowed
            else "G19_P1B_SAFETY_BLOCKED"
        ),

        "night_launch": bool(allowed),
        "frames": TRAIN_COUNTER["frames"],

        "updates": policy.agent.updates,
        "episodes": episodes,
        "replay_count": len(replay),

        "vision_actor_gradient":
            policy.grad_actor,

        "vision_critic_gradient":
            policy.grad_critic,

        "vision_encoder_gradient":
            policy.grad_encoder,

        "zero_init_max_delta": delta,

        "trained_visual_action_max_delta":
            visual_action_delta,

        "checkpoint_roundtrip_max_delta":
            restore_error,

        "frozen_dev": baseline,
        "fusion_dev": candidate,

        "source_hashes": hashes,

        "model_promotion": False,

        "deadline_host":
            "2026-10-19T08:30:00+08:00"
    }

    write_json(
        out / "pilot_gate.json",
        report
    )

    print(
        "G19_P1B_PILOT_FINAL",
        json.dumps(
            report,
            ensure_ascii=False
        ),
        flush=True
    )


def night(mod, prep, hashes, out):
    gate = json.loads(
        (
            out / "pilot_gate.json"
        ).read_text()
    )

    if (
        gate.get("gate")
        != "G19_P1B_READY_FOR_GUARDED_NIGHT"
        or gate.get("night_launch") is not True
        or gate.get("source_hashes") != hashes
    ):
        raise RuntimeError(
            "PILOT_SAFETY_GATE_MISSING"
        )

    if time.time() > CUT - 1500:
        raise RuntimeError(
            "PAST_NIGHT_CUTOFF"
        )

    C = prep[2]

    policy = new_model(
        mod, prep
    )

    replay = C.Replay(
        MAXMEM
    )

    episodes = restore(
        policy,
        replay,
        torch.load(
            out / "pilot_resume.pt",
            map_location="cpu",
            weights_only=False
        )
    )

    best_safe = sum(
        gate["fusion_dev"][name]["safe"]
        for name in ("left", "right")
    )

    unsafe_rounds = 0
    no_gain_rounds = 0
    round_id = 0
    reason = "DEADLINE"

    print(
        "G19_P1B_NIGHT_START=PASS",
        "FRAMES=", TRAIN_COUNTER["frames"],
        "EPISODES=", episodes,
        flush=True
    )

    while (
        time.time() < CUT - 600
        and TRAIN_COUNTER["frames"] < 150000
        and round_id < 22
    ):
        episodes = train_chunk(
            mod, prep,
            policy,
            replay,
            episodes,
            6000
        )

        round_id += 1

        # 每轮先保存完整恢复点，再评估。
        save_pt(
            out / "night_resume.pt",
            checkpoint(
                policy,
                replay,
                episodes
            )
        )

        current = evaluate(
            mod, prep, policy
        )

        safe = safety_gate(
            current,
            gate["frozen_dev"]
        )

        score = sum(
            current[name]["safe"]
            for name in ("left", "right")
        )

        if safe and score > best_safe:
            best_safe = score
            no_gain_rounds = 0

            save_pt(
                out / "best_dev_actor_vision_research_only.pt",
                {
                    "actor":
                        policy.agent.actor.state_dict(),

                    "vision":
                        policy.vision.state_dict(),

                    "frames":
                        TRAIN_COUNTER["frames"],

                    "dev": current,
                    "promotion": False
                }
            )

        else:
            no_gain_rounds += 1

        if safe:
            unsafe_rounds = 0
        else:
            unsafe_rounds += 1

        status = {
            "round": round_id,

            "frames":
                TRAIN_COUNTER["frames"],

            "updates":
                policy.agent.updates,

            "episodes": episodes,
            "safe_gate": safe,

            "dev": current,
            "frozen_dev": gate["frozen_dev"],

            "best_safe_count": best_safe,

            "consecutive_safety_failures":
                unsafe_rounds,

            "no_improvement_rounds":
                no_gain_rounds,

            "checkpoint": str(
                out / "night_resume.pt"
            ),

            "promotion": False
        }

        write_json(
            out / "night_status.json",
            status
        )

        check_sources(
            mod, prep, hashes
        )

        print(
            "G19_P1B_NIGHT_ROUND",
            json.dumps(
                status,
                ensure_ascii=False
            ),
            flush=True
        )

        if unsafe_rounds >= 1:
            reason = "SAFETY_REGRESSION"
            break

        if no_gain_rounds >= 3:
            reason = "NO_CONFIRMED_PROGRESS"
            break

    if TRAIN_COUNTER["frames"] >= 150000:
        reason = "FRAME_CAP"

    if round_id >= 22:
        reason = "ROUND_CAP"

    write_json(
        out / "night_final.json",
        {
            "gate": "G19_P1B_NIGHT_FINISHED",
            "reason": reason,

            "frames":
                TRAIN_COUNTER["frames"],

            "rounds": round_id,
            "promotion": False,
            "source_unchanged": True
        }
    )

    print(
        "G19_P1B_NIGHT_FINAL",
        reason,
        TRAIN_COUNTER["frames"],
        flush=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("pilot", "night"),
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    args = parser.parse_args()
    out = Path(args.out).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.set_num_threads(1)

    host_date = datetime.now(
        timezone(
            timedelta(hours=8)
        )
    ).date().isoformat()

    if host_date not in (
        "2026-10-18",
        "2026-10-19"
    ):
        raise RuntimeError(
            "EXPERIMENT_DATE_EXPIRED"
        )

    mod, prep, hashes = source_gate()

    print(
        "G19_P1B_REAL_MULTIMODAL_SAC_START",
        args.mode,
        flush=True
    )

    if args.mode == "pilot":
        pilot(
            mod, prep, hashes, out
        )

    else:
        night(
            mod, prep, hashes, out
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "G19_P1B_BLOCK",
            repr(exc),
            flush=True
        )

        traceback.print_exc()

        sys.exit(2)
