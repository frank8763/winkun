import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import torch


HOME = Path.home()

V3 = HOME / (
    "mkmini_stage_g19_p1b_v3_night_"
    "20261018_220337_273191331/p1b.py"
)

EXPECTED_V3 = (
    "df2df255e93842759bcbf3b2a233e965"
    "438989955b60a21cdf6ae3dd9eacf6d6"
)

CUTOFF = datetime(
    2026, 10, 19, 8, 30,
    tzinfo=timezone(timedelta(hours=8))
).timestamp()

DEV = (2028993101, 2028993102)

SCENES = (
    "open",
    "soft_left",
    "soft_right",
    "left",
    "right"
)

ARMS = {
    "lidar_control": {
        "actor_lr": 2e-7,
        "vision_lr": 0.0,
        "critic_lr": 5e-5,
        "course": (
            "open", "soft_left",
            "soft_right", "open"
        )
    },

    "depth_conservative": {
        "actor_lr": 3e-7,
        "vision_lr": 4e-7,
        "critic_lr": 5e-5,
        "course": (
            "open", "soft_left",
            "soft_right", "open"
        )
    },

    "depth_exploration": {
        "actor_lr": 8e-7,
        "vision_lr": 1e-6,
        "critic_lr": 5e-5,
        "course": (
            "open", "soft_left",
            "soft_right", "left", "right"
        )
    }
}


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ) + "\n"
    )

    os.replace(temp, path)


def save_pt(path, data):
    path = Path(path)
    temp = path.with_suffix(".partial")

    torch.save(data, temp)
    os.replace(temp, path)


def load_native():
    if (
        not V3.is_file()
        or sha(V3) != EXPECTED_V3
    ):
        raise RuntimeError(
            "V3_FROZEN_SOURCE_HASH_CHANGED"
        )

    spec = importlib.util.spec_from_file_location(
        "g19_p2_native", V3
    )

    v = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v
    spec.loader.exec_module(v)

    return v


def install_scenes(v):
    import mujoco as mj

    original = v.create_nearfield

    def create(
        mjarg, N, project,
        task, seed, variant
    ):
        if variant == "open":
            env = None
            renderer = None

            try:
                env, obs, info, goal = N.create(
                    project,
                    "reference",
                    task,
                    int(seed)
                )

                renderer = mj.Renderer(
                    env.model,
                    height=120,
                    width=160
                )

                depth = v.depth_frame(
                    renderer, env
                )

                return (
                    env,
                    np.asarray(obs, np.float32),
                    np.asarray(goal, float).reshape(2),
                    renderer,
                    depth,
                    None
                )

            except Exception:
                if renderer is not None:
                    renderer.close()

                if env is not None:
                    env.close()

                raise

        if variant not in (
            "left", "right",
            "soft_left", "soft_right"
        ):
            raise RuntimeError(
                "INVALID_CURRICULUM_VARIANT"
            )

        side = variant.rsplit("_", 1)[-1]

        env, obs, goal, renderer, depth, pos = (
            original(
                mjarg, N, project,
                task, seed, side
            )
        )

        if variant.startswith("soft_"):
            try:
                geom = mj.mj_name2id(
                    env.model,
                    mj.mjtObj.mjOBJ_GEOM,
                    v.TARGET
                )

                if geom < 0:
                    raise RuntimeError(
                        "SOFT_OBSTACLE_MISSING"
                    )

                env.model.geom_size[
                    geom, 0:2
                ] = [0.085, 0.095]

                mj.mj_forward(
                    env.model,
                    env.data
                )

                obs = np.asarray(
                    env._observation(
                        update_history=False
                    ),
                    np.float32
                )

                if (
                    obs.shape != (112,)
                    or not np.allclose(
                        obs[:48],
                        env._lidar48(),
                        atol=1e-5,
                        rtol=0
                    )
                ):
                    raise RuntimeError(
                        "SOFT_FULL_OBSERVATION_DESYNC"
                    )

                depth = v.depth_frame(
                    renderer, env
                )

            except Exception:
                renderer.close()
                env.close()
                raise

        return (
            env, obs, goal,
            renderer, depth, pos
        )

    v.create_nearfield = create


def new_policy(v, mod, prep, cfg):
    p = v.new_model(mod, prep)
    agent = p.agent

    # Actor、Critic、Vision 使用独立优化器。
    agent.ap = torch.optim.Adam(
        agent.actor.parameters(),
        lr=cfg["actor_lr"]
    )

    agent.cp = torch.optim.Adam(
        list(agent.q1.parameters())
        + list(agent.q2.parameters()),
        lr=cfg["critic_lr"]
    )

    p.vp = (
        torch.optim.Adam(
            p.vision.parameters(),
            lr=cfg["vision_lr"]
        )
        if cfg["vision_lr"] > 0
        else None
    )

    if p.vp is None:
        for parameter in p.vision.parameters():
            parameter.requires_grad_(False)

        for parameter in p.vision_target.parameters():
            parameter.requires_grad_(False)

    p.grad_actor = 0.0
    p.grad_critic = 0.0
    p.grad_encoder = 0.0

    return p


def sac_update(v, p, batch):
    sx, ix, act, ret, nx, discount, terminal = batch

    ag = p.agent
    device = p.device

    state = p.fused_state(sx)

    action = torch.as_tensor(
        act,
        dtype=torch.float32,
        device=device
    )

    index = torch.as_tensor(
        ix,
        dtype=torch.long,
        device=device
    )

    reward = torch.as_tensor(
        ret,
        dtype=torch.float32,
        device=device
    )

    gamma = torch.as_tensor(
        discount,
        dtype=torch.float32,
        device=device
    )

    done = torch.as_tensor(
        terminal,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        next_state = p.fused_state(
            nx, target=True
        )

        target = (
            reward
            + gamma * (1.0 - done)
            * ag._value(
                next_state,
                (ag.t1, ag.t2)
            )
        )

    loss_q = (
        (
            ag.q1(state, action, index)
            - target
        ).square().mean()
        +
        (
            ag.q2(state, action, index)
            - target
        ).square().mean()
    )

    ag.cp.zero_grad()

    if p.vp is not None:
        p.vp.zero_grad()

    loss_q.backward()

    # 保存 Critic 产生的视觉梯度。
    critic_vision_grad = (
        [
            None if x.grad is None
            else x.grad.detach().clone()
            for x in p.vision.parameters()
        ]
        if p.vp is not None
        else []
    )

    if p.vp is not None:
        p.grad_critic = max(
            p.grad_critic,
            sum(
                float(g.abs().sum().item())
                for g in critic_vision_grad
                if g is not None
            )
        )

    torch.nn.utils.clip_grad_norm_(
        list(ag.q1.parameters())
        + list(ag.q2.parameters()),
        5.0
    )

    ag.cp.step()

    if p.vp is not None:
        p.vp.zero_grad()

    for critic in (ag.q1, ag.q2):
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    try:
        state2 = p.fused_state(sx)

        actions, logpa, probs, logpd = (
            ag.actor.sample_all(state2)
        )

        terms = []

        for k in range(len(p.C.DURATIONS)):
            duration_id = torch.full(
                (state2.shape[0],),
                k,
                device=device,
                dtype=torch.long
            )

            qmin = torch.minimum(
                ag.q1(
                    state2,
                    actions[:, k],
                    duration_id
                ),
                ag.q2(
                    state2,
                    actions[:, k],
                    duration_id
                )
            )

            terms.append(
                ag.alpha_c * logpa[:, k]
                + ag.alpha_d * logpd[:, k]
                - qmin
            )

        loss_actor = (
            probs
            * torch.stack(
                terms, dim=-1
            )
        ).sum(-1).mean()

        ag.ap.zero_grad()
        loss_actor.backward()

        if p.vp is not None:
            actor_grad = [
                None if x.grad is None
                else x.grad.detach().clone()
                for x in p.vision.parameters()
            ]

            p.grad_actor = max(
                p.grad_actor,
                sum(
                    float(g.abs().sum().item())
                    for g in actor_grad
                    if g is not None
                )
            )

            first = (
                p.vision.encoder[0]
                .weight.grad
            )

            if first is not None:
                p.grad_encoder = max(
                    p.grad_encoder,
                    float(
                        first.abs().sum().item()
                    )
                )

            # 将两种损失的视觉梯度合并。
            # 视觉参数仅执行一次 optimizer.step。
            for parameter, grad_q in zip(
                p.vision.parameters(),
                critic_vision_grad
            ):
                if grad_q is not None:
                    parameter.grad = (
                        grad_q
                        if parameter.grad is None
                        else parameter.grad + grad_q
                    )

        torch.nn.utils.clip_grad_norm_(
            ag.actor.parameters(), 5.0
        )

        if p.vp is not None:
            torch.nn.utils.clip_grad_norm_(
                p.vision.parameters(), 5.0
            )

        ag.ap.step()

        if p.vp is not None:
            p.vp.step()

    finally:
        for critic in (ag.q1, ag.q2):
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

    with torch.no_grad():
        for source, target_net in (
            (ag.q1, ag.t1),
            (ag.q2, ag.t2)
        ):
            for a, b in zip(
                source.parameters(),
                target_net.parameters()
            ):
                b.lerp_(a, ag.tau)

        for a, b in zip(
            p.vision.parameters(),
            p.vision_target.parameters()
        ):
            b.lerp_(a, ag.tau)

    ag.updates += 1

    if not (
        math.isfinite(loss_q.item())
        and math.isfinite(loss_actor.item())
    ):
        raise RuntimeError(
            "NONFINITE_MULTIMODAL_SAC_LOSS"
        )

    return (
        float(loss_q.item()),
        float(loss_actor.item())
    )


def train(
    v, mod, prep, policy,
    replay, episodes,
    budget, cfg, deadline
):
    start = v.TRAIN_COUNTER["frames"]

    contacts = 0
    goals = 0

    while (
        v.TRAIN_COUNTER["frames"] - start
        < budget
    ):
        if time.time() > deadline - 180:
            break

        remaining = (
            budget
            - (
                v.TRAIN_COUNTER["frames"]
                - start
            )
        )

        variant = cfg["course"][
            episodes % len(cfg["course"])
        ]

        result = v.episode(
            mod,
            prep,
            policy,
            variant,
            2028820000 + episodes,
            training=True,
            replay=replay,
            frame_cap=min(1100, remaining)
        )

        contacts += result["contact"]
        goals += result["safe"]
        episodes += 1

    gained = (
        v.TRAIN_COUNTER["frames"] - start
    )

    if gained <= 0:
        raise RuntimeError(
            "NO_TRAINING_FRAMES"
        )

    return episodes, gained, contacts, goals


def evaluate(v, mod, prep, policy):
    results = {}

    for variant in SCENES:
        rows = []

        for seed in DEV:
            v.seed_all(seed + 7)

            rows.append(
                v.episode(
                    mod,
                    prep,
                    policy,
                    variant,
                    seed
                )
            )

        results[variant] = {
            "n": len(rows),
            "safe": sum(
                r["safe"] for r in rows
            ),
            "contacts": sum(
                r["contact"] for r in rows
            ),
            "progress": round(
                sum(
                    r["progress_m"]
                    for r in rows
                ) / len(rows),
                6
            )
        }

    return results


def safe(candidate, baseline):
    for variant in SCENES:
        c = candidate[variant]
        b = baseline[variant]

        if c["contacts"] > b["contacts"]:
            return False

        if c["safe"] < b["safe"]:
            return False

        tolerance = max(
            0.009,
            0.05 * abs(b["progress"])
        )

        if (
            c["progress"]
            < b["progress"] - tolerance
        ):
            return False

    return True


def ranking(stats):
    return (
        sum(x["safe"] for x in stats.values()),
        -sum(
            x["contacts"]
            for x in stats.values()
        ),
        round(
            sum(
                x["progress"]
                for x in stats.values()
            ),
            6
        )
    )


def snapshot(
    v, p, replay, episodes,
    arm, cfg, hashes
):
    data = v.checkpoint(
        p, replay, episodes
    )

    data["branch"] = arm
    data["branch_config"] = cfg
    data["source_hashes"] = hashes

    data["vp"] = (
        p.vp.state_dict()
        if p.vp is not None
        else None
    )

    data["gate_scope"] = (
        "RESEARCH_ONLY_NO_PROMOTION"
    )

    return data


def resume(
    v, mod, prep,
    arm, cfg, replay,
    path, hashes
):
    p = new_policy(
        v, mod, prep, cfg
    )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if (
        data.get("branch") != arm
        or data.get("branch_config") != cfg
        or data.get("source_hashes") != hashes
    ):
        raise RuntimeError(
            "RESUME_PROVENANCE_INVALID"
        )

    episodes = v.restore(
        p, replay, data
    )

    if p.vp is not None:
        if data.get("vp") is None:
            raise RuntimeError(
                "VISION_OPTIMIZER_STATE_MISSING"
            )

        p.vp.load_state_dict(
            data["vp"]
        )

        for values in p.vp.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(
                        p.device
                    )

    elif data.get("vp") is not None:
        raise RuntimeError(
            "LIDAR_BRANCH_VISION_OPTIMIZER_INVALID"
        )

    return p, episodes


def checkpoint_roundtrip(
    v, mod, prep, p,
    replay, episodes,
    arm, cfg, hashes, path
):
    save_pt(
        path,
        snapshot(
            v, p, replay,
            episodes, arm, cfg, hashes
        )
    )

    recovered, loaded_episodes = resume(
        v, mod, prep,
        arm, cfg,
        prep[2].Replay(v.MAXMEM),
        path,
        hashes
    )

    state = np.full(
        v.FID + v.DIM,
        0.2,
        dtype=np.float32
    )

    with torch.no_grad():
        error = float(
            (
                p.fused_state(state)
                - recovered.fused_state(state)
            ).abs().max().item()
        )

    if (
        error > 1e-7
        or loaded_episodes != episodes
        or recovered.agent.updates
        != p.agent.updates
    ):
        raise RuntimeError(
            "CHECKPOINT_ROUNDTRIP_FAILED"
        )

    return error


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

    if time.time() >= CUTOFF - 600:
        raise RuntimeError(
            "NIGHT_DEADLINE_EXPIRED"
        )

    v = load_native()
    v.MAXMEM = 600

    mod, prep, hashes = v.source_gate()

    install_scenes(v)

    (
        g5b, H, C, N,
        proof, frozen, snap,
        d1, protected, device, ck
    ) = prep

    v.FROZEN_AGENT = mod.agent_new(
        g5b, H, C, ck,
        snap, d1,
        device,
        202710700
    )

    v.FROZEN_AGENT.actor.eval()

    for parameter in (
        v.FROZEN_AGENT.actor.parameters()
    ):
        parameter.requires_grad_(False)

    source_hashes = dict(hashes)
    source_hashes[str(V3)] = sha(V3)

    if args.mode == "pilot":
        print(
            "G19_P2_FOREGROUND_START=TRUE",
            flush=True
        )

        v.nearfield_gate(mod, prep)

        baseline = evaluate(
            v, mod, prep, None
        )

        print(
            "G19_P2_FROZEN_DEV",
            json.dumps(baseline),
            flush=True
        )

        accepted = []
        results = {}

        for arm, cfg in ARMS.items():
            directory = out / arm

            directory.mkdir(
                exist_ok=True
            )

            v.seed_all(
                2026101800
                + len(results) * 13
            )

            v.TRAIN_COUNTER["frames"] = 0

            policy = new_policy(
                v, mod, prep, cfg
            )

            policy.update = (
                lambda batch, p=policy:
                sac_update(v, p, batch)
            )

            replay = C.Replay(v.MAXMEM)

            (
                episodes,
                gained,
                contacts,
                goals
            ) = train(
                v, mod, prep,
                policy, replay,
                0, 3200,
                cfg,
                time.time() + 550
            )

            if (
                gained != 3200
                or policy.agent.updates < 10
            ):
                raise RuntimeError(
                    "PILOT_INCOMPLETE_" + arm
                )

            if cfg["vision_lr"] > 0:
                if not all((
                    policy.grad_actor > 0,
                    policy.grad_critic > 0,
                    policy.grad_encoder > 0
                )):
                    raise RuntimeError(
                        "VISION_GRADIENT_FAILED_" + arm
                    )

            development = evaluate(
                v, mod, prep, policy
            )

            recovery_error = checkpoint_roundtrip(
                v, mod, prep,
                policy, replay,
                episodes,
                arm, cfg,
                source_hashes,
                directory / "resume.pt"
            )

            passed = safe(
                development, baseline
            )

            results[arm] = {
                "safe_gate": passed,
                "frames": gained,
                "updates": policy.agent.updates,
                "train_contacts": contacts,
                "train_goals": goals,
                "vision_grad_actor":
                    policy.grad_actor,
                "vision_grad_critic":
                    policy.grad_critic,
                "vision_grad_encoder":
                    policy.grad_encoder,
                "resume_error":
                    recovery_error,
                "dev": development
            }

            print(
                "G19_P2_BRANCH_PILOT",
                json.dumps({
                    "arm": arm,
                    **results[arm]
                }),
                flush=True
            )

            if passed:
                accepted.append(arm)

            else:
                (
                    directory / "resume.pt"
                ).unlink(missing_ok=True)

            v.check_sources(
                mod, prep, hashes
            )

        gate = {
            "gate": (
                "G19_P2_MULTIARM_READY"
                if accepted
                else "G19_P2_MULTIARM_BLOCKED"
            ),
            "accepted": accepted,
            "baseline": baseline,
            "arms": results,
            "sources": source_hashes,
            "promotion": False,
            "deadline":
                "2026-10-19T08:30:00+08:00"
        }

        write_json(
            out / "pilot_gate.json",
            gate
        )

        print(
            "G19_P2_PILOT_GATE=",
            gate["gate"],
            "ACCEPTED=",
            ",".join(accepted),
            flush=True
        )

    else:
        gate = json.loads(
            (
                out / "pilot_gate.json"
            ).read_text()
        )

        if (
            gate["gate"]
            != "G19_P2_MULTIARM_READY"
            or not gate["accepted"]
            or gate["sources"] != source_hashes
        ):
            raise RuntimeError(
                "PILOT_ADMISSION_MISSING"
            )

        # 后台启动前验证全部支线恢复点。
        for arm in gate["accepted"]:
            temporary_replay = (
                C.Replay(v.MAXMEM)
            )

            resume(
                v, mod, prep,
                arm,
                ARMS[arm],
                temporary_replay,
                out / arm / "resume.pt",
                source_hashes
            )

        print(
            "G19_P2_NIGHT_START=PASS",
            "ARMS=" + ",".join(
                gate["accepted"]
            ),
            flush=True
        )

        if (
            out / "night_final.json"
        ).exists():
            raise RuntimeError(
                "NIGHT_ALREADY_COMPLETED"
            )

        if (
            out / "night_status.json"
        ).exists():
            previous = json.loads(
                (
                    out / "night_status.json"
                ).read_text()
            )

            status = previous["status"]
            global_frames = int(
                previous["total_night_frames"]
            )

            best = tuple(
                previous["best_score"]
            )

            print(
                "G19_P2_NIGHT_RESUMED_FROM_STATUS=TRUE",
                flush=True
            )

        else:
            status = {
                arm: {
                    "active": True,
                    "rounds": 0,
                    "frames": 3200,
                    "reason": "RUNNING"
                }
                for arm in gate["accepted"]
            }

            best = ranking(
                gate["baseline"]
            )

            global_frames = 0

        while (
            time.time() < CUTOFF - 600
            and global_frames < 900000
            and any(
                row["active"]
                for row in status.values()
            )
        ):
            for arm in gate["accepted"]:
                row = status[arm]

                if (
                    not row["active"]
                    or time.time() >= CUTOFF - 600
                    or global_frames >= 900000
                ):
                    continue

                cfg = ARMS[arm]
                directory = out / arm
                replay = C.Replay(v.MAXMEM)

                policy, episodes = resume(
                    v, mod, prep,
                    arm, cfg, replay,
                    directory / "resume.pt",
                    source_hashes
                )

                policy.update = (
                    lambda batch, p=policy:
                    sac_update(v, p, batch)
                )

                (
                    episodes,
                    gained,
                    contacts,
                    goals
                ) = train(
                    v, mod, prep,
                    policy, replay,
                    episodes,
                    min(
                        4000,
                        900000 - global_frames
                    ),
                    cfg,
                    CUTOFF
                )

                global_frames += gained

                row["frames"] += gained
                row["rounds"] += 1

                development = evaluate(
                    v, mod, prep, policy
                )

                passed = safe(
                    development,
                    gate["baseline"]
                )

                improved = (
                    passed
                    and ranking(development) > best
                )

                if improved:
                    best = ranking(
                        development
                    )

                    save_pt(
                        out / "best_research_only.pt",
                        snapshot(
                            v, policy, replay,
                            episodes,
                            arm, cfg,
                            source_hashes
                        )
                    )

                    write_json(
                        out / "best_research.json",
                        {
                            "arm": arm,
                            "score": list(best),
                            "dev": development,
                            "promotion": False,
                            "independent_confirmatory_test":
                                False
                        }
                    )

                if passed:
                    save_pt(
                        directory / "resume.pt",
                        snapshot(
                            v, policy, replay,
                            episodes,
                            arm, cfg,
                            source_hashes
                        )
                    )

                else:
                    row["active"] = False
                    row["reason"] = (
                        "DEV_REGRESSION"
                    )

                if contacts > 5:
                    row["active"] = False
                    row["reason"] = (
                        "TRAIN_CONTACT_BUDGET"
                    )

                row.update({
                    "dev": development,
                    "safe_gate": passed,
                    "train_contacts": contacts,
                    "train_goals": goals,
                    "best_research_improved":
                        improved
                })

                write_json(
                    out / "night_status.json",
                    {
                        "status": status,
                        "total_night_frames":
                            global_frames,
                        "best_score":
                            list(best),
                        "promotion": False,
                        "deadline":
                            "2026-10-19T08:30:00+08:00"
                    }
                )

                print(
                    "G19_P2_NIGHT_ROUND",
                    json.dumps({
                        "arm": arm,
                        "frames": row["frames"],
                        "safe_gate": passed,
                        "contacts": contacts,
                        "dev": development,
                        "improved": improved
                    }),
                    flush=True
                )

                v.check_sources(
                    mod, prep, hashes
                )

        write_json(
            out / "night_final.json",
            {
                "gate":
                    "G19_P2_NIGHT_FINISHED",
                "status": status,
                "total_night_frames":
                    global_frames,
                "best_score":
                    list(best),
                "promotion": False
            }
        )

        print(
            "G19_P2_NIGHT_FINISHED",
            global_frames,
            flush=True
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            "G19_P2_BLOCK",
            repr(exc),
            flush=True
        )

        traceback.print_exc()
        sys.exit(2)
