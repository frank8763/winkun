import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim=112, action_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim=112, action_dim=2):
        super().__init__()

        def network():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 800),
                nn.ReLU(),
                nn.Linear(800, 600),
                nn.ReLU(),
                nn.Linear(600, 1)
            )

        self.q1 = network()
        self.q2 = network()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)

        return self.q1(x), self.q2(x)


class TD3:
    def __init__(
        self,
        state_dim=112,
        action_dim=2,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    ):
        self.device = torch.device(device)

        self.actor = Actor(
            state_dim, action_dim
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(
            state_dim, action_dim
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-3
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=1e-3
        )

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_updates = 0

    @torch.no_grad()
    def act(self, observation):
        state = torch.as_tensor(
            np.asarray(
                observation,
                dtype=np.float32
            ).reshape(1, -1),
            device=self.device
        )

        action = self.actor(state)

        return action.cpu().numpy()[0]

    def update(self, batch):
        state, action, reward, next_state, done = [
            torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.device
            )
            for x in batch
        ]

        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)

        if done.ndim == 1:
            done = done.unsqueeze(-1)

        with torch.no_grad():

            noise = (
                torch.randn_like(action)
                * self.policy_noise
            ).clamp(
                -self.noise_clip,
                self.noise_clip
            )

            next_action = (
                self.actor_target(next_state)
                + noise
            ).clamp(-1.0, 1.0)

            q1_target, q2_target = (
                self.critic_target(
                    next_state,
                    next_action
                )
            )

            target_q = reward + (
                1.0 - done
            ) * self.gamma * torch.min(
                q1_target,
                q2_target
            )

        q1, q2 = self.critic(
            state,
            action
        )

        critic_loss = (
            F.mse_loss(q1, target_q)
            + F.mse_loss(q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_updates += 1

        actor_loss_value = None

        if (
            self.total_updates
            % self.policy_delay == 0
        ):

            actor_loss = -self.critic.q1(
                torch.cat(
                    (
                        state,
                        self.actor(state)
                    ),
                    dim=-1
                )
            ).mean()

            self.actor_optimizer.zero_grad()
            td3_t9_actions = self.actor(state)
            actor_loss = actor_loss + 0.03000000 * (
                td3_t9_actions[:, 1].square().mean()
                + 0.5 * td3_t9_actions[:, 0].square().mean()
            )
            t12_left, t12_right = self.t12_pairs
            t12_l = self.actor(t12_left)[:, 1]
            t12_r = self.actor(t12_right)[:, 1]
            t12_gap = t12_r - t12_l
            t12_goal_loss = torch.relu(0.4 - t12_gap).square().mean()
            actor_loss = actor_loss + 0.05 * t12_goal_loss
            # T46: only preserve actions from two verified
            # 3m success trajectories. Training-time only.
            if not hasattr(self, "_t46_success_obs"):
                import hashlib as _bc_hashlib
                import numpy as _bc_numpy
            
                _bc_path = '/home/linghu/mkmini_stage_td3_t42_verified_3m_replay_20261027_v1/verified_3m_success.npz'
                _bc_hash = _bc_hashlib.sha256()
            
                with open(_bc_path, "rb") as _bc_file:
                    for _bc_chunk in iter(
                        lambda: _bc_file.read(1048576), b""
                    ):
                        _bc_hash.update(_bc_chunk)
            
                if _bc_hash.hexdigest() != '743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b':
                    raise RuntimeError("T46_SUCCESS_ARCHIVE_CHANGED")
            
                with _bc_numpy.load(_bc_path) as _bc_npz:
                    _bc_obs = _bc_numpy.asarray(
                        _bc_npz["states"],
                        dtype=_bc_numpy.float32
                    ).copy()
            
                    _bc_actions = _bc_numpy.asarray(
                        _bc_npz["raw_actions"],
                        dtype=_bc_numpy.float32
                    ).copy()
            
                    _bc_offsets = (
                        _bc_npz["episode_offsets"].tolist()
                    )
            
                if (
                    _bc_obs.shape != (20319, 112)
                    or _bc_actions.shape != (20319, 2)
                    or _bc_offsets != [0, 10612, 20319]
                    or not _bc_numpy.isfinite(_bc_obs).all()
                    or not _bc_numpy.isfinite(_bc_actions).all()
                ):
                    raise RuntimeError("T46_SUCCESS_SHAPE_INVALID")
            
                self._t46_success_obs = torch.as_tensor(
                    _bc_obs,
                    dtype=state.dtype,
                    device=state.device
                )
            
                self._t46_success_actions = torch.as_tensor(
                    _bc_actions,
                    dtype=state.dtype,
                    device=state.device
                )
            
                print(
                    "T46_SUCCESS_ARCHIVE_LOADED=PASS",
                    "LEFT=10612",
                    "RIGHT=9707",
                    flush=True
                )
            
            # 左右各128个成功状态，防止一个方向主导损失。
            # 根据更新次数确定采样，不改变原训练器探索随机数。
            import numpy as _bc_numpy
            _bc_rng = _bc_numpy.random.default_rng(
                202610274600000 + int(self.total_updates)
            )
            
            _bc_left = _bc_rng.integers(
                0, 10612, size=128
            )
            
            _bc_right = _bc_rng.integers(
                10612, 20319, size=128
            )
            
            _bc_indices = torch.as_tensor(
                _bc_numpy.concatenate(
                    [_bc_left, _bc_right]
                ),
                dtype=torch.long,
                device=state.device
            )
            
            _bc_predictions = self.actor(
                self._t46_success_obs[_bc_indices]
            )
            
            _bc_targets = (
                self._t46_success_actions[_bc_indices]
            )
            
            _bc_loss = (
                _bc_predictions - _bc_targets
            ).square().mean()
            
            actor_loss = actor_loss + 1.0 * _bc_loss
            
            if not hasattr(self, "_t46_bc_announced"):
                self._t46_bc_announced = True
                print(
                    "T46_SUCCESS_BC_ACTIVE=PASS",
                    "BATCH=256",
                    "LEFT=128",
                    "RIGHT=128",
                    "WEIGHT=1.0",
                    flush=True
                )
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_value = float(
                actor_loss.detach().cpu()
            )

            with torch.no_grad():

                for target, current in zip(
                    self.actor_target.parameters(),
                    self.actor.parameters()
                ):
                    target.mul_(
                        1.0 - self.tau
                    ).add_(
                        current,
                        alpha=self.tau
                    )

                for target, current in zip(
                    self.critic_target.parameters(),
                    self.critic.parameters()
                ):
                    target.mul_(
                        1.0 - self.tau
                    ).add_(
                        current,
                        alpha=self.tau
                    )

        return {
            "critic_loss": float(
                critic_loss.detach().cpu()
            ),
            "actor_loss": actor_loss_value,
            "updates": self.total_updates
        }
