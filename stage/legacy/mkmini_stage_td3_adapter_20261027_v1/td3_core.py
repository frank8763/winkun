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
