"""
Independent DQN learners for the two Hunt agents.

DQNLearner: one Q-network + target network + replay buffer + optimizer.
    - epsilon-greedy acting
    - Double-DQN target: the online net picks the next action, the target net
      evaluates it,  y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a)).
      This reduces the overestimation bias of the plain max target.
    - Huber loss and gradient-norm clipping for stability.

Team: maps the env's two agents onto learners.
    - shared:   ONE learner for both agents (parameter sharing). A one-hot agent ID
                is appended to each observation, because the env treats A and B
                differently (on distance ties the stag chases B). Each agent still
                acts from its own observation only (decentralized execution).
    - separate: TWO independent learners, each with its own network and buffer.

Why Q-learning here (and not V + lookahead, as for Peg Solitaire): the next state
depends on the partner's action and on random stag moves, so successor states
cannot be computed in advance; Q(s, a) has to be learned directly.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from model import QNetwork
from replay import ReplayBuffer


class DQNLearner:
    def __init__(
        self,
        in_dim,
        n_actions=5,
        hidden=(64, 64),
        gamma=0.95,
        lr=5e-4,
        buffer_capacity=50_000,
        batch_size=64,
        grad_clip=10.0,
        double=True,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.double = double

        self.online = QNetwork(in_dim, n_actions, hidden)
        self.target = copy.deepcopy(self.online)  # frozen copy, used only for targets
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, in_dim)

    # ------------------------------------------------------------- acting

    @torch.no_grad()
    def greedy(self, x):
        """x: (k, in_dim) numpy array -> (k,) greedy actions."""
        q = self.online(torch.as_tensor(x, dtype=torch.float32))
        return q.argmax(dim=1).numpy()

    def act(self, x, epsilon, rng):
        actions = self.greedy(x)
        if epsilon > 0.0:
            explore = rng.random(len(actions)) < epsilon
            actions[explore] = rng.integers(0, self.n_actions, size=int(explore.sum()))
        return actions

    # ----------------------------------------------------------- learning

    def update(self, rng):
        """One gradient step on a uniform minibatch. Returns the loss (or None)."""
        if len(self.buffer) < self.batch_size:
            return None
        b = self.buffer.sample(self.batch_size, rng)
        obs = torch.as_tensor(b["obs"])
        actions = torch.as_tensor(b["actions"])
        rewards = torch.as_tensor(b["rewards"])
        next_obs = torch.as_tensor(b["next_obs"])
        dones = torch.as_tensor(b["dones"])

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_a = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_a).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            y = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q, y)  # Huber: robust to the large +5/-5 errors
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def state_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])


class Team:
    """The two Hunt agents, backed by one shared learner or two separate ones."""

    def __init__(self, mode, obs_dim, **learner_kwargs):
        if mode not in ("shared", "separate"):
            raise ValueError("mode must be 'shared' or 'separate'")
        self.mode = mode
        self.shared = mode == "shared"
        in_dim = obs_dim + 2 if self.shared else obs_dim  # + one-hot agent ID
        n_learners = 1 if self.shared else 2
        self.learners = [DQNLearner(in_dim, **learner_kwargs) for _ in range(n_learners)]
        self._ids = np.eye(2, dtype=np.float32)

    def _inputs(self, obs):
        """(2, obs_dim) env observation -> (2, in_dim) network inputs."""
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._ids], axis=1) if self.shared else obs

    def learner_of(self, agent):
        return self.learners[0] if self.shared else self.learners[agent]

    def act(self, obs, epsilon=0.0, rng=None):
        x = self._inputs(obs)
        if self.shared:
            return [int(a) for a in self.learners[0].act(x, epsilon, rng)]
        return [int(self.learners[i].act(x[i : i + 1], epsilon, rng)[0]) for i in range(2)]

    def store(self, obs, actions, rewards, next_obs, done):
        x, nx = self._inputs(obs), self._inputs(next_obs)
        for i in range(2):
            self.learner_of(i).buffer.add(x[i], actions[i], rewards[i], nx[i], done)

    def update(self, rng):
        losses = [l.update(rng) for l in self.learners]
        losses = [l for l in losses if l is not None]
        return float(np.mean(losses)) if losses else None

    def sync_target(self):
        for l in self.learners:
            l.sync_target()

    def greedy_controller(self):
        """Controller for evaluation: joint obs -> greedy actions (no exploration)."""
        return lambda obs: self.act(obs, epsilon=0.0)

    def state_dict(self):
        return {"mode": self.mode, "learners": [l.state_dict() for l in self.learners]}

    def load_state_dict(self, state):
        for l, s in zip(self.learners, state["learners"]):
            l.load_state_dict(s)
