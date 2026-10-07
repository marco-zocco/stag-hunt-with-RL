"""
DQNLearner: one Q-network + target network + replay buffer + optimizer.
    - epsilon-greedy acting
    - Double-DQN target: the online net -> next action, target net -> evaluates it, 
        y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a)).
    - Huber loss and gradient-norm clipping
    - optional hysteretic update
    - optional NoisyNet exploration

Team: maps the env's two agents onto learners.
    - shared: Full parameter sharing. A one-hot agent ID. Decentralized execution
    - separate: Two independent learners with two networks and buffers
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
        noisy=False,
        sigma0=0.5,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.double = double
        self.noisy = noisy

        self.online = QNetwork(in_dim, n_actions, hidden, noisy=noisy, sigma0=sigma0)
        self.target = copy.deepcopy(self.online)  # frozen copy, used only for targets
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, in_dim)

    # --------------------  acting

    @torch.no_grad()
    def greedy(self, x):
        """x: (k, in_dim) numpy array -> (k,) greedy actions (noisy nets: mean weights)."""
        self.online.eval()  # only changes anything for noisy layers
        q = self.online(torch.as_tensor(x, dtype=torch.float32))
        self.online.train()
        return q.argmax(dim=1).numpy()

    @torch.no_grad()
    def act(self, x, epsilon, rng):
        """Training-time action: argmax of the (noisy, if enabled) Q-values, then epsilon."""
        if self.noisy:
            # current acting noise sample; Team.resample_noise() decides how often it changes
            actions = self.online(torch.as_tensor(x, dtype=torch.float32)).argmax(dim=1).numpy()
        else:
            actions = self.greedy(x)
        if epsilon > 0.0:
            explore = rng.random(len(actions)) < epsilon
            actions[explore] = rng.integers(0, self.n_actions, size=int(explore.sum()))
        return actions

    # ------------------- learning

    def resample_noise(self):
        self.online.reset_noise()

    def update(self, rng, neg_weight=1.0):
        """
        One gradient step on a uniform minibatch. Returns the loss (or None).
        neg_weight: hysteretic beta, the loss weight of transitions with a
        negative TD error (1.0 = plain DQN).
        """
        if len(self.buffer) < self.batch_size:
            return None
        if self.noisy:
            # fresh, independent noise for the online and target nets in this
            # update (as in the paper); the acting noise is put back afterwards so
            # that a per-episode acting sample is not changed by learning
            acting_noise = self.online.get_noise()
            self.online.reset_noise()
            self.target.reset_noise()
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

        # Huber: robust to the large +5/-5 errors
        if neg_weight == 1.0:
            loss = F.smooth_l1_loss(q, y)
        else:
            per_sample = F.smooth_l1_loss(q, y, reduction="none")
            # TD error delta = y - Q: weight 1 if delta >= 0, beta if delta < 0.
            # The weight is a constant (no gradient flows through it).
            w = torch.where(y < q.detach(), neg_weight, 1.0)
            loss = (w * per_sample).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.noisy:
            self.online.set_noise(acting_noise)
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

    def resample_noise(self):
        """New acting noise sample for every learner (no-op without noisy nets)."""
        for l in self.learners:
            l.resample_noise()

    def update(self, rng, neg_weight=1.0):
        losses = [l.update(rng, neg_weight) for l in self.learners]
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
