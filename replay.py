"""
Uniform experience replay over single-agent transitions (s, a, r, s', done).

In shared mode both agent's transitions go into the same buffer (each tagged by its agent-ID input); 
in separate mode each agent has its own buffer.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self._ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self._ptr = (self._ptr + 1) % self.capacity  # overwrite the oldest when full
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self):
        return self._size
