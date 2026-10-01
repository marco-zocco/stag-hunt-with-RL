"""
Thin wrapper around the Gymnasium-Stag-Hunt "Hunt" environment.

The underlying env (giorgiofranceschelli/Gymnasium-Stag-Hunt) is used as an
external dependency; this module only adapts its interface for our training code:

  1. Seeding      - the env ignores reset(seed=...). It draws randomness from BOTH
                    Python's `random` (plant/stag placement) and `numpy.random`
                    (stag movement), so we seed both globally.
  2. Observations - uint8 grid coordinates -> float32 in [0, 1] (divided by grid
                    size - 1). Shape (2, obs_dim): row i is agent i's view; with
                    flip_obs=True each agent sees its own position first.
  3. Termination  - the env sets terminated = truncated = (t >= max_timesteps).
                    Hunt has no terminal state: the episode end is purely a time
                    limit. We return terminated=False, truncated=True so that
                    value-based agents keep bootstrapping at the cut-off.
  4. Rewards      - tuple (r_A, r_B) -> float32 array of shape (2,).
  5. Events       - per-step and per-episode counts of stag catches, maulings and
                    forages, inferred from the reward pair. These are the
                    cooperation metrics we report.

Two rules for code that uses this wrapper:
  - Never draw random numbers from the GLOBAL `random` / `np.random` (e.g. for
    epsilon-greedy): that would consume the env's stream and break
    reproducibility. Use a private generator, e.g. np.random.default_rng(seed).
  - Observations contain positions only. An agent-ID input, needed when both
    agents share one network (the env treats A and B differently: on distance
    ties the stag chases B), is added on the agent side, so that non-learning
    policies can keep parsing the raw observation.
"""

import random

import gymnasium as gym
import numpy as np

import gymnasium_stag_hunt  # noqa: F401  (registers the StagHunt-* env ids)


class HuntEnv:
    """Two-agent Hunt environment with a clean, reproducible interface."""

    n_agents = 2
    n_actions = 5  # LEFT=0, DOWN=1, RIGHT=2, UP=3, STAND=4

    def __init__(
        self,
        grid_size=(5, 5),
        max_timesteps=100,
        forage_quantity=2,
        stag_reward=5,
        forage_reward=1,
        mauling_punishment=-5,
        stag_follows=True,
        run_away_after_maul=False,
        seed=None,
    ):
        # Events are read from the reward pair, so the three reward values must be
        # distinguishable. The env itself enforces stag > forage >= 0 > maul; we
        # additionally need forage > 0, otherwise a forage looks like "nothing".
        if forage_reward <= 0:
            raise ValueError("forage_reward must be > 0 for event counting to work")

        # Full configuration, saved alongside every result for reproducibility.
        self.config = dict(
            grid_size=list(grid_size),
            max_timesteps=max_timesteps,
            forage_quantity=forage_quantity,
            stag_reward=stag_reward,
            forage_reward=forage_reward,
            mauling_punishment=mauling_punishment,
            stag_follows=stag_follows,
            run_away_after_maul=run_away_after_maul,
        )

        self.grid_size = tuple(grid_size)
        self.max_timesteps = max_timesteps
        self.stag_reward = float(stag_reward)
        self.forage_reward = float(forage_reward)
        self.mauling_punishment = float(mauling_punishment)

        self._env = gym.make(
            "StagHunt-Hunt-v0",
            grid_size=self.grid_size,
            obs_type="coords",
            enable_multiagent=True,
            max_timesteps=max_timesteps,
            forage_quantity=forage_quantity,
            stag_reward=stag_reward,
            forage_reward=forage_reward,
            mauling_punishment=mauling_punishment,
            stag_follows=stag_follows,
            run_away_after_maul=run_away_after_maul,
            flip_obs=True,
            disable_env_checker=True,  # silences warnings about the tuple reward
        )

        # Coordinates are laid out as [x, y, x, y, ...] for A, B, stag, plants.
        n_entities = 3 + forage_quantity
        self.obs_dim = 2 * n_entities
        w, h = self.grid_size
        self._scale = np.tile([w - 1, h - 1], n_entities).astype(np.float32)

        self._seed_next = seed
        self._episode_events = None

    # ------------------------------------------------------------------ API

    def reset(self, seed=None):
        """Start a new episode. Returns (obs, info), obs of shape (2, obs_dim)."""
        seed = seed if seed is not None else self._seed_next
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._seed_next = None  # seed only once; later episodes continue the stream

        obs, _ = self._env.reset()
        self._episode_events = self._zero_events()
        return self._process_obs(obs), {}

    def step(self, actions):
        """
        actions: sequence of 2 ints (one per agent).
        Returns obs (2, obs_dim), rewards (2,), terminated, truncated, info.
        info["events"] holds this step's events, and on the last step
        info["episode"] holds the totals for the whole episode.
        """
        a = [int(actions[0]), int(actions[1])]
        obs, rewards, done, _, _ = self._env.step(a)

        rewards = np.asarray(rewards, dtype=np.float32)
        events = self._events_from_rewards(rewards)
        for k, v in events.items():
            self._episode_events[k] += v

        terminated = False  # no terminal states in Hunt
        truncated = bool(done)  # time limit reached

        info = {"events": events}
        if truncated:
            info["episode"] = dict(self._episode_events, length=self.max_timesteps)
        return self._process_obs(obs), rewards, terminated, truncated, info

    def render(self):
        """Print the grid as text: A, B = agents, S = stag, P = plant, * = overlap."""
        game = self._env.unwrapped.game
        w, h = self.grid_size
        grid = [["." for _ in range(w)] for _ in range(h)]

        def put(pos, ch):
            x, y = int(pos[0]), int(pos[1])
            grid[y][x] = ch if grid[y][x] == "." else "*"

        for p in game.PLANTS:
            put(p, "P")
        put(game.STAG, "S")
        put(game.A_AGENT, "A")
        put(game.B_AGENT, "B")
        print("\n".join(" ".join(row) for row in grid) + "\n")

    def close(self):
        self._env.close()

    # -------------------------------------------------------------- helpers

    def _process_obs(self, obs):
        return np.asarray(obs, dtype=np.float32) / self._scale

    @staticmethod
    def _zero_events():
        return {"stag": 0, "maul_A": 0, "maul_B": 0, "forage_A": 0, "forage_B": 0}

    def _events_from_rewards(self, r):
        ev = self._zero_events()
        if r[0] == self.stag_reward and r[1] == self.stag_reward:
            ev["stag"] = 1  # joint catch: the cooperative outcome
            return ev
        for i, name in enumerate("AB"):
            if r[i] == self.mauling_punishment:
                ev[f"maul_{name}"] = 1  # went for the stag alone
            elif r[i] == self.forage_reward:
                ev[f"forage_{name}"] = 1  # took the safe option
        return ev


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    env = HuntEnv(max_timesteps=100)

    # 1) shapes and value range
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, env.obs_dim) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    # flip_obs: agent B sees itself first, so the rows swap A and B positions
    assert np.allclose(obs[0][:2], obs[1][2:4]) and np.allclose(obs[0][2:4], obs[1][:2])

    # 2) reproducibility: same seed + same actions -> identical trajectory
    def rollout(seed):
        env.reset(seed=seed)
        act_rng = np.random.default_rng(123)  # separate RNG for the actions
        traj, done = [], False
        while not done:
            o, r, term, trunc, info = env.step(act_rng.integers(0, 5, size=2))
            traj.append(np.concatenate([o.ravel(), r]))
            done = term or trunc
        return np.array(traj), info["episode"]

    t1, ep1 = rollout(42)
    t2, ep2 = rollout(42)
    t3, _ = rollout(7)
    assert np.array_equal(t1, t2), "same seed must reproduce the same episode"
    assert not np.array_equal(t1, t3), "different seeds should differ"
    assert len(t1) == 100

    # 3) event counts must match the rewards collected
    total_r = t1[:, -2:].sum(axis=0)
    expected_A = 5 * ep1["stag"] - 5 * ep1["maul_A"] + 1 * ep1["forage_A"]
    expected_B = 5 * ep1["stag"] - 5 * ep1["maul_B"] + 1 * ep1["forage_B"]
    assert np.isclose(total_r[0], expected_A) and np.isclose(total_r[1], expected_B)

    print("obs_dim:", env.obs_dim)
    print("episode (seed 42, random actions):", ep1, "| returns:", total_r)
    print("all checks passed")
    env.close()
