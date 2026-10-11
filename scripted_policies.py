"""
Hand-coded policies for the Hunt environment.

CoordinatedPlanner: it is a joint controller
(one decision for both agents) used as a reference for what coordination achieves
"""

import numpy as np

LEFT, DOWN, RIGHT, UP, STAND = 0, 1, 2, 3, 4


def _unnormalize(obs_row, grid_size):
    """[0, 1] floats -> integer grid coordinates"""
    w, h = grid_size
    scale = np.tile([w - 1, h - 1], len(obs_row) // 2)
    return np.rint(obs_row * scale).astype(int)


def _step_towards(pos, target, rng):
    """Greedy move that reduces the Manhattan distance to target"""
    options = []
    if target[0] > pos[0]:
        options.append(RIGHT)
    if target[0] < pos[0]:
        options.append(LEFT)
    if target[1] > pos[1]:
        options.append(DOWN)
    if target[1] < pos[1]:
        options.append(UP)
    return rng.choice(options) if options else STAND


class RandomPolicy:
    """Random action (no startegy)"""

    name = "random"

    def __init__(self, rng):
        self.rng = rng  

    def act(self, obs_row):
        return int(self.rng.integers(0, 5))


class ChaseStagPolicy:
    """Always walk towards the stag (naive 'cooperate' strategy)"""

    name = "stag"

    def __init__(self, rng, grid_size):
        self.rng, self.grid_size = rng, grid_size

    def act(self, obs_row):
        c = _unnormalize(obs_row, self.grid_size)
        return int(_step_towards(c[0:2], c[4:6], self.rng))


class ChasePlantPolicy:
    """Always walk towards the nearest plant (safe 'defect' strategy)"""

    name = "plant"

    def __init__(self, rng, grid_size):
        self.rng, self.grid_size = rng, grid_size

    def act(self, obs_row):
        c = _unnormalize(obs_row, self.grid_size)
        me = c[0:2]
        plants = c[6:].reshape(-1, 2)
        nearest = plants[np.argmin(np.abs(plants - me).sum(axis=1))]
        return int(_step_towards(me, nearest, self.rng))


class CoordinatedPlanner:
    """
    Reference JOINT controller: chooses both agents' actions together

    It knows the stag's rule (the stag moves first: one cell towards the nearest
    agent, ties -> agent B; or, with stag_follows=False, one random step that
    stays inside the grid) and tries all 25 joint actions, scoring each by the
    expected immediate reward over the stag's possible moves, with a small bonus
    for staying close to the stag. It is NOT a learned agent and uses knowledge
    of the dynamics

    Works from obs[0] (agent A's view), where A's position comes first.
    """

    name = "planner"
    joint = True

    def __init__(self, grid_size, stag_follows=True, stag_reward=5, mauling_punishment=-5):
        self.grid_size = grid_size
        self.stag_follows = stag_follows
        self.catch_value = 2 * stag_reward   # both agents are paid
        self.maul_value = mauling_punishment

    def _move(self, pos, action):
        (x, y), (w, h) = pos, self.grid_size
        if action == LEFT:
            x = max(x - 1, 0)
        elif action == RIGHT:
            x = min(x + 1, w - 1)
        elif action == UP:
            y = max(y - 1, 0)
        elif action == DOWN:
            y = min(y + 1, h - 1)
        return (x, y)

    def _stag_moves(self, stag, a, b):
        """The stag's possible moves this step, mirroring the env's _move_stag."""
        if not self.stag_follows:
            # random step, never off the grid (env: _random_move)
            (w, h) = self.grid_size
            moves = [LEFT, RIGHT, UP, DOWN]
            if stag[0] == 0:
                moves.remove(LEFT)
            elif stag[0] == w - 1:
                moves.remove(RIGHT)
            if stag[1] == 0:
                moves.remove(UP)
            elif stag[1] == h - 1:
                moves.remove(DOWN)
            return moves
        da = np.hypot(a[0] - stag[0], a[1] - stag[1])
        db = np.hypot(b[0] - stag[0], b[1] - stag[1])
        target = a if da < db else b  # ties go to B, as in the env
        moves = []
        if stag[0] < target[0]:
            moves.append(RIGHT)
        if stag[0] > target[0]:
            moves.append(LEFT)
        if stag[1] > target[1]:
            moves.append(UP)
        if stag[1] < target[1]:
            moves.append(DOWN)
        return moves or [STAND]

    def act_joint(self, obs):
        c = _unnormalize(obs[0], self.grid_size)
        a, b, stag = tuple(c[0:2]), tuple(c[2:4]), tuple(c[4:6])
        stag_moves = self._stag_moves(stag, a, b)

        best, best_score = (STAND, STAND), -np.inf
        for ja in range(5):
            for jb in range(5):
                na, nb = self._move(a, ja), self._move(b, jb)
                score = 0.0
                for sm in stag_moves:  # each possible stag move, equally likely
                    ns = self._move(stag, sm)
                    on_a, on_b = na == ns, nb == ns
                    reward = self.catch_value if (on_a and on_b) else (self.maul_value if (on_a or on_b) else 0)
                    dist = abs(na[0] - ns[0]) + abs(na[1] - ns[1]) + abs(nb[0] - ns[0]) + abs(nb[1] - ns[1])
                    score += (reward - 0.1 * dist) / len(stag_moves)
                if score > best_score:
                    best, best_score = (ja, jb), score
        return list(best)


def make_policy(name, rng, grid_size, **env):
    """env: optional stag_follows / stag_reward / mauling_punishment (used by the planner)"""
    if name == "random":
        return RandomPolicy(rng)
    if name == "stag":
        return ChaseStagPolicy(rng, grid_size)
    if name == "plant":
        return ChasePlantPolicy(rng, grid_size)
    if name == "planner":
        return CoordinatedPlanner(grid_size, **env)
    raise ValueError(f"unknown policy: {name}")
