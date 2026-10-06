"""
Evaluation rotocol:
  - episode i is reset with evaluation seed i (seeds 0, 1, 2, ...), every controller
    faces the same sequence of starting layouts;
  - metrics are rescaled to "per 100 timesteps" and reported as mean +/- std over
    episodes;
  - the global random state is saved before and restored after
"""

import random

import numpy as np

EVENT_KEYS = ["stag", "maul_A", "maul_B", "forage_A", "forage_B"]
METRICS = EVENT_KEYS + ["return_A", "return_B"]


def evaluate(controller, env, n_episodes, first_seed=0):
    py_state, np_state = random.getstate(), np.random.get_state()
    per_ep = {m: [] for m in METRICS}
    try:
        for i in range(n_episodes):
            obs, _ = env.reset(seed=first_seed + i)
            ret = np.zeros(2)
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(controller(obs))
                ret += r
                done = term or trunc

            ep = info["episode"]
            k = 100.0 / ep["length"]  # counts -> "per 100 steps"
            for m in EVENT_KEYS:
                per_ep[m].append(ep[m] * k)
            per_ep["return_A"].append(ret[0] * k)
            per_ep["return_B"].append(ret[1] * k)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_ep.items()}
