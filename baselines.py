"""
Baseline evaluation: how do non-learning controllers perform on Hunt?

Controllers:
    random+random, stag+stag, plant+plant, stag+plant  (independent policies:
        the last three form an empirical payoff matrix of the grid game)
    planner  (joint controller that knows the stag's rule: a reference for
        what coordinated hunting can achieve, not a learned result)

All controllers are measured with the shared protocol in evaluation.py (same
evaluation seeds as the learned agents, metrics per 100 timesteps).
Results are printed and saved (with the env config) to results/baselines.json.

Usage:  python baselines.py [--episodes 200] [--max-timesteps 100] [--seed 0]
"""

import argparse
import json
import os

import numpy as np

from evaluation import METRICS, evaluate
from scripted_policies import make_policy
from stag_env import HuntEnv

CONTROLLERS = ["random+random", "stag+stag", "plant+plant", "stag+plant", "planner"]


def make_controller(name, rng, grid_size):
    """Returns a function mapping the joint obs (2, obs_dim) to [action_A, action_B]."""
    if name == "planner":
        return make_policy("planner", rng, grid_size).act_joint
    name_a, name_b = name.split("+")
    pol_a, pol_b = make_policy(name_a, rng, grid_size), make_policy(name_b, rng, grid_size)
    return lambda obs: [pol_a.act(obs[0]), pol_b.act(obs[1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-timesteps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first evaluation seed")
    args = p.parse_args()

    results, env_config = {}, None
    header = f"{'controller':<15}" + "".join(f"{m:>16}" for m in METRICS)
    print(f"Per 100 steps, mean ± std over {args.episodes} episodes\n")
    print(header)
    print("-" * len(header))
    for name in CONTROLLERS:
        env = HuntEnv(max_timesteps=args.max_timesteps)
        rng = np.random.default_rng(args.seed)  # policies' own RNG, separate from the env's
        controller = make_controller(name, rng, env.grid_size)
        res = evaluate(controller, env, args.episodes, first_seed=args.seed)
        env_config = env.config
        env.close()

        results[name] = res
        print(f"{name:<15}" + "".join(f"{res[m][0]:>9.2f} ±{res[m][1]:>6.2f}" for m in METRICS))

    os.makedirs("results", exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump({"run_args": vars(args), "env_config": env_config, "results": results}, f, indent=2)
    print("\nsaved to results/baselines.json")


if __name__ == "__main__":
    main()
