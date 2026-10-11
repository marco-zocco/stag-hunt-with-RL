"""
Controllers:
    random+random, stag+stag, plant+plant, stag+plant  (independent policies:
        the last three form an empirical payoff matrix of the grid game)
    planner  (joint controller that knows the stag's rule: a reference for
        what coordinated hunting can achieve, not a learned result)


Usage:  python baselines.py [--episodes 200] [--max-timesteps 100] [--seed 0]
        python baselines.py --no-stag-follows          # same environment options as train.py
        python baselines.py --grid-size 9 9 --controllers planner

Results go to results/baselines.json for the default environment, and to
results/baselines_<env>.json otherwise (e.g. baselines_nofollow.json, baselines_grid9.json).
"""

import argparse
import json
import os

import numpy as np

from evaluation import METRICS, evaluate
from scripted_policies import make_policy
from stag_env import HuntEnv
from train import env_label  # same environment tags as the run names

CONTROLLERS = ["random+random", "stag+stag", "plant+plant", "stag+plant", "planner"]


def make_controller(name, rng, grid_size, planner_env=None):
    """Returns a function mapping the joint obs (2, obs_dim) to [action_A, action_B]."""
    if name == "planner":
        return make_policy("planner", rng, grid_size, **(planner_env or {})).act_joint
    name_a, name_b = name.split("+")
    pol_a, pol_b = make_policy(name_a, rng, grid_size), make_policy(name_b, rng, grid_size)
    return lambda obs: [pol_a.act(obs[0]), pol_b.act(obs[1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-timesteps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first evaluation seed")
    # environment (same flags and defaults as train.py)
    p.add_argument("--grid-size", type=int, nargs=2, default=[5, 5])
    p.add_argument("--stag-reward", type=float, default=5)
    p.add_argument("--forage-reward", type=float, default=1)
    p.add_argument("--mauling-punishment", type=float, default=-5)
    p.add_argument("--no-stag-follows", action="store_true")
    p.add_argument("--controllers", nargs="+", default=CONTROLLERS, choices=CONTROLLERS)
    args = p.parse_args()

    env_kwargs = dict(
        grid_size=tuple(args.grid_size),
        max_timesteps=args.max_timesteps,
        stag_reward=args.stag_reward,
        forage_reward=args.forage_reward,
        mauling_punishment=args.mauling_punishment,
        stag_follows=not args.no_stag_follows,
    )
    planner_env = dict(stag_follows=not args.no_stag_follows, stag_reward=args.stag_reward,
                       mauling_punishment=args.mauling_punishment)
    tag = env_label({**env_kwargs, "grid_size": list(args.grid_size)})  # "" for the default environment

    results, env_config = {}, None
    header = f"{'controller':<15}" + "".join(f"{m:>16}" for m in METRICS)
    print(f"Environment: {tag.lstrip('_') or 'default'} | per 100 steps, mean ± std over {args.episodes} episodes\n")
    print(header)
    print("-" * len(header))
    for name in args.controllers:
        env = HuntEnv(**env_kwargs)
        rng = np.random.default_rng(args.seed)  # policies' own RNG, separate from the env's
        controller = make_controller(name, rng, env.grid_size, planner_env)
        res = evaluate(controller, env, args.episodes, first_seed=args.seed)
        env_config = env.config
        env.close()

        results[name] = res
        print(f"{name:<15}" + "".join(f"{res[m][0]:>9.2f} ±{res[m][1]:>6.2f}" for m in METRICS))

    os.makedirs("results", exist_ok=True)
    out = f"results/baselines{tag}.json"
    with open(out, "w") as f:
        json.dump({"run_args": vars(args), "env_config": env_config, "results": results}, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
