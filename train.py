"""
Train two DQN agents on Hunt.

Each agent learns from a prosocial reward (Peysakhovich & Lerer, 2017, Def. 3):
    U_i = (1 - alpha_i) * r_i + alpha_i * r_j
    alpha = 0 -> selfish, alpha = 0.5 -> fully prosocial (equal weights).
alpha only changes what the agents LEARN from; all logged returns and evaluation
metrics use the env's true rewards.

Protocol:
  - training randomness comes from --seed (use seeds >= 10000, disjoint from the
    evaluation seeds 0, 1, 2, ...);
  - every --eval-every steps: greedy evaluation (epsilon = 0) on --eval-episodes
    fixed evaluation seeds, appended to eval_log.csv;
  - at the end: greedy evaluation on --final-eval-episodes seeds (same seeds as
    baselines.py), saved to final_eval.json.

Outputs in runs/<name>/: config.json, eval_log.csv, final_eval.json, model.pt
(model.pt is also written at every evaluation, atomically, so a killed job keeps
its latest weights).

Examples:
    python train.py --mode shared --alpha 0.5 --mauling-punishment -1   # sanity run
    python train.py --mode shared --alpha 0.0 --seed 10001               # default rewards
"""

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch

from dqn import Team
from evaluation import METRICS, evaluate
from stag_env import HuntEnv


def linear_schedule(start, end, frac):
    return start + (end - start) * min(max(frac, 0.0), 1.0)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # never leaves a half-written file behind


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # agents
    p.add_argument("--mode", choices=["shared", "separate"], default="shared")
    p.add_argument("--alpha", type=float, default=0.0, help="prosociality of both agents")
    p.add_argument("--alpha-a", type=float, default=None, help="override alpha for agent A")
    p.add_argument("--alpha-b", type=float, default=None, help="override alpha for agent B")
    # environment
    p.add_argument("--grid-size", type=int, nargs=2, default=[5, 5])
    p.add_argument("--max-timesteps", type=int, default=100)
    p.add_argument("--stag-reward", type=float, default=5)
    p.add_argument("--forage-reward", type=float, default=1)
    p.add_argument("--mauling-punishment", type=float, default=-5)
    p.add_argument("--run-away-after-maul", action="store_true")
    # DQN
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--target-sync", type=int, default=1_000, help="env steps between target syncs")
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-frac", type=float, default=0.5, help="fraction of training to anneal epsilon")
    p.add_argument("--no-double", action="store_true", help="plain DQN target instead of Double DQN")
    # protocol
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-episodes", type=int, default=200)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None, help="run name (default: built from the config)")
    args = p.parse_args(argv)

    args.alpha_a = args.alpha if args.alpha_a is None else args.alpha_a
    args.alpha_b = args.alpha if args.alpha_b is None else args.alpha_b
    if args.name is None:
        args.name = (
            f"{args.mode}_a{args.alpha_a:g}-{args.alpha_b:g}"
            f"_maul{args.mauling_punishment:g}_s{args.seed}"
        )
    return args


def train(args):
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)

    # --- reproducibility: every source of randomness is seeded from --seed
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # tiny network: one thread is fastest, and plays well with job arrays
    rng = np.random.default_rng(args.seed)  # exploration + replay sampling (never the global RNG)

    env_kwargs = dict(
        grid_size=tuple(args.grid_size),
        max_timesteps=args.max_timesteps,
        stag_reward=args.stag_reward,
        forage_reward=args.forage_reward,
        mauling_punishment=args.mauling_punishment,
        run_away_after_maul=args.run_away_after_maul,
    )
    env = HuntEnv(**env_kwargs)
    eval_env = HuntEnv(**env_kwargs)  # separate instance: evaluation never touches the training episode

    team = Team(
        args.mode,
        env.obs_dim,
        hidden=tuple(args.hidden),
        gamma=args.gamma,
        lr=args.lr,
        buffer_capacity=args.buffer,
        batch_size=args.batch_size,
        double=not args.no_double,
    )
    alphas = np.array([args.alpha_a, args.alpha_b], dtype=np.float32)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "env_config": env.config}, f, indent=2)

    log_path = os.path.join(run_dir, "eval_log.csv")
    fields = ["step", "episodes", "epsilon", "loss", "train_return_A", "train_return_B", "minutes"]
    fields += [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fields)
    writer.writeheader()

    obs, _ = env.reset(seed=args.seed)
    ep_return = np.zeros(2)
    recent_returns = deque(maxlen=20)  # true env returns of the last training episodes
    recent_losses = deque(maxlen=1_000)
    episodes, t0 = 0, time.time()

    for step in range(1, args.total_steps + 1):
        epsilon = linear_schedule(args.eps_start, args.eps_end, step / (args.eps_frac * args.total_steps))

        actions = team.act(obs, epsilon, rng)
        next_obs, r, terminated, truncated, info = env.step(actions)

        # what each agent learns from: (1 - alpha_i) * r_i + alpha_i * r_j
        learn_r = (1.0 - alphas) * r + alphas * r[::-1]
        team.store(obs, actions, learn_r, next_obs, float(terminated))  # truncation is NOT a terminal

        ep_return += r
        obs = next_obs
        if terminated or truncated:
            recent_returns.append(ep_return.copy())
            ep_return[:] = 0.0
            episodes += 1
            obs, _ = env.reset()  # continues the seeded random stream

        if step >= args.learning_starts:
            loss = team.update(rng)
            if loss is not None:
                recent_losses.append(loss)
        if step % args.target_sync == 0:
            team.sync_target()

        if step % args.eval_every == 0 or step == args.total_steps:
            res = evaluate(team.greedy_controller(), eval_env, args.eval_episodes)
            tr = np.mean(recent_returns, axis=0) if recent_returns else np.full(2, np.nan)
            row = {
                "step": step,
                "episodes": episodes,
                "epsilon": round(epsilon, 4),
                "loss": round(float(np.mean(recent_losses)), 5) if recent_losses else "",
                "train_return_A": round(float(tr[0]), 2),
                "train_return_B": round(float(tr[1]), 2),
                "minutes": round((time.time() - t0) / 60, 2),
            }
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = round(res[m][0], 3), round(res[m][1], 3)
            writer.writerow(row)
            log_file.flush()
            atomic_torch_save({"args": vars(args), "team": team.state_dict()}, os.path.join(run_dir, "model.pt"))
            print(
                f"step {step:>7} | eps {epsilon:.2f} | loss {row['loss']} | "
                f"eval: stag {res['stag'][0]:5.2f}  maul A/B {res['maul_A'][0]:5.2f}/{res['maul_B'][0]:5.2f}  "
                f"forage A/B {res['forage_A'][0]:5.2f}/{res['forage_B'][0]:5.2f}  "
                f"return A/B {res['return_A'][0]:7.2f}/{res['return_B'][0]:7.2f} | {row['minutes']:.1f} min",
                flush=True,
            )

    log_file.close()

    final = evaluate(team.greedy_controller(), eval_env, args.final_eval_episodes)
    with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
        json.dump({"episodes": args.final_eval_episodes, "results": final}, f, indent=2)
    print(f"\nfinal greedy evaluation ({args.final_eval_episodes} episodes, per 100 steps):")
    for m in METRICS:
        print(f"  {m:<9} {final[m][0]:8.2f} ± {final[m][1]:.2f}")
    print(f"saved to {run_dir}/")

    env.close()
    eval_env.close()
    return final


if __name__ == "__main__":
    train(parse_args())
