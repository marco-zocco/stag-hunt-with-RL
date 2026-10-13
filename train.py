"""Train two DQN agents on Hunt

Prosocial reward:s
U_i = (1 - alpha_i) * r_i + alpha_i * r_j

Options include hysteretic DQN (--hyst-beta) and NoisyNet exploration
(--noisy). Hysteresis can be annealed with --hyst-anneal-start/end

Training uses --seed. Evaluation uses fixed seeds and runs every
--eval-every steps

Outputs in runs/<name>/: config.json, eval_log.csv, final_eval.json,
and model.pt
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


def hyst_beta_at(args, step):
    """Hysteretic beta at this step: constant, or annealed linearly to 1."""
    if args.hyst_beta >= 1.0 or args.hyst_anneal_start >= 1.0:
        return args.hyst_beta
    start, end = args.hyst_anneal_start * args.total_steps, args.hyst_anneal_end * args.total_steps
    return linear_schedule(args.hyst_beta, 1.0, (step - start) / max(end - start, 1.0))


def method_label(a):
    """
    Short tag for the learning rule of a run (a: dict of train.py args).
    Used in run names and by aggregate.py; .get() keeps older runs readable.
    """
    label = "dqn"
    if a.get("hyst_beta", 1.0) < 1.0:
        label = f"hyst{a['hyst_beta']:g}"
        if a.get("hyst_anneal_start", 1.0) < 1.0:
            label += f"-ann{a['hyst_anneal_start']:g}-{a['hyst_anneal_end']:g}"
    if a.get("noisy", False):
        noisy = "noisy" if a.get("noise_resample", "step") == "step" else "noisyep"
        label = noisy if label == "dqn" else f"{label}+{noisy}"
        if (a.get("eps_start", 0.0), a.get("eps_end", 0.0)) != (0.0, 0.0):
            label += f"+eps{a['eps_start']:g}-{a['eps_end']:g}"
    elif (a.get("eps_start", 1.0), a.get("eps_end", 0.05)) != (1.0, 0.05):
        label += f"+eps{a['eps_start']:g}-{a['eps_end']:g}"
    uses_eps = not a.get("noisy", False) or (a.get("eps_start", 0.0), a.get("eps_end", 0.0)) != (0.0, 0.0)
    if uses_eps and a.get("eps_frac", 0.5) != 0.5:
        label += f"+epsfrac{a['eps_frac']:g}"  # epsilon reaches its minimum after this fraction of training
    return label


def env_label(a):
    """
    Suffix for run names listing the environment settings that differ from the
    defaults (the mauling penalty is always in the name already). Empty for the
    default environment, so runs made before this option existed keep their names.
    """
    out = ""
    grid = list(a.get("grid_size", [5, 5]))
    if grid != [5, 5]:
        out += f"_grid{grid[0]}" if grid[0] == grid[1] else f"_grid{grid[0]}x{grid[1]}"
    if a.get("max_timesteps", 100) != 100:
        out += f"_T{a['max_timesteps']}"
    if a.get("stag_reward", 5) != 5:
        out += f"_stag{a['stag_reward']:g}"
    if a.get("forage_reward", 1) != 1:
        out += f"_forage{a['forage_reward']:g}"
    if a.get("run_away_after_maul", False):
        out += "_runaway"
    if not a.get("stag_follows", True):
        out += "_nofollow"
    return out


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
    p.add_argument("--no-stag-follows", action="store_true",
                   help="the stag moves randomly instead of walking towards the nearest agent")
    # DQN
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--target-sync", type=int, default=1_000, help="env steps between target syncs")
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--eps-start", type=float, default=None, help="default 1.0 (0.0 with --noisy)")
    p.add_argument("--eps-end", type=float, default=None, help="default 0.05 (0.0 with --noisy)")
    p.add_argument("--eps-frac", type=float, default=0.5, help="fraction of training to anneal epsilon")
    p.add_argument("--no-double", action="store_true", help="plain DQN target instead of Double DQN")
    # learning-rule variants
    p.add_argument("--hyst-beta", type=float, default=1.0, help="loss weight of negative TD errors (1 = off)")
    p.add_argument("--hyst-anneal-start", type=float, default=1.0, help="fraction of training where beta starts rising to 1 (1 = never)")
    p.add_argument("--hyst-anneal-end", type=float, default=1.0, help="fraction of training where beta reaches 1")
    p.add_argument("--noisy", action="store_true", help="NoisyNet layers instead of epsilon-greedy")
    p.add_argument("--noisy-sigma0", type=float, default=0.5, help="initial noise scale (paper: 0.5)")
    p.add_argument("--noise-resample", choices=["step", "episode"], default="step", help="how often the acting noise changes")
    # protocol
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-episodes", type=int, default=200)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None, help="run name (default: built from the config)")
    args = p.parse_args(argv)

    args.stag_follows = not args.no_stag_follows  # stored under the HuntEnv argument name
    args.alpha_a = args.alpha if args.alpha_a is None else args.alpha_a
    args.alpha_b = args.alpha if args.alpha_b is None else args.alpha_b
    if args.eps_start is None:
        args.eps_start = 0.0 if args.noisy else 1.0
    if args.eps_end is None:
        args.eps_end = 0.0 if args.noisy else 0.05
    if not 0.0 < args.hyst_beta <= 1.0:
        p.error("--hyst-beta must be in (0, 1]")
    if args.hyst_anneal_start < 1.0 and not args.hyst_anneal_start < args.hyst_anneal_end <= 1.0:
        p.error("need --hyst-anneal-start < --hyst-anneal-end <= 1")
    if args.name is None:
        method = method_label(vars(args))
        args.name = (
            f"{args.mode}"
            + ("" if method == "dqn" else f"_{method}")  # plain DQN keeps the old run names
            + f"_a{args.alpha_a:g}-{args.alpha_b:g}_maul{args.mauling_punishment:g}"
            + env_label(vars(args))  # empty for the default environment: old run names unchanged
            + ("" if args.total_steps == 300_000 else f"_steps{args.total_steps // 1000}k")
            + f"_s{args.seed}"
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
        stag_follows=args.stag_follows,
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
        noisy=args.noisy,
        sigma0=args.noisy_sigma0,
    )
    alphas = np.array([args.alpha_a, args.alpha_b], dtype=np.float32)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "env_config": env.config}, f, indent=2)

    log_path = os.path.join(run_dir, "eval_log.csv")
    fields = ["step", "episodes", "epsilon", "hyst_beta", "loss", "train_return_A", "train_return_B", "minutes"]
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
        beta = hyst_beta_at(args, step)
        if args.noisy and (args.noise_resample == "step" or step == 1):
            team.resample_noise()

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
            if args.noisy and args.noise_resample == "episode":
                team.resample_noise()

        if step >= args.learning_starts:
            loss = team.update(rng, neg_weight=beta)
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
                "hyst_beta": round(beta, 4),
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
                f"step {step:>7} | eps {epsilon:.2f} | beta {beta:.2f} | loss {row['loss']} | "
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
