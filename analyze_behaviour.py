"""
Behaviour analysis of trained Stag Hunt agents.

For every run folder with a model.pt, plays greedy evaluation episodes and measures:

  catches        
  maul, forage   
  coloc          % of steps on which A and B stand on the same cell
  dist_AB        mean Manhattan distance between A and B
  meet           % of episodes in which A and B ever share a cell
  meet_t         mean step at which they first share a cell
  stand_A/B      % of steps on which each agent chooses STAND
  huddle_catch   % of catches where A and B were ALREADY on the same cell
                 before the catching step
  wait_catch     % of catches where both agents chose STAND on the catching step
  pre_dist       mean A-B distance just before a catch (0 = huddle, >0 = converging)

Usage (from the repo root):
  python analyze_behaviour.py                        # all runs in runs/
  python analyze_behaviour.py --runs runs/shared_hyst*  --episodes 200
  python analyze_behaviour.py --show runs/<one_run>  # ASCII replay of one episode

Writes results/behaviour_runs.csv (one row per run) and
results/behaviour_summary.csv (per configuration, cooperative seeds only).
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import torch

STAND = 4
ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP", "STAND"]
COOP_THRESHOLD = 8.0  # same threshold as aggregate.py: 8 catches x 5 = 40 = foraging return


# --------------------------------------------------------- loading

def env_kwargs(a):
    """The HuntEnv arguments a run was trained with (taken from its saved args)."""
    import inspect
    sys.path.insert(0, os.getcwd())
    from stag_env import HuntEnv
    params = inspect.signature(HuntEnv.__init__).parameters
    kw = {k: a[k] for k in params if k not in ("self", "seed") and k in a and a[k] is not None}
    if "grid_size" in kw:
        g = kw["grid_size"]
        kw["grid_size"] = tuple(int(x) for x in g) if isinstance(g, (list, tuple)) else (int(g), int(g))
    return kw


def make_env(a):
    """The environment the run was trained in (grid, rewards, stag_follows, ...)."""
    from stag_env import HuntEnv
    return HuntEnv(**env_kwargs(a))


def load_controller(model_path):
    sys.path.insert(0, os.getcwd())
    from dqn import Team

    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    obs_dim = make_env(a).obs_dim
    team = Team(a["mode"], obs_dim, hidden=tuple(a["hidden"]), noisy=a.get("noisy", False))
    team.load_state_dict(ck["team"])
    return team.greedy_controller(), a


def grid_size(a):
    g = a.get("grid_size", 5)
    return int(g[0] if isinstance(g, (list, tuple)) else g)


def non_default_env(a):
    defaults = {"grid_size": 5, "stag_reward": 5, "forage_reward": 1, "mauling_punishment": -5,
                "max_timesteps": 100, "forage_quantity": 2, "stag_follows": True, "run_away_after_maul": False}
    out = {}
    for k, d in defaults.items():
        v = a.get(k, d)
        v = v[0] if isinstance(v, (list, tuple)) else v
        if v is not None and v != d and not (isinstance(d, (int, float)) and not isinstance(d, bool)
                                              and float(v) == float(d)):
            out[k] = v
    return out


def decode(obs, g):
    """Agent A's observation -> integer cells: A, B, stag (entries 0:2, 2:4, 4:6)."""
    c = np.rint(np.asarray(obs)[0] * (g - 1)).astype(int)
    return c[0:2], c[2:4], c[4:6]


def final_eval_catches(run_dir):
    p = os.path.join(run_dir, "final_eval.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if "catch" in k.lower() or k.lower() == "stag":
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, dict) and isinstance(v.get("mean"), (int, float)):
                        return float(v["mean"])
                    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                        return float(v[0])  # train.py saves [mean, std]
                if isinstance(v, dict):
                    stack.append(v)
    return None


# ------------------------------------------------------ analysis

def analyse_run(job):
    run_dir, episodes, seed0 = job
    torch.set_num_threads(1)
    name = os.path.basename(os.path.normpath(run_dir))
    ctrl, a = load_controller(os.path.join(run_dir, "model.pt"))
    env = make_env(a)
    g = grid_size(a)
    stag_r = float(a.get("stag_reward", 5))
    maul_r = float(a.get("mauling_punishment", -5))
    forage_r = float(a.get("forage_reward", 1))

    steps = catches = huddle = wait = coloc = 0
    maul = np.zeros(2)
    forage = np.zeros(2)
    stand = np.zeros(2)
    dist_sum = pre_dist_sum = 0.0
    meet_eps, meet_ts = 0, []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        A, B, _ = decode(obs, g)
        met_t, t, done = None, 0, False
        while not done:
            out = ctrl(obs)
            act = np.asarray(out).reshape(-1)[:2]
            was_together = bool((A == B).all())
            before = int(np.abs(A - B).sum())

            obs, r, te, tr, _ = env.step(out)
            done = bool(np.any(te) or np.any(tr))
            r = np.asarray(r, dtype=float).reshape(-1)
            t += 1
            steps += 1
            stand += act == STAND

            if r[0] == stag_r and r[1] == stag_r:
                catches += 1
                huddle += was_together
                wait += bool((act == STAND).all())
                pre_dist_sum += before
            maul += r == maul_r
            forage += r == forage_r

            A, B, _ = decode(obs, g)
            together = bool((A == B).all())
            coloc += together
            dist_sum += int(np.abs(A - B).sum())
            if together and met_t is None:
                met_t = t
        if met_t is not None:
            meet_eps += 1
            meet_ts.append(met_t)

    pct = lambda x, n: 100.0 * x / n if n else float("nan")
    return {
        "run": name,
        "config": re.sub(r"_s\d+$", "", name),
        "seed": (re.search(r"_s(\d+)$", name) or [None, ""])[1],
        "episodes": episodes,
        "catches": catches / episodes,
        "final_eval_catches": final_eval_catches(run_dir),
        "maul": maul.sum() / episodes, "forage": forage.sum() / episodes,
        "maul_A": maul[0] / episodes, "maul_B": maul[1] / episodes,
        "forage_A": forage[0] / episodes, "forage_B": forage[1] / episodes,
        "coloc": pct(coloc, steps),
        "dist_AB": dist_sum / steps,
        "meet": pct(meet_eps, episodes),
        "meet_t": float(np.mean(meet_ts)) if meet_ts else float("nan"),
        "stand_A": pct(stand[0], steps), "stand_B": pct(stand[1], steps),
        "huddle_catch": pct(huddle, catches),
        "wait_catch": pct(wait, catches),
        "pre_dist": pre_dist_sum / catches if catches else float("nan"),
        "env": json.dumps(non_default_env(a)) if non_default_env(a) else "default",
    }


# --------------------------------------------------------------------------- replay

def show(run_dir, steps, seed):
    ctrl, a = load_controller(os.path.join(run_dir, "model.pt"))
    env = make_env(a)
    g = grid_size(a)
    obs, _ = env.reset(seed=seed)
    print(f"{os.path.basename(os.path.normpath(run_dir))} | episode seed {seed}")
    print("A, B = agents, * = both agents on one cell, S = stag, @ = stag on an agent's cell, p = plant")
    print("x grows to the right, y grows downwards; agents start in the top corners\n")
    for t in range(steps + 1):
        c = np.rint(np.asarray(obs)[0] * (g - 1)).astype(int)
        grid = [["." for _ in range(g)] for _ in range(g)]
        for k in range(6, len(c), 2):
            grid[c[k + 1]][c[k]] = "p"
        for k, ch in ((0, "A"), (2, "B")):
            x, y = c[k], c[k + 1]
            grid[y][x] = "*" if grid[y][x] in ("A", "B") else ch
        sx, sy = c[4], c[5]
        grid[sy][sx] = "@" if grid[sy][sx] in ("A", "B", "*") else "S"
        print(f"step {t}")
        print("\n".join("  " + " ".join(row) for row in grid))
        if t == steps:
            break
        out = ctrl(obs)
        act = np.asarray(out).reshape(-1)[:2]
        obs, r, *_ = env.step(out)
        print(f"  -> A {ACTION_NAMES[act[0]]}, B {ACTION_NAMES[act[1]]}   rewards A/B: {np.asarray(r).tolist()}\n")


# ----------------------------------------------------- main

COLS = [("catches", "catch", "{:6.1f}"), ("final_eval_catches", "eval", "{:5.1f}"),
        ("maul", "maul", "{:5.2f}"), ("forage", "forage", "{:6.2f}"),
        ("coloc", "coloc%", "{:6.0f}"), ("dist_AB", "dAB", "{:5.2f}"),
        ("meet", "meet%", "{:5.0f}"), ("meet_t", "meet_t", "{:6.1f}"),
        ("stand_A", "standA%", "{:7.0f}"), ("stand_B", "standB%", "{:7.0f}"),
        ("huddle_catch", "huddle%", "{:7.0f}"), ("wait_catch", "wait%", "{:5.0f}"),
        ("pre_dist", "preD", "{:5.2f}")]


def fmt(v, f):
    return " " * len(f.format(0.0)) if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run folders (default: every runs/* with a model.pt)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", default=None, help="print an ASCII replay of one run instead")
    ap.add_argument("--show-steps", type=int, default=15)
    args = ap.parse_args()

    if args.show:
        show(args.show, args.show_steps, args.seed)
        return

    runs = sorted(d for d in (args.runs or glob.glob("runs/*")) if os.path.exists(os.path.join(d, "model.pt")))
    if not runs:
        sys.exit("no run folders with a model.pt found")
    print(f"analysing {len(runs)} runs x {args.episodes} episodes on {args.jobs} processes ...", flush=True)
    with Pool(min(args.jobs, len(runs))) as pool:
        rows = pool.map(analyse_run, [(r, args.episodes, args.seed) for r in runs])
    rows.sort(key=lambda r: (r["config"], r["seed"]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "behaviour_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # per-run table
    width = max(len(r["config"]) for r in rows) + 9
    print("\n" + "run".ljust(width) + " ".join(h.rjust(len(f.format(0.0))) for _, h, f in COLS))
    last = None
    for r in rows:
        if last is not None and r["config"] != last:
            print()
        last = r["config"]
        label = f"{r['config']} s{r['seed']}" if r["seed"] else r["config"]
        print(label.ljust(width) + " ".join(fmt(r[k], f) for k, _, f in COLS))

    # per-config summary over cooperative seeds only
    summ = []
    for cfg in sorted({r["config"] for r in rows}):
        rs = [r for r in rows if r["config"] == cfg]
        co = [r for r in rs if r["catches"] > COOP_THRESHOLD]
        s = {"config": cfg, "seeds": len(rs), "coop_seeds": len(co)}
        for k, _, _ in COLS:
            vals = [r[k] for r in co if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
            s[k] = float(np.mean(vals)) if vals else float("nan")
        summ.append(s)
    with open(os.path.join(args.out, "behaviour_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    print(f"\nSUMMARY: mean over cooperative seeds only (catches > {COOP_THRESHOLD:g})")
    width = max(len(s["config"]) for s in summ) + 2
    print("config".ljust(width) + " coop " + " ".join(h.rjust(len(f.format(0.0))) for _, h, f in COLS))
    for s in summ:
        print(s["config"].ljust(width) + f" {s['coop_seeds']}/{s['seeds']}  " + " ".join(fmt(s[k], f) for k, _, f in COLS))
    print(f"\nsaved {args.out}/behaviour_runs.csv and {args.out}/behaviour_summary.csv")


if __name__ == "__main__":
    main()
