"""
Uses every run trained with SEPARATE: agent A and agent B are two
independent Q-networks. 
Runs are only paired with runs trained in the same environment configuration.

Usage (from the repo root):
  python crossplay.py                       # all separate-network runs in runs/
  python crossplay.py --filter noisyep      # only runs whose name matches a regex
  python crossplay.py --episodes 200

Writes results/crossplay_pairs.csv (one row per pair) and results/crossplay_summary.txt.
"""

import argparse
import csv
import glob
import inspect
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool

import numpy as np
import torch

STAND = 4
COOP = 8.0      # same threshold as aggregate.py (8 catches x 5 = 40 = foraging return)
WAITER = 40.0   # % STAND in self-play above which an agent counts as a "waiter"


# --------------------------------------------------------------------------- loading

def env_kwargs(a):
    """The HuntEnv arguments a run was trained with (taken from its saved args)."""
    from stag_env import HuntEnv
    params = inspect.signature(HuntEnv.__init__).parameters
    kw = {k: a[k] for k in params if k not in ("self", "seed") and k in a and a[k] is not None}
    if "grid_size" in kw and not isinstance(kw["grid_size"], (list, tuple)):
        kw["grid_size"] = (int(kw["grid_size"]), int(kw["grid_size"]))
    if "grid_size" in kw:
        kw["grid_size"] = tuple(int(g) for g in kw["grid_size"])
    return kw


def load_args(run_dir):
    return torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)["args"]


def load_pair(run_dir):
    """-> (agent A, agent B, args) as DQNLearners holding the trained online networks."""
    from dqn import DQNLearner
    from stag_env import HuntEnv
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    obs_dim = HuntEnv(**env_kwargs(a)).obs_dim
    agents = []
    for k in range(2):
        l = DQNLearner(obs_dim, hidden=tuple(a["hidden"]), noisy=a.get("noisy", False), buffer_capacity=1)
        l.online.load_state_dict(ck["team"]["learners"][k]["online"])
        agents.append(l)
    return agents[0], agents[1], a


# ------------------------------------------- playing

RUNS = []      # filled in every worker by _init
AGENTS = {}


def _init(runs):
    torch.set_num_threads(1)
    sys.path.insert(0, os.getcwd())
    global RUNS, AGENTS
    RUNS = runs
    with redirect_stdout(io.StringIO()):          # silence the pygame banner
        AGENTS = {r: load_pair(r) for r in runs}


def play(job):
    i, j, episodes, seed0 = job
    from stag_env import HuntEnv
    A = AGENTS[RUNS[i]][0]
    B = AGENTS[RUNS[j]][1]
    a = AGENTS[RUNS[i]][2]
    env = HuntEnv(**env_kwargs(a))
    scale = env._scale

    steps = catches = coloc = 0
    maul = np.zeros(2); forage = np.zeros(2); stand = np.zeros(2)
    meet_eps, meet_ts = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        met, t, done = None, 0, False
        while not done:
            act = [int(A.greedy(obs[0:1])[0]), int(B.greedy(obs[1:2])[0])]
            obs, r, te, tr, info = env.step(act)
            done = te or tr
            t += 1; steps += 1
            ev = info["events"]
            catches += ev["stag"]
            maul += [ev["maul_A"], ev["maul_B"]]
            forage += [ev["forage_A"], ev["forage_B"]]
            stand += [act[0] == STAND, act[1] == STAND]
            c = np.rint(obs[0] * scale).astype(int)
            together = bool((c[0:2] == c[2:4]).all())
            coloc += together
            if together and met is None:
                met = t
        if met is not None:
            meet_eps += 1
            meet_ts.append(met)
    return {
        "i": i, "j": j,
        "catches": catches / episodes,
        "maul": maul.sum() / episodes, "forage": forage.sum() / episodes,
        "meet": 100.0 * meet_eps / episodes,
        "meet_t": float(np.mean(meet_ts)) if meet_ts else float("nan"),
        "coloc": 100.0 * coloc / steps,
        "stand_A": 100.0 * stand[0] / steps, "stand_B": 100.0 * stand[1] / steps,
    }


# --------------------------------------- main

def short(name):
    """separate_hyst0.2_a0-0_maul-5_s10002 -> hyst0.2 a0 s2"""
    n = re.sub(r"^separate_", "", name)
    m = re.match(r"(?:(.*?)_)?a([\d.]+)-[\d.]+_maul-?[\d.]+_s\d*(\d)$", n)
    if not m:
        return n
    method = m.group(1) or "dqn"
    return f"{method} a{m.group(2).rstrip('0').rstrip('.') or '0'} s{m.group(3)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run folders (default: runs/*)")
    ap.add_argument("--filter", default=None, help="regex on the run name, e.g. 'hyst|noisyep'")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())

    cand = sorted(d.rstrip("/") for d in (args.runs or glob.glob("runs/*")) if os.path.exists(os.path.join(d, "model.pt")))
    if args.filter:
        cand = [d for d in cand if re.search(args.filter, os.path.basename(d))]
    with redirect_stdout(io.StringIO()):
        cargs = {d: load_args(d) for d in cand}
    runs = [d for d in cand if cargs[d]["mode"] == "separate"]
    if len(runs) < 2:
        sys.exit("need at least two runs trained with --mode separate")

    # only pair runs trained in the same environment configuration
    envkey = {d: json.dumps(env_kwargs(cargs[d]), sort_keys=True) for d in runs}
    jobs = [(i, j, args.episodes, args.seed)
            for i in range(len(runs)) for j in range(len(runs)) if envkey[runs[i]] == envkey[runs[j]]]
    print(f"{len(runs)} separate-network runs -> {len(jobs)} pairs x {args.episodes} episodes "
          f"on {args.jobs} processes ...", flush=True)
    with Pool(args.jobs, initializer=_init, initargs=(runs,)) as pool:
        res = pool.map(play, jobs, chunksize=4)

    names = [os.path.basename(r) for r in runs]
    cfg = [re.sub(r"_s\d+$", "", n) for n in names]
    selfp = {r["i"]: r for r in res if r["i"] == r["j"]}
    coop_run = {i: selfp[i]["catches"] > COOP for i in selfp}
    role_A = {i: "waiter" if selfp[i]["stand_A"] > WAITER else "walker" for i in selfp}
    role_B = {i: "waiter" if selfp[i]["stand_B"] > WAITER else "walker" for i in selfp}

    rows = []
    for r in res:
        i, j = r["i"], r["j"]
        rows.append({
            "A_from": names[i], "B_from": names[j],
            "kind": "self" if i == j else ("same_config" if cfg[i] == cfg[j] else "cross_config"),
            "A_selfplay_coop": coop_run[i], "B_selfplay_coop": coop_run[j],
            "A_role": role_A[i], "B_role": role_B[j],
            **{k: v for k, v in r.items() if k not in ("i", "j")},
        })
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "crossplay_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- summary text
    out = io.StringIO()
    def p(*a): print(*a, file=out)

    def stats(sel):
        if not sel:
            return "n=   0"
        c = np.array([r["catches"] for r in sel])
        return (f"n={len(sel):4d}  catches {c.mean():5.1f} ± {c.std():4.1f}   cooperative {100*np.mean(c > COOP):5.1f}%"
                f"   meet {np.mean([r['meet'] for r in sel]):5.1f}%   maul {np.mean([r['maul'] for r in sel]):5.2f}"
                f"   forage {np.mean([r['forage'] for r in sel]):5.1f}")

    both = [r for r in rows if r["A_selfplay_coop"] and r["B_selfplay_coop"]]
    p(f"CROSS-PLAY over pairs whose two source runs both cooperate in self-play "
      f"({sum(coop_run.values())} of {len(runs)} runs cooperate)\n")
    p("  self-play                    " + stats([r for r in both if r["kind"] == "self"]))
    p("  cross-play, same config      " + stats([r for r in both if r["kind"] == "same_config"]))
    p("  cross-play, different config " + stats([r for r in both if r["kind"] == "cross_config"]))
    p("  cross-play, all              " + stats([r for r in both if r["kind"] != "self"]))

    p("\nCROSS-PLAY BY ROLE (cooperative source runs, i != j)   A role x B role")
    for ra in ("walker", "waiter"):
        for rb in ("walker", "waiter"):
            p(f"  A {ra:6s} + B {rb:6s}         " + stats([r for r in both if r["kind"] != "self"
                                                          and r["A_role"] == ra and r["B_role"] == rb]))

    p("\nCROSS-PLAY BY TRAINING METHOD (cooperative source runs, i != j)   rows: A's method, cols: B's method, mean catches")
    meth = lambda n: re.sub(r"_a[\d.]+-[\d.]+_maul.*$", "", re.sub(r"^separate_", "", n)) or "dqn"
    meths = sorted({meth(names[i]) for i in selfp if coop_run[i]})
    p("  " + " " * 14 + "".join(f"{m[:9]:>10s}" for m in meths))
    for ma in meths:
        cells = []
        for mb in meths:
            sel = [r["catches"] for r in both if r["kind"] != "self" and meth(r["A_from"]) == ma and meth(r["B_from"]) == mb]
            cells.append(f"{np.mean(sel):10.1f}" if sel else f"{'-':>10s}")
        p(f"  {ma[:14]:14s}" + "".join(cells))

    idx = [i for i in range(len(runs)) if coop_run[i]]
    if idx:
        p("\nCATCH MATRIX (cooperative runs only): row = run providing agent A, column = run providing agent B")
        p("diagonal = self-play\n")
        p("      " + "".join(f"{'r'+str(k+1):>6s}" for k in range(len(idx))))
        cell = {(r["i"], r["j"]): r["catches"] for r in res}
        for k, i in enumerate(idx):
            p(f"{'r'+str(k+1):>6s}" + "".join(
                (f"{cell[(i, j)]:5.1f}*" if i == j else f"{cell[(i, j)]:6.1f}") if (i, j) in cell else f"{'-':>6s}"
                for j in idx))
        p("\nlegend (self-play catches, role of A / role of B):")
        for k, i in enumerate(idx):
            p(f"  r{k+1:<3d} {short(names[i]):24s} {selfp[i]['catches']:5.1f}   A {role_A[i]:6s}  B {role_B[i]}")

    text = out.getvalue()
    print("\n" + text)
    with open(os.path.join(args.out, "crossplay_summary.txt"), "w") as f:
        f.write(text)
    print(f"saved {args.out}/crossplay_pairs.csv and {args.out}/crossplay_summary.txt")


if __name__ == "__main__":
    main()
