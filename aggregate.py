"""
Aggregate results from completed runs across seeds.

Usage:
    python aggregate.py --runs runs --out results/summary.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from train import env_label, method_label  # same tags as the run names

KEYS = ["stag", "maul_A", "maul_B", "forage_A", "forage_B", "return_A", "return_B"]
COOP_THRESHOLD = 8.0  # catches per 100 steps needed to out-earn foraging


def load_runs(runs_dir):
    runs = []
    for final_path in sorted(glob.glob(os.path.join(runs_dir, "*", "final_eval.json"))):
        run_dir = os.path.dirname(final_path)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(final_path) as f:
            final = json.load(f)["results"]
        a, env = cfg["args"], cfg["env_config"]
        env_tag = env_label({**env, "grid_size": list(env.get("grid_size", [5, 5]))})
        group = (
            a["mode"], method_label(a), a["alpha_a"], a["alpha_b"], env["mauling_punishment"],
            env["stag_reward"], env["forage_reward"], env["run_away_after_maul"], a["total_steps"],
            env_tag,
        )
        runs.append((group, a["seed"], {k: final[k][0] for k in KEYS}))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/summary.csv")
    args = p.parse_args()

    groups = defaultdict(list)
    for group, seed, metrics in load_runs(args.runs):
        groups[group].append((seed, metrics))
    if not groups:
        print(f"no finished runs found in {args.runs}/")
        return

    rows = []
    for group in sorted(groups):
        mode, method, aa, ab, maul, stag_r, forage_r, run_away, steps, env_tag = group
        seeds = sorted(groups[group])
        row = {
            "mode": mode, "method": method, "alpha_A": aa, "alpha_B": ab, "maul": maul,
            "stag_reward": stag_r, "forage_reward": forage_r, "run_away": run_away,
            "steps": steps, "env": env_tag.lstrip("_") or "default", "n_seeds": len(seeds),
            "coop_seeds": sum(m["stag"] > COOP_THRESHOLD for _, m in seeds),
            "stag_per_seed": " ".join(f"{m['stag']:.1f}" for _, m in seeds),
        }
        for k in KEYS:
            vals = np.array([m[k] for _, m in seeds])
            row[f"{k}_mean"], row[f"{k}_std"] = round(float(vals.mean()), 2), round(float(vals.std()), 2)
        rows.append(row)

    print(f"{'config':<58}{'seeds':>6}{'coop':>6}{'catches':>16}{'return A':>17}{'return B':>17}   catches per seed")
    for r in rows:
        cfg = f"{r['mode']} {r['method']} a={r['alpha_A']:g}/{r['alpha_B']:g} maul={r['maul']:g}"
        if r["env"] != "default":
            cfg += " " + r["env"].replace("_", " ")
        print(
            f"{cfg:<58}{r['n_seeds']:>6}{r['coop_seeds']:>6}"
            f"{r['stag_mean']:>9.2f} ±{r['stag_std']:>5.2f}"
            f"{r['return_A_mean']:>10.1f} ±{r['return_A_std']:>5.1f}"
            f"{r['return_B_mean']:>10.1f} ±{r['return_B_std']:>5.1f}   {r['stag_per_seed']}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
