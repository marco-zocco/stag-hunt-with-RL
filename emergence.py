"""
When does hunting emerge in the random-stag runs (stag_follows=False)?

For every hysteretic random-stag run, finds the first mid-training evaluation
with more than 8 catches per 100 steps (same threshold as aggregate.py) and
compares it with the step at which epsilon reaches its minimum
(eps_frac * total_steps). Resolution = --eval-every of the run (10k steps).

Usage (from the repo root):
  python emergence.py                      # default: runs/*hyst0.2*nofollow*
  python emergence.py --pattern 'runs/*nofollow*'

Writes results/emergence.csv.
"""

import argparse
import csv
import glob
import json
import os

COOP = 8.0  # catches per 100 steps; 8 x 5 = 40 = return of pure foraging


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="runs/*hyst0.2*nofollow*", help="glob of the run folders")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = []
    for d in sorted(glob.glob(args.pattern)):
        a = json.load(open(os.path.join(d, "config.json")))["args"]
        rows = list(csv.DictReader(open(os.path.join(d, "eval_log.csv"))))
        eps_min = int(a["eps_frac"] * a["total_steps"])
        first = next((int(r["step"]) for r in rows if float(r["stag_mean"]) > COOP), None)
        final = float(rows[-1]["stag_mean"])
        out.append(dict(run=os.path.basename(d), mode=a["mode"], total_steps=a["total_steps"], eps_frac=a["eps_frac"],
                        eps_min_step=eps_min, first_hunt_step=first,
                        steps_after_eps_min="" if first is None else first - eps_min,
                        final_catches=round(final, 2)))
        print(f"{out[-1]['run']:70s} eps min {eps_min:>8} | first >{COOP:g} catches: {str(first):>8} | final {final:5.1f}")
    if not out:
        raise SystemExit(f"no runs match {args.pattern}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "emergence.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"saved {path}, {len(out)} runs")


if __name__ == "__main__":
    main()
