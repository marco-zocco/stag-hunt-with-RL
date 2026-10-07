"""
Run train.py over a grid of configurations x seeds

Each run is a separate process (clean random state, one CPU thread each), with
its output saved to runs/<name>/stdout.log
"""

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import train


def build_runs(args, extra):
    runs = []
    for mode, alpha, maul, seed in itertools.product(args.modes, args.alphas, args.mauls, args.seeds):
        train_args = [
            "--mode", mode,
            "--alpha", str(alpha),
            "--mauling-punishment", str(maul),
            "--seed", str(seed),
            "--out", args.out,
            *extra,
        ]
        name = train.parse_args(train_args).name  # same naming as train.py
        runs.append((name, [sys.executable, "train.py", *train_args]))
    return runs


def execute(name, cmd, out_dir):
    run_dir = os.path.join(out_dir, name)
    if os.path.exists(os.path.join(run_dir, "final_eval.json")):
        return name, "skipped (already done)"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "stdout.log"), "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    return name, "done" if code == 0 else f"FAILED (exit {code}, see stdout.log)"


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["shared"], choices=["shared", "separate"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--mauls", nargs="+", type=float, default=[-1.0, -5.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002])
    p.add_argument("--jobs", type=int, default=1, help="runs in parallel (local only)")
    p.add_argument("--out", default="runs")
    p.add_argument("--print-commands", action="store_true", help="print commands instead of running")
    args = p.parse_args(argv)

    runs = build_runs(args, extra)
    if args.print_commands:
        for _, cmd in runs:
            print(" ".join(shlex.quote(c) for c in ["python", *cmd[1:]]))
        return

    print(f"{len(runs)} runs, {args.jobs} in parallel", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, name, cmd, args.out) for name, cmd in runs]
        for f in futures:
            name, status = f.result()
            print(f"  {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
