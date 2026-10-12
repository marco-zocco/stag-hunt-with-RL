# Stag Hunt with RL

Two independently learning DQN agents in the *Hunt* game of [Gymnasium-Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt).
Course project for *Autonomous and Adaptive Systems* (University of Bologna, 2025/26).

The study asks whether two selfish agents can learn to hunt the stag together at the environment's default rewards,
and under which conditions this cooperation survives (punishment and temptation size, grid size, a randomly moving
stag, unseen partners). Plain DQN almost never cooperates; hysteretic updates and NoisyNet exploration make
cooperation emerge reliably, also with fully separate networks. Full results and discussion are in the report;
the numbers behind them are in `results/`.

## Environment

`stag_env.py` wraps `StagHunt-Hunt-v0` with coordinate observations and both agents controlled.
The default parameters are listed below; all of them can be changed with `train.py` flags
(`--grid-size`, `--max-timesteps`, `--stag-reward`, `--forage-reward`, `--mauling-punishment`,
`--run-away-after-maul`, `--no-stag-follows`).

| setting | default |
|---|---|
| grid | 5×5, agents start in the two top corners |
| episode | 100 steps (no terminal states) |
| rewards | joint catch +5 each, hunting alone −5 (mauling), plant +1 |
| stag | moves one cell towards the nearest agent every step |
| observation | (x, y) of A, B, stag and 2 plants, normalised to [0, 1]; agent B sees itself first |

## What is implemented

- **DQN** with Double-DQN targets, target network, uniform replay, Huber loss; MLP 64-64.
- **Networks**: `--mode separate` (two independent learners) or `--mode shared` (one network with a one-hot agent ID).
- **Prosocial reward**: `--alpha` (agent i learns from (1 − α)·r_i + α·r_j; α = 0 is selfish).
- **Hysteretic DQN**: `--hyst-beta 0.2` (negative TD errors weighted by β < 1).
- **NoisyNet exploration**: `--noisy`, with `--noise-resample step` or `episode`.
- **Evaluation**: greedy policy on 200 fixed evaluation seeds at the end of training.

## Repository

| file | purpose |
|---|---|
| `stag_env.py` | wrapper around the Hunt environment: seeding, normalised observations, event counting |
| `model.py` | Q-network (MLP), optional NoisyNet layers |
| `replay.py` | uniform replay buffer |
| `dqn.py` | `DQNLearner` (Double DQN, hysteretic loss, NoisyNet handling) and `Team` (shared / separate) |
| `evaluation.py` | fixed-seed greedy evaluation protocol |
| `train.py` | one training run; all options are command-line flags |
| `sweep.py` | builds (or runs) a grid of configurations × seeds |
| `aggregate.py` | per-configuration summary across seeds → `results/summary.csv` |
| `scripted_policies.py`, `baselines.py` | scripted references (random, always stag, always plant, joint planner) |
| `analyze_behaviour.py` | how trained agents behave (meeting, co-location, waiting, catch configuration), ASCII replays |
| `crossplay.py` | pairs agents from different separate-network runs |
| `colab/staghunt_colab.ipynb` | notebook used to run all experiments on Google Colab |
| `experiments/run_list.txt` | the exact command of every run in `runs/` |
| `runs/` | one folder per run: `config.json`, `eval_log.csv`, `final_eval.json`, `model.pt` |
| `results/` | summaries computed from `runs/` |

Run names encode the configuration, e.g. `separate_hyst0.2_a0-0_maul-5_nofollow_steps1000k_s10000`:
network mode, learning rule (none = plain DQN), α of A and B, mauling penalty, non-default environment settings
(`grid7`, `forage2`, `nofollow`, …), training length if not 300k, seed.

## Installation

Python 3.10+.

```
git clone https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git
git clone https://github.com/marco-zocco/stag-hunt-with-RL.git
cd stag-hunt-with-RL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../Gymnasium-Stag-Hunt
```

## Usage

```
# one run (defaults: shared network, plain DQN, alpha 0, 300k steps)
python train.py --mode separate --hyst-beta 0.2 --seed 10000
python train.py --mode separate --noisy --noise-resample episode --no-stag-follows --total-steps 1000000

# several runs: print the commands, or run them locally in parallel
python sweep.py --modes separate --alphas 0 --mauls -5 --seeds 10000 10001 10002 --print-commands -- --hyst-beta 0.2
python sweep.py --modes separate --alphas 0 --mauls -1 -5 --jobs 4 -- --hyst-beta 0.2

# summaries and analysis (read runs/, write results/)
python aggregate.py
python baselines.py                      # also: --no-stag-follows, --grid-size 9 9, ...
python analyze_behaviour.py --episodes 200
python analyze_behaviour.py --show runs/<run_name> --show-steps 20
python crossplay.py --episodes 200
```

## Reproducing

Every run in `runs/` is listed in `experiments/run_list.txt`. Training is deterministic given the seed
(same code and library versions give identical results). The full set was run on Google Colab with
`colab/staghunt_colab.ipynb`, one run per CPU core; the networks are small MLPs, so no GPU is used.
`aggregate.py`, `baselines.py`, `analyze_behaviour.py` and `crossplay.py` regenerate everything in `results/`.

## References

Environment: G. Franceschelli, [Gymnasium-Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt).
Methods: DQN, Double DQN, hysteretic DQN, NoisyNet and prosocial rewards; full references are in the report.
