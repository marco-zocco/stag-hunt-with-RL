#!/bin/bash
# SLURM job array: one training run per array task.
#
# 1) Generate the list of runs (one command per line), e.g. the learning-rule
#    variants at the default rewards (maul -5):
#      python sweep.py --alphas 0 0.5 --mauls -5 --print-commands -- --hyst-beta 0.2          >  commands.txt
#      python sweep.py --alphas 0 0.5 --mauls -5 --print-commands -- --noisy                  >> commands.txt
#      python sweep.py --alphas 0 0.5 --mauls -5 --print-commands -- --noisy --noise-resample episode >> commands.txt
# 2) Set --array to 1-<number of lines>:   wc -l < commands.txt
# 3) Submit from the repo root:            sbatch slurm_array.sh
# 4) When all tasks are done:              python aggregate.py
#
# Adapt the partition / account lines to your cluster.

#SBATCH --job-name=staghunt
#SBATCH --array=1-18
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
mkdir -p slurm_logs
source .venv/bin/activate

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" commands.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${CMD}"
eval "${CMD}"
