#!/bin/bash
#SBATCH -J lightsail-stage2-fmm
#SBATCH -p hcpu1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -a 0-2                       # 3 seeds: 42, 123, 456
#SBATCH -o slurm_logs/stage2_fmm-%A_%a.out

# Multi-seed Stage 2 BO with full-wave FMM stabilization proxy (P5).
#
# Submit:   sbatch scripts/sbatch_stage2_fmm.sh
# Status:   squeue -u $USER
# Logs:     tail -f slurm_logs/stage2_fmm-<JOBID>_<TASK>.out
#
# The job array index maps to a seed via the SEEDS list below.

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd /home/ejlee0315/Lightsail
source .venv/bin/activate

mkdir -p slurm_logs

python scripts/run_experiment.py \
    --config configs/stage2_fmm.yaml \
    --n-iter 80 \
    --seed "$SEED" \
    --name "stage2_fmm_s${SEED}"
