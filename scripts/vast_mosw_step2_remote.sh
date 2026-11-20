#!/bin/bash
# Runs on the Vast.ai instance. rules_mos_step1.md amendment 1: one AS run on MoS-w d = 50 (sigma_max 23.2, 300 epochs, seed 0, controller snapshots at epochs 100/200/300), then scripts/mosw_step2_eval.py.
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
D=results/mosw_s23.2; mkdir -p $D
python train.py experiment=mosw_as sigma_max=23.2 dim=50 K=10 num_epochs=300 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 seed=0 '+save_controller_epochs=[100,200,300]' \
  buffer_on_device=true fast_train=true fast_sampling=true project=mos_pilot use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=mosw_s23.2 hydra.run.dir=$D '~lancher' > $D/stdout.log 2>&1 || { echo "FAILED training" > /workspace/status.txt; exit 1; }
echo "evaluating" > /workspace/status.txt
python scripts/mosw_step2_eval.py $D /workspace/mosw_results/step2.json --n_paths 20000 --budget_s ${EVAL_BUDGET:-420} > /workspace/step2_eval.log 2>&1 || { echo "FAILED eval" > /workspace/status.txt; exit 1; }
echo "DONE" > /workspace/status.txt
