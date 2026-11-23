#!/bin/bash
# Runs on the Vast.ai instance (rules_oracle_buffer.md). ONE run: oracle buffer at (128, 8, 6) x 0.75 (sigma_max 1.602001), 500 epochs from the step-B start controller, seed 0. NOT LAUNCHED without the user's decision.
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
N=sd_o_d128; DIR=results/$N; mkdir -p $DIR
python train.py experiment=gmm_dk_as_oracle dim=128 K=8 delta=6 sigma_max=1.602001 init_controller=results/wf_start/d128/controller_start.pt init_buffer=results/wf_start/d128/start_samples.pt init_buffer_key=buffer_x1 \
  num_epochs=500 eval_freq=50 num_eval_samples=10000 +dump_eval_samples=true dump_final=true dump_n_samples=10000 seed=0 buffer_on_device=true fast_train=true fast_sampling=true project=selftrain_dartbank use_wandb=${WANDB:-true} +code_commit=${COMMIT} \
  +save_controller_epochs=[100,200,300,400,500] exp_name=$N hydra.run.dir=$DIR '~lancher' > $DIR/stdout.log 2>&1 || echo "FAILED training $N" >> /workspace/status.txt
echo "trained $N $(date +%s)" >> /workspace/progress.txt
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
