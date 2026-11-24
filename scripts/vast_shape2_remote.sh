#!/bin/bash
# Runs on the Vast.ai instance (rules_shape_correction_2.md). TWO runs (start (ii) only, amendment 2026-11-24) in parallel, buffer (M') = mean shift 1.0 (exact masses and covariance, fixed direction per mode), (128, 8, 6) x 0.75, 500 epochs, seed 0:
#   sd_Mp_{as,dsm}_i_d128  : start (i) = the common step-B controller ; sd_Mp_{as,dsm}_ii_d128 : start (ii) = re-initialised controller (random weights; no init_controller).
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
run() { EXP=$1; START=$2; N=$3; DIR=results/$N; mkdir -p $DIR
  if [ $START = i ]; then IC="init_controller=results/wf_start/d128/controller_start.pt"; else IC=""; fi
  ( python train.py experiment=$EXP adjoint_matcher.mode=M adjoint_matcher.shift=1.0 dim=128 K=8 delta=6 sigma_max=1.602001 $IC init_buffer=results/wf_start/d128/start_samples.pt init_buffer_key=buffer_x1 \
      num_epochs=500 eval_freq=50 num_eval_samples=10000 +dump_eval_samples=true dump_final=true dump_n_samples=10000 seed=0 buffer_on_device=true fast_train=true fast_sampling=true project=selftrain_dartbank use_wandb=${WANDB:-true} +code_commit=${COMMIT} \
      +save_controller_epochs=[100,200,300,400,500] exp_name=$N hydra.run.dir=$DIR '~lancher' > $DIR/stdout.log 2>&1 || echo "FAILED training $N" >> /workspace/status.txt
    echo "trained $N $(date +%s)" >> /workspace/progress.txt ) &
}
# amendment 2026-11-24: start (ii) only (start (i) runs removed)
run gmm_dk_as_shape_as ii sd_Mp_as_ii_d128; sleep 1
run gmm_dk_as_shape_dsm ii sd_Mp_dsm_ii_d128
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
