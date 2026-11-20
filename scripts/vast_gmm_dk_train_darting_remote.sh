#!/bin/bash
# Runs on the Vast.ai instance. Protocol: rules_train_darting.md. Continuations from the seed-0 controllers (x 0.5): (a) AS alone, (b) AS + darting (gamma 1), 2 cells, 4 runs in parallel.
#   bash vast_gmm_dk_train_darting_remote.sh smoke   -> 3 epochs of (b) at (32,8,6), W&B off (GPU / CUDA-graph compatibility of the new matcher)
#   bash vast_gmm_dk_train_darting_remote.sh run     -> EPOCHS (default 500) epochs, 4 runs in parallel
# Start files: /workspace/asbs/cont/<start run>/{controller_start.pt,start_samples.pt}. Logs: /workspace/asbs/results/gmmdk_cont_<a|b>_d<d>_K8_D<Delta>_s0/stdout.log ; status in /workspace/status.txt
set -e
PHASE=${1:?smoke or run}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "running" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
EPOCHS=${EPOCHS:-500}
EXTRA_COMMON="num_epochs=${EPOCHS} eval_freq=${EVAL_FREQ:-50} num_eval_samples=10000 +dump_eval_samples=true dump_final=true dump_n_samples=10000 seed=0 buffer_on_device=true fast_train=true fast_sampling=true project=gmm_dk_cont use_wandb=${WANDB:-true} +code_commit=${COMMIT} init_buffer_key=buffer_x1 ~lancher"
pids=()
run() { arm=$1; d=$2; D=$3; sig=$4; start=$5; shift 5
  exp=gmm_dk_as; [ "$arm" = "b" ] && exp=gmm_dk_as_darting
  N=gmmdk_cont_${arm}_d${d}_K8_D${D}_s0; DIR=results/$N; mkdir -p $DIR
  python train.py experiment=$exp dim=$d K=8 delta=$D sigma_max=$sig exp_name=$N init_controller=cont/$start/controller_start.pt init_buffer=cont/$start/start_samples.pt hydra.run.dir=$DIR ${EXTRA_COMMON} "$@" > $DIR/stdout.log 2>&1 & pids+=($!); }
if [ "$PHASE" = "smoke" ]; then
  EPOCHS=3; WANDB=false
  run b 32 6 1.25 gmmdk_d32_K8_D6_s0_sx0.5 num_epochs=3 eval_freq=3 use_wandb=false
else
  run a 8 8 2.236068 gmmdk_d8_K8_D8_s0_sx0.5; run b 8 8 2.236068 gmmdk_d8_K8_D8_s0_sx0.5
  run a 32 6 1.25 gmmdk_d32_K8_D6_s0_sx0.5; run b 32 6 1.25 gmmdk_d32_K8_D6_s0_sx0.5
fi
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail = 0 ]; then echo "DONE" > /workspace/status.txt; else echo "FAILED training" > /workspace/status.txt; fi
