#!/bin/bash
# Runs on the Vast.ai instance. Protocol: predictions_dk.md. AS only, (d, K, Delta) study, fast options.
#   bash vast_gmm_dk_remote.sh pilot   -> (8,2,8) (8,8,8) (32,2,8) (32,8,8) (32,8,4), seed 0, 1500 epochs, in parallel
#   bash vast_gmm_dk_remote.sh sweep   -> addendum 1: 15 sweep runs + 4 sigma_max ablation runs (19 in parallel)
#   bash vast_gmm_dk_remote.sh smoke   -> 2 epochs of (32,8,8)
# Logs: /workspace/asbs/results/gmmdk_d<d>_K<K>_D<Delta>_s<seed>/stdout.log ; status in /workspace/status.txt
set -e
PHASE=${1:?pilot or smoke}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "setup" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
COMMON="experiment=gmm_dk_as project=gmm_dk dump_final=true dump_n_samples=10000 +dump_eval_samples=true buffer_on_device=true fast_train=true fast_sampling=true use_wandb=true +code_commit=${COMMIT} ~lancher"
pids=()
run() { d=$1; K=$2; D=$3; s=$4; shift 4; N=gmmdk_d${d}_K${K}_D${D}_s${s}${SUF}; DIR=results/$N; mkdir -p $DIR
  python train.py ${COMMON} dim=$d K=$K delta=$D seed=$s exp_name=$N hydra.run.dir=$DIR "$@" > $DIR/stdout.log 2>&1 & pids+=($!); }
echo "running" > /workspace/status.txt
if [ "$PHASE" = "smoke" ]; then
  run 32 8 8 0 num_epochs=2 eval_freq=1 use_wandb=false
elif [ "$PHASE" = "sweep" ]; then
  # Addendum 1 of predictions_dk.md: Delta sweep (the 5 pilot cells already exist), 15 new runs + 4 sigma_max ablation runs
  for D in 4 5 6 7; do run 8 2 $D 0; run 8 8 $D 0; run 32 2 $D 0; done
  for D in 5 6 7; do run 32 8 $D 0; done
  SUF=_sx0.5 run 8 8 8 0 sigma_max=2.236068; SUF=_sx2 run 8 8 8 0 sigma_max=8.944272
  SUF=_sx0.5 run 32 8 6 0 sigma_max=1.25; SUF=_sx2 run 32 8 6 0 sigma_max=5.0
else
  run 8 2 8 0; run 8 8 8 0; run 32 2 8 0; run 32 8 8 0; run 32 8 4 0
fi
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail = 0 ]; then echo "DONE" > /workspace/status.txt; else echo "FAILED training" > /workspace/status.txt; fi
