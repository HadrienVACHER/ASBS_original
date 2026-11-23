#!/bin/bash
# Runs on the Vast.ai instance (rules_train_weightfree.md). 9 continuations of 500 epochs (seed 0): arms a (AS), t (tempered IS alpha 0.5), d (darting gamma 1.75) x cells d8 (8,8,8) x0.5, d32 (32,8,6) x0.5,
# d128 (128,8,6) x0.75 from epoch 1500. At most 6 runs at once: wave 1 = d128 + d8, wave 2 = d32. Start files in /workspace/asbs/results/wf_start/<cell>/. Env: NO_NEW_RUN_REL = seconds after launch
# after which no new run is started.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
STOPNEW=$(( $(date +%s) + ${NO_NEW_RUN_REL:-3300} ))
COMMON="num_epochs=500 eval_freq=50 num_eval_samples=10000 +dump_eval_samples=true dump_final=true dump_n_samples=10000 seed=0 buffer_on_device=true fast_train=true fast_sampling=true project=train_weightfree use_wandb=${WANDB:-true} +code_commit=${COMMIT} init_buffer_key=buffer_x1 +save_controller_epochs=[100,200,300,400,500] ~lancher"
declare -A CELL=( [d8]="dim=8 K=8 delta=8 sigma_max=2.236068" [d32]="dim=32 K=8 delta=6 sigma_max=1.25" [d128]="dim=128 K=8 delta=6 sigma_max=1.602001" )
run() { arm=$1; c=$2
  case $arm in a) X="experiment=gmm_dk_as";; t) X="experiment=gmm_dk_as_tempered_is adjoint_matcher.alpha=0.5";; d) X="experiment=gmm_dk_as_darting adjoint_matcher.gamma=1.75";; esac
  N=wf_${arm}_${c}; DIR=results/$N; mkdir -p $DIR
  if [ $(date +%s) -gt $STOPNEW ]; then echo "SKIPPED $N (deadline)" >> /workspace/progress.txt; return; fi
  ( python train.py $X ${CELL[$c]} init_controller=results/wf_start/$c/controller_start.pt init_buffer=results/wf_start/$c/start_samples.pt exp_name=$N hydra.run.dir=$DIR $COMMON > $DIR/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; echo "trained $N $(date +%s)" >> /workspace/progress.txt ) &
}
for c in d128 d8; do for arm in a t d; do run $arm $c; done; done
wait
for arm in a t d; do run $arm d32; done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
