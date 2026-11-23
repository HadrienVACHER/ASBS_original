#!/bin/bash
# Runs on the Vast.ai instance (rules_selftrain_dartbank.md). 7 continuations of 500 epochs (seed 0), at most 6 at once (a free slot starts the next run):
#   s_d128 (128,8,6) x0.75 from ep1500 [arm (s) only: (a) and (t) are reused from step B], s_k8d32 (32,8,8) x0.5, s_mosw10 MoS-w d=10 from the ep250 snapshot,
#   t_k8d32, t_mosw10 (tempered IS alpha 0.5), a_k8d32, a_mosw10 (AS). Start files in /workspace/asbs/results/wf_start/<cell>/.
# Env: NO_NEW_RUN_REL = seconds after launch after which no new run is started.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
STOPNEW=$(( $(date +%s) + ${NO_NEW_RUN_REL:-2400} ))
COMMON="num_epochs=500 eval_freq=50 num_eval_samples=10000 +dump_eval_samples=true dump_final=true dump_n_samples=10000 seed=0 buffer_on_device=true fast_train=true fast_sampling=true project=selftrain_dartbank use_wandb=${WANDB:-true} +code_commit=${COMMIT} init_buffer_key=buffer_x1 +save_controller_epochs=[100,200,300,400,500] ~lancher"
declare -A CELL=( [k8d32]="dim=32 K=8 delta=8 sigma_max=1.414214" [mosw10]="dim=10 K=10 sigma_max=23.2" [d128]="dim=128 K=8 delta=6 sigma_max=1.602001" )
declare -A EXPN=( [a_k8d32]="gmm_dk_as" [t_k8d32]="gmm_dk_as_tempered_is adjoint_matcher.alpha=0.5" [s_k8d32]="gmm_dk_as_dartbank"
                  [a_mosw10]="mosw_as" [t_mosw10]="mosw_as_tempered_is adjoint_matcher.alpha=0.5" [s_mosw10]="mosw_as_dartbank" [s_d128]="gmm_dk_as_dartbank" )
JOBS=(s_d128 s_k8d32 s_mosw10 t_k8d32 t_mosw10 a_k8d32 a_mosw10)
run() { key=$1; arm=${key%%_*}; c=${key#*_}; N=sd_${arm}_${c}; DIR=results/$N; mkdir -p $DIR
  if [ $(date +%s) -gt $STOPNEW ]; then echo "SKIPPED $N (deadline)" >> /workspace/progress.txt; return; fi
  ( python train.py experiment=${EXPN[$key]} ${CELL[$c]} init_controller=results/wf_start/$c/controller_start.pt init_buffer=results/wf_start/$c/start_samples.pt exp_name=$N hydra.run.dir=$DIR $COMMON > $DIR/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; echo "trained $N $(date +%s)" >> /workspace/progress.txt ) &
}
for key in "${JOBS[@]}"; do
  while [ $(jobs -rp | wc -l) -ge 6 ]; do sleep 5; done
  run $key; sleep 1
done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
