#!/bin/bash
# Runs on the Vast.ai instance (rules_gmmdk_step4.md). 4 AS trainings in parallel: GMM-dk (d, K, Delta) = (128, 8, 6) and (256, 8, 6), sigma_max = mult x 2 sqrt(1 + Delta^2/(2d)), mult in {0.5, 0.75}, seed 0, 1500 epochs,
# controller snapshots every 250 epochs; one Q worker per run (waits for each snapshot); then, per dimension, the sigma_max rule and the comparison on the chosen controller.
# Env: EVAL_DEADLINE_REL = seconds after launch at which no new evaluation starts.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
DEADLINE=$(( $(date +%s) + ${EVAL_DEADLINE_REL:-3000} ))
EPS="[250,500,750,1000,1250,1500]"
OUT=/workspace/step4_results; mkdir -p $OUT
declare -A NAMES
for d in 128 256; do for m in 0.5 0.75; do
  sig=$(python3 -c "import math; print(round($m*2*math.sqrt(1+6.0**2/(2*$d)),6))")
  N=gmmdk4_d${d}_sx${m}; DIR=results/$N; mkdir -p $DIR; NAMES[$d]="${NAMES[$d]:-} $N"
  ( python train.py experiment=gmm_dk_as dim=$d K=8 delta=6.0 sigma_max=$sig seed=0 num_epochs=1500 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 "+save_controller_epochs=$EPS" \
      buffer_on_device=true fast_train=true fast_sampling=true project=gmm_dk_step4 use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=$N hydra.run.dir=$DIR '~lancher' > $DIR/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; echo "trained $N" >> /workspace/progress.txt ) &
done; done
sleep 45   # let the config.yaml files appear before the workers read them
for d in 128 256; do for N in ${NAMES[$d]}; do
  ( python scripts/gmmdk_step4_eval.py q results/$N $OUT --d $d --wait --deadline $DEADLINE > /workspace/step4_q_$N.log 2>&1 \
      || echo "FAILED q $N" >> /workspace/status.txt; echo "Q done $N" >> /workspace/progress.txt ) &
done; done
wait
for d in 128 256; do
  set -- ${NAMES[$d]}
  ( python scripts/gmmdk_step4_eval.py cmp $OUT --d $d --runs $1 $2 --rdirs results/$1 results/$2 --deadline $DEADLINE > /workspace/step4_cmp_d$d.log 2>&1 \
      || echo "FAILED cmp d=$d" >> /workspace/status.txt; echo "cmp done d=$d" >> /workspace/progress.txt ) &
done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
