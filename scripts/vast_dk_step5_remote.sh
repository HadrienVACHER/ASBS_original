#!/bin/bash
# Runs on the Vast.ai instance (rules_dk_step5.md). 4 AS trainings in parallel on MoS-dk (d, K, Delta) = (32, 8, 6) and (128, 8, 6), sigma_max = mult x 2 sqrt(1 + Delta^2/(2d)), mult in {0.5, 0.75}, seed 0, 1500 epochs, snapshots every 250 epochs,
# one Q worker per run; then, per target, the sigma_max rule and the comparison. In the background from the start: basin truth of the 4 targets, and the S = 40 supplement for the two step-4 GMM-dk controllers (fixed, no retraining).
# Env: EVAL_DEADLINE_REL = seconds after launch at which no new evaluation starts.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
DEADLINE=$(( $(date +%s) + ${EVAL_DEADLINE_REL:-4800} ))
EPS="[250,500,750,1000,1250,1500]"
OUT=/workspace/step5_results; mkdir -p $OUT
# step-4 GMM-dk controllers (chosen by the step-4 rule), same layout as at training time
for d in 128 256; do N=gmmdk4_d${d}_sx0.5; mkdir -p results/$N; cp figures/gmmdk_step4/controllers/${N}_ep1500.pt results/$N/controller_ep1500.pt; cp figures/gmmdk_step4/configs/$N.config.yaml results/$N/config.yaml; done
declare -A NAMES
for d in 32 128; do for m in 0.5 0.75; do
  sig=$(python3 -c "import math; print(round($m*2*math.sqrt(1+6.0**2/(2*$d)),6))")
  N=mosdk5_d${d}_sx${m}; DIR=results/$N; mkdir -p $DIR; NAMES[$d]="${NAMES[$d]:-} $N"
  ( python train.py experiment=mosdk_as dim=$d K=8 delta=6.0 sigma_max=$sig seed=0 num_epochs=1500 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 "+save_controller_epochs=$EPS" \
      buffer_on_device=true fast_train=true fast_sampling=true project=dk_step5 use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=$N hydra.run.dir=$DIR '~lancher' > $DIR/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; echo "trained $N" >> /workspace/progress.txt ) &
done; done
( for spec in "mos 32" "mos 128" "gmm 128" "gmm 256"; do set -- $spec
    python scripts/dk_step5_eval.py truth $OUT --fam $1 --d $2 >> /workspace/step5_truth.log 2>&1 || echo "FAILED truth $spec" >> /workspace/status.txt
  done; echo "truth done" >> /workspace/progress.txt ) &
( for d in 128 256; do
    python scripts/dk_step5_eval.py cmp $OUT --fam gmm --d $d --fixed gmmdk4_d${d}_sx0.5:1500:results/gmmdk4_d${d}_sx0.5 --deadline $DEADLINE > /workspace/step5_supp_gmm_d$d.log 2>&1 || echo "FAILED supp gmm d=$d" >> /workspace/status.txt
  done; echo "gmm supplement done" >> /workspace/progress.txt ) &
sleep 45   # let the config.yaml files appear before the workers read them
for d in 32 128; do for N in ${NAMES[$d]}; do
  ( python scripts/dk_step5_eval.py q results/$N $OUT --fam mos --d $d --wait --deadline $DEADLINE > /workspace/step5_q_$N.log 2>&1 \
      || echo "FAILED q $N" >> /workspace/status.txt; echo "Q done $N" >> /workspace/progress.txt ) &
done; done
wait
for d in 32 128; do
  set -- ${NAMES[$d]}
  ( python scripts/dk_step5_eval.py cmp $OUT --fam mos --d $d --runs $1 $2 --rdirs results/$1 results/$2 --deadline $DEADLINE > /workspace/step5_cmp_mos_d$d.log 2>&1 \
      || echo "FAILED cmp mos d=$d" >> /workspace/status.txt; echo "cmp done mos d=$d" >> /workspace/progress.txt ) &
done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
