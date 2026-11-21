#!/bin/bash
# Runs on the Vast.ai instance (rules_mosw_step3.md). Two AS trainings in parallel (MoS-w d = 50 and d = 10, sigma_max 23.2, 1500 epochs, seed 0, controller snapshots every 250 epochs),
# and one evaluation worker per problem that waits for each snapshot, computes Q, then the comparisons if Q passes. Env: EVAL_DEADLINE_REL = seconds after launch at which no new evaluation starts.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
DEADLINE=$(( $(date +%s) + ${EVAL_DEADLINE_REL:-2100} ))
EPS="[250,500,750,1000,1250,1500]"
mkdir -p /workspace/step3_results
for D in 50 10; do
  R=results/mosw3_d$D; mkdir -p $R
  ( python train.py experiment=mosw_as sigma_max=23.2 dim=$D K=10 num_epochs=1500 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 seed=0 "+save_controller_epochs=$EPS" \
      buffer_on_device=true fast_train=true fast_sampling=true project=mos_step3 use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=mosw3_d${D}_s23.2 hydra.run.dir=$R '~lancher' > $R/stdout.log 2>&1 \
      || echo "FAILED training d=$D" >> /workspace/status.txt; echo "trained d=$D" >> /workspace/progress.txt ) &
done
sleep 45   # let the config.yaml files appear before the workers read them
for D in 50 10; do
  R=results/mosw3_d$D
  ( python scripts/mosw_step3_eval.py $R /workspace/step3_results/d$D/out.json --d $D --wait --deadline $DEADLINE > /workspace/step3_eval_d$D.log 2>&1 \
      || echo "FAILED eval d=$D" >> /workspace/status.txt; echo "evaluated d=$D" >> /workspace/progress.txt ) &
done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
