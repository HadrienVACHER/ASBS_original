#!/bin/bash
# Runs on the Vast.ai instance (rules_mosw_ns.md). 4 trainings in parallel: MoS-w d in {50, 10}, sigma_max 23.2, seed 0, 1500 epochs, arms (b) damping eta = 0.5 and (c) Newton-Stein h = 1.5,
# snapshots every 250 epochs. The reused AS runs (arm (a), step 3) are uploaded in results/mns_d{50,10}_as. One Q-only evaluation worker per run (scripts/mosw_step3_eval.py --q_only).
# Env: EVAL_DEADLINE_REL = seconds after launch at which no new evaluation starts.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
DEADLINE=$(( $(date +%s) + ${EVAL_DEADLINE_REL:-2160} ))
EPS="[250,500,750,1000,1250,1500]"
OUT=/workspace/mns_results; mkdir -p $OUT
for D in 50 10; do for A in damp:damped ns:newton_stein; do
  arm=${A%%:*}; exp=${A##*:}; N=mns_d${D}_${arm}; R=results/$N; mkdir -p $R
  ( python train.py experiment=mosw_as_${exp} sigma_max=23.2 dim=$D K=10 num_epochs=1500 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 seed=0 "+save_controller_epochs=$EPS" \
      buffer_on_device=true fast_train=true fast_sampling=true project=mosw_ns use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=$N hydra.run.dir=$R '~lancher' > $R/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; echo "trained $N $(date +%s)" >> /workspace/progress.txt ) &
done; done
sleep 45   # let the config.yaml files appear before the workers read them
for D in 50 10; do for arm in as damp ns; do
  N=mns_d${D}_${arm}
  ( python scripts/mosw_step3_eval.py results/$N $OUT/$N/out.json --d $D --wait --deadline $DEADLINE --q_only > /workspace/mns_eval_$N.log 2>&1 \
      || echo "FAILED eval $N" >> /workspace/status.txt; echo "evaluated $N $(date +%s)" >> /workspace/progress.txt ) &
done; done
wait
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
