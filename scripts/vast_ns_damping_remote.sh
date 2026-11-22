#!/bin/bash
# Runs on the Vast.ai instance (rules_ns_damping.md). 8 trainings in parallel: GMM-dk (32, 8, Delta), Delta in {4, 8}, sigma_max x {1, 2}, arms (b) damping eta = 0.5 and (c) Newton-Stein h = 1,
# seed 0, 1500 epochs, controller snapshots every 250 epochs, eval dumps every 50. The 4 reused AS runs (arm (a)) are uploaded in results/nsd_d32_D*_as_sx* (controller_ep1500.pt, config, eval dumps).
# Two evaluation workers (one per Delta) wait for each snapshot; then the verdict script. Env: EVAL_DEADLINE_REL = seconds after launch at which no new evaluation starts.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt; : > /workspace/progress.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
DEADLINE=$(( $(date +%s) + ${EVAL_DEADLINE_REL:-2880} ))
EPS="[250,500,750,1000,1250,1500]"
OUT=/workspace/nsd_results; mkdir -p $OUT
for D in 4 8; do for m in 1 2; do for A in damp:damped ns:newton_stein; do
  arm=${A%%:*}; exp=${A##*:}; N=nsd_d32_D${D}_${arm}_sx${m}; DIR=results/$N; mkdir -p $DIR
  if [ "$m" = "1" ]; then SIG=""; else SIG="sigma_max=$(python3 -c "import math; print(round(2*2*math.sqrt(1+$D**2/64),6))")"; fi   # x1: the rule resolver (as the reused sweep runs); x2: 6 decimals (as the reused window runs)
  ( python train.py experiment=gmm_dk_as_${exp} dim=32 K=8 delta=$D $SIG seed=0 num_epochs=1500 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 +dump_eval_samples=true "+save_controller_epochs=$EPS" \
      buffer_on_device=true fast_train=true fast_sampling=true project=ns_damping use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=$N hydra.run.dir=$DIR '~lancher' > $DIR/stdout.log 2>&1 \
      || echo "FAILED training $N" >> /workspace/status.txt; touch $DIR/TRAIN_DONE; echo "trained $N $(date +%s)" >> /workspace/progress.txt ) &
done; done; done
sleep 45   # let the config.yaml files appear before the workers read them
for D in 4 8; do
  R=""; for m in 1 2; do for arm in as damp ns; do R="$R results/nsd_d32_D${D}_${arm}_sx${m}"; done; done
  ( python scripts/ns_damping_eval.py $R --out $OUT --delta $D --final_only nsd_d32_D${D}_as_sx1 nsd_d32_D${D}_as_sx2 --wait --deadline $DEADLINE > /workspace/nsd_eval_D$D.log 2>&1 \
      || echo "FAILED eval D=$D" >> /workspace/status.txt; echo "eval done D=$D $(date +%s)" >> /workspace/progress.txt ) &
done
wait
python scripts/ns_damping_rules.py $OUT > /workspace/nsd_rules.log 2>&1 || echo "FAILED rules" >> /workspace/status.txt
grep -q FAILED /workspace/status.txt && exit 1
echo "DONE" > /workspace/status.txt
