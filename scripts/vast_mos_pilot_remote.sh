#!/bin/bash
# Runs on the Vast.ai instance. Protocol: rules_mos_pilot.md. 3 AS runs on the MoS d = 50 (sigma_max 5.8, 11.6, 23.2; 300 epochs; seed 0), then scripts/mos_pilot_eval.py.
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "training" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
pids=()
for sg in 5.8 11.6 23.2; do
  D=results/mospilot_s$sg; mkdir -p $D
  python train.py experiment=mos_as sigma_max=$sg dim=50 K=10 num_epochs=300 eval_freq=50 num_eval_samples=5000 dump_final=true dump_n_samples=10000 seed=0 \
    buffer_on_device=true fast_train=true fast_sampling=true project=mos_pilot use_wandb=${WANDB:-true} +code_commit=${COMMIT} exp_name=mospilot_s$sg hydra.run.dir=$D '~lancher' > $D/stdout.log 2>&1 & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail != 0 ]; then echo "FAILED training" > /workspace/status.txt; exit 1; fi
echo "evaluating" > /workspace/status.txt
python scripts/mos_pilot_eval.py results /workspace/pilot_results.json --n_paths 20000 --budget_s ${EVAL_BUDGET:-300} > /workspace/pilot_eval.log 2>&1 || { echo "FAILED eval" > /workspace/status.txt; exit 1; }
echo "DONE" > /workspace/status.txt
