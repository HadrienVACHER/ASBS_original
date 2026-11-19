#!/bin/bash
# Runs on the Vast.ai instance. Protocol: rules_gmm2d_radius.md. AS only, gmm 2D, K=3, radius sweep. Logs: /workspace/asbs/results/gmm2d_r<R>_s0/stdout.log ; status in /workspace/status.txt
set -e
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd /workspace/asbs
echo "setup" > /workspace/status.txt
if ! python -c "import bgflow, hydra, ot, wandb" 2>/dev/null; then
  pip install -q hydra-core omegaconf POT termcolor wandb pillow scipy matplotlib einops > /workspace/pip1.log 2>&1
  pip install -q "git+https://github.com/atong01/bgflow" > /workspace/pip2.log 2>&1 || { echo "FAILED bgflow install" > /workspace/status.txt; exit 1; }
fi
echo "running" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
pids=()
for R in ${RADII:-2 4 6}; do
  D=results/gmm2d_r${R}_s0; mkdir -p $D
  python train.py experiment=gmm_as dim=2 K=3 radius=${R} seed=0 num_epochs=1000 eval_freq=50 project=gmm2d_radius exp_name=gmm2d_r${R} dump_final=true use_wandb=true +code_commit=${COMMIT} hydra.run.dir=$D ~lancher > $D/stdout.log 2>&1 & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail = 0 ]; then echo "DONE" > /workspace/status.txt; else echo "FAILED training" > /workspace/status.txt; fi
