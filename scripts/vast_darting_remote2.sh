#!/bin/bash
# Runs on the Vast.ai instance (not on the user's machine). Protocol: rules_darting_seeds_gamma.md. Same continuation protocol as C1 / C2 WITH fast_sampling, fast_train, buffer_on_device.
#   bash vast_darting_remote2.sh phase1   -> D3 seeds 1, 2 in parallel
#   bash vast_darting_remote2.sh phase2   -> D10 and D30, seed 1 only (2 runs in parallel)
#   bash vast_darting_remote2.sh explore  -> seed 1: D100, D30 + focus, AS + focus (CF), D100 + focus (4 runs in parallel; eq_w2 every 10 epochs)
#   bash vast_darting_remote2.sh confirm  -> rules_darting_confirm.md: D30 seeds 2,3,4 (darting gamma 30) + AS controls C2 (re-run), C3, C4 (seeds 2, 3, 4), 6 runs in parallel
# Logs: /workspace/asbs/results/<name>_s<seed>/stdout.log ; status in /workspace/status.txt (DONE or FAILED). WANDB_API_KEY comes from the environment of the launching ssh session.
set -e
PHASE=${1:?phase1 or phase2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /workspace/asbs
echo "setup" > /workspace/status.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee /workspace/env_check.txt
nproc | tee -a /workspace/env_check.txt
pip install -q hydra-core omegaconf POT termcolor wandb pillow scipy matplotlib einops > /workspace/pip1.log 2>&1
pip install -q "git+https://github.com/atong01/bgflow" > /workspace/pip2.log 2>&1 || { echo "FAILED bgflow install" > /workspace/status.txt; tail -5 /workspace/pip2.log; exit 1; }
echo "tests" > /workspace/status.txt
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
python tests/test_dw4_darting.py > /workspace/test_kernel.log 2>&1 || { echo "FAILED kernel test" > /workspace/status.txt; tail -5 /workspace/test_kernel.log; exit 1; }
python tests/test_dw4_darting_matcher.py > /workspace/test_matcher.log 2>&1 || { echo "FAILED matcher test" > /workspace/status.txt; tail -5 /workspace/test_matcher.log; exit 1; }
echo "running" > /workspace/status.txt
COMMIT=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown)
COMMON="sigma_max=2 num_epochs=500 nfe=100 eval_freq=10 evaluator.eq_w2_every=${EQ_EVERY:-100} num_eval_samples=5000 project=as dump_final=true buffer_on_device=true fast_train=true fast_sampling=true init_controller=results/dw4_e1500_94252_controller.pt init_buffer=results/dw4_e1500_94252_samples.pt init_buffer_key=buffer_x1 use_wandb=true +code_commit=${COMMIT} ~lancher"
pids=()
run() { name=$1; seed=$2; gamma=$3; shift 3; D=results/${name}_s${seed}; mkdir -p $D
  python train.py experiment=dw4_as_darting_states seed=${seed} adjoint_matcher.gamma=${gamma} exp_name=${name} hydra.run.dir=$D ${COMMON} "$@" > $D/stdout.log 2>&1 & pids+=($!); }
run_as() { name=$1; seed=$2; shift 2; D=results/${name}_s${seed}; mkdir -p $D
  python train.py experiment=dw4_as_states seed=${seed} exp_name=${name} hydra.run.dir=$D ${COMMON} "$@" > $D/stdout.log 2>&1 & pids+=($!); }
if [ "$PHASE" = "explore" ]; then
  # exploration, seed 1, descriptive (RESULTS.md): eq_w2 every 10 epochs; focus = 50 % of the regression times in [0, 0.05]
  export EQ_EVERY=10
  COMMON="${COMMON/eq_w2_every=100/eq_w2_every=10}"
  F="+adjoint_matcher.t_focus=0.5 +adjoint_matcher.t_focus_max=0.05"
  run dw4_cont_D100 1 100.0; run dw4_cont_D30F 1 30.0 $F; run_as dw4_cont_CF 1 $F; run dw4_cont_D100F 1 100.0 $F
elif [ "$PHASE" = "confirm" ]; then
  run dw4_cont_D30 2 30.0; run dw4_cont_D30 3 30.0; run dw4_cont_D30 4 30.0; run_as dw4_cont_C2 2; run_as dw4_cont_C3 3; run_as dw4_cont_C4 4
elif [ "$PHASE" = "phase1" ]; then
  run dw4_cont_D3 1 3.0; run dw4_cont_D3 2 3.0
else
  run dw4_cont_D10 1 10.0; run dw4_cont_D30 1 30.0   # seed 1 only (user decision 2026-11-18)
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail = 0 ]; then echo "DONE" > /workspace/status.txt; else echo "FAILED training" > /workspace/status.txt; fi
