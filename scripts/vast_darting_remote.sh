#!/bin/bash
# Runs on the Vast.ai instance (not on the user's machine). W&B logging ON (project as, run dw4_cont_D3; WANDB_API_KEY is passed through the environment of the launching ssh session, never written to a file). Deployment check (exactness tests, ~1 min), then ONE darting continuation run (rules_darting_run.md) on the GPU: same protocol as the AS arms C1 / C2
# (continuation from the controller + buffer of the AS run 94252, seed 1, 500 epochs, nfe 100, evaluation every 10 epochs on 5000 samples), without the CUDA-only speed options (fast_sampling, fast_train, buffer_on_device: equivalent, not used).
#   bash vast_darting_remote.sh      -> /workspace/asbs/results/dw4_cont_D3_s1/stdout.log  (and DONE or FAILED in /workspace/status.txt)
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # the host exposes 64 cores: with the default 32 torch threads the small batched eigh / quench ran 20x slower than on the laptop
cd /workspace/asbs
echo "setup" > /workspace/status.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee /workspace/env_check.txt
pip install -q hydra-core omegaconf POT termcolor wandb pillow scipy matplotlib einops > /workspace/pip1.log 2>&1
pip install -q "git+https://github.com/atong01/bgflow" > /workspace/pip2.log 2>&1 || { echo "FAILED bgflow install" > /workspace/status.txt; tail -5 /workspace/pip2.log; exit 1; }
echo "tests" > /workspace/status.txt
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
python tests/test_dw4_darting.py > /workspace/test_kernel.log 2>&1 || { echo "FAILED kernel test" > /workspace/status.txt; tail -5 /workspace/test_kernel.log; exit 1; }
python tests/test_dw4_darting_matcher.py > /workspace/test_matcher.log 2>&1 || { echo "FAILED matcher test" > /workspace/status.txt; tail -5 /workspace/test_matcher.log; exit 1; }
echo "running" > /workspace/status.txt
D=results/dw4_cont_D3_s1
mkdir -p $D
python train.py experiment=dw4_as_darting_states seed=1 sigma_max=2 num_epochs=500 nfe=100 eval_freq=10 evaluator.eq_w2_every=100 num_eval_samples=5000 project=as dump_final=true \
  init_controller=results/dw4_e1500_94252_controller.pt init_buffer=results/dw4_e1500_94252_samples.pt init_buffer_key=buffer_x1 exp_name=dw4_cont_D3 use_wandb=true +code_commit=$(cat /workspace/code_commit.txt 2>/dev/null || echo unknown) '~lancher' hydra.run.dir=$D > $D/stdout.log 2>&1 && echo "DONE" > /workspace/status.txt || echo "FAILED training" > /workspace/status.txt
