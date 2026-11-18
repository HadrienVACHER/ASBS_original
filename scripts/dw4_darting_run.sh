#!/bin/bash
# One DW-4 continuation run with the darting control variate (rules_darting_run.md), local CPU: from the controller and buffer of the AS run 94252, 500 epochs, evaluation every 10 epochs.
#   scripts/dw4_darting_run.sh [seed]     -> results/dw4_cont_D3_s<seed>/ (stdout in stdout.log: the evaluation dictionaries are parsed from it)
set -e
SEED=${1:-1}
cd "$(dirname "$0")/.."
D=results/dw4_cont_D3_s${SEED}
mkdir -p $D
MPLBACKEND=Agg python train.py experiment=dw4_as_darting_states seed=${SEED} sigma_max=2 num_epochs=500 nfe=100 eval_freq=10 evaluator.eq_w2_every=100 num_eval_samples=5000 project=as dump_final=true \
  init_controller=results/dw4_e1500_94252_controller.pt init_buffer=results/dw4_e1500_94252_samples.pt init_buffer_key=buffer_x1 exp_name=dw4_cont_D3 use_wandb=false '~lancher' hydra.run.dir=$D > $D/stdout.log 2>&1
