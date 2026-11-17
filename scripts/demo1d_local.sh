#!/bin/bash
# Local CPU re-run of the 1D two-mode demo (GMM1D: modes at -3 and +3, weights 0.1 / 0.9, scale 0.5), report only, no tuning:
# the configs of configs/experiment/demo_as*.yaml are unchanged except num_epochs (1000: in the W&B runs of the plain AS the left-mode mass is stationary from epoch 100), dump_final and use_wandb=false.
#   usage: scripts/demo1d_local.sh <demo_as | demo_as_barker | demo_as_mh> <seed> [epochs]
set -e
EXP=$1; SEED=${2:-0}; EP=${3:-1000}
cd "$(dirname "$0")/.."
MPLBACKEND=Agg python train.py experiment=$EXP seed=$SEED num_epochs=$EP eval_freq=100 dump_final=true use_wandb=false '~lancher' hydra.run.dir=results/demo1d/${EXP}_s${SEED}
