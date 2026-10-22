#!/bin/bash
# Submit the EMA comparison matrix on d = K = 24 and 36 (one qsub each). Run from the repo root on the cluster:
#   bash scripts/launch_ema_matrix.sh
# Prints the job ids; each job trains AS for 2000 epochs and logs to W&B (suffix in the run name).
cd "$(dirname "$0")/.." || exit 1
C="24:24:20:4.25 36:36:20:4.25"

sub() { # suffix experiment train_args
  qsub -v "STAGE=as,EXPERIMENT=$2,EXP_SUFFIX=$1,CONFIGS=$C,TRAIN_ARGS=$3" scripts/as_then_imh_gmm.pbs
}

# controls: plain AS and pure-MH corrected buffer, with EMA of the controller
sub _ema            gmm_as            "ema_decay=0.999"
sub _corr_ema       gmm_as_corrected  "correction_every=100 ema_decay=0.999"
# alternation with plain AS, with EMA
sub _corr_mix50_ema gmm_as_corrected  "correction_every=100 ema_decay=0.999 adjoint_matcher.model_frac=0.5"
sub _corr_stop600_ema gmm_as_corrected "correction_every=100 ema_decay=0.999 adjoint_matcher.stop_epoch=600"
sub _corr_alt50_ema gmm_as_corrected  "correction_every=100 ema_decay=0.999 adjoint_matcher.alternate_every=50"
# width test: finer time grid, exact buffer + EMA
sub _exact_ema_nfe400 gmm_as_target_buffer "ema_decay=0.999 nfe=400"
# small-t emphasis (a new key in this config, hence the +)
sub _exact_ema_tfocus gmm_as_target_buffer "ema_decay=0.999 +adjoint_matcher.t_focus=0.5"
