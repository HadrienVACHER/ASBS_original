# Copyright (c) Meta Platforms, Inc. and affiliates.

# ASBS on a K-mode Gaussian mixture, swept over dimensions (3 ... 165).
# 8, 39 and 165 match the dimensions of DW4, LJ13 and LJ55.
python train.py \
    experiment=gmm_asbs \
    dim=3,8,16,39,80,165 \
    seed=0,1,2 \
    -m &
