#!/usr/bin/env bash

export PATH=/home/rdai/anaconda3/envs/torch1.9/bin:$PATH

python train.py \
-dataset TSU \
-mode rgb \
-split_setting CS \
-model PDAN \
-train True \
-num_channel 512 \
-lr 0.0002 \
-kernelsize 3 \
-comp_info TSU_TCN \
-APtype map \
-epoch 140 \
-batch_size 2 \

