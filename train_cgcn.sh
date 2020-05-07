#!/bin/bash

DEVICE=$1
SAVE_ID=$2
CUDA_VISIBLE_DEVICES=$DEVICE python3.7 train.py --id $SAVE_ID --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 100 --num_epoch 200 --pooling max --mlp_layers 2 --pooling_l2 0.003
