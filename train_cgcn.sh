#!/bin/bash

DEVICE_ID=$1
SAVE_ID=$2
CUDA_VISIBLE_DEVICES=$DEVICE_ID python3.7 train.py --id $SAVE_ID --seed 0 --prune_k 1 --lr .3 --rnn_hidden 200 --num_epoch 200 --pooling max --mlp_layers 2 --pooling_l2 0.003 --batch_size 50 --adj_type concat_deprel --log_step 20 --deprel_emb_dim 125 --deprel_dropout .1
