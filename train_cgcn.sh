#!/bin/bash

DEVICE_ID=$1
SAVE_ID=$2
CUDA_VISIBLE_DEVICES=$DEVICE_ID python3.7 train.py --id $SAVE_ID --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --word_dropout 0.04 --emb_dropout 0.0 --deprel_emb 50 --adj_type full_deprel --batch_size 50
