#!/bin/bash

DEVICE_ID=$1
SAVE_ID=$2
CUDA_VISIBLE_DEVICES=$DEVICE_ID python3.7 train_semeval.py --id $SAVE_ID --seed 0 --prune_k 1 --lr .3 --rnn_hidden 200 --num_epoch 200 --pooling max --mlp_layers 2 --pooling_l2 0.003 --batch_size 25 --log_step 40 --num_tree_lstms 1 --deprel_emb_dim 0 --emb_dropout .0 --word_dropout .04
