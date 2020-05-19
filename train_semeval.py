"""
Train a model on TACRED.
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict

from data.semeval_loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, semeval_constant as constant, helper
from utils.vocab import Vocab


def str2bool(v):
    return v.lower() in ('true')


cwd = os.getcwd()
on_server = 'Desktop' not in cwd
# Local paths
local_data_dir = '/Volumes/External HDD/dataset/semeval/data/json'
local_vocab_dir = '/Volumes/External HDD/dataset/semeval/data/vocab'
local_model_save_dir = '/Volumes/External HDD/dataset/semeval/saved_models'
local_test_save_dir = os.path.join(cwd, 'semeval_test_performances')
os.makedirs(local_test_save_dir, exist_ok=True)
# Server paths
server_data_dir = '/usr0/home/gis/data/semeval/data/json'
server_vocab_dir = '/usr0/home/gis/data/semeval/data/vocab'
server_model_save_dir = '/usr0/home/gis/research/tacred-exploration/saved_models'
server_test_save_dir = '/usr0/home/gis/research/tacred-exploration/tacred_test_performances'
# paths
data_dir = server_data_dir if on_server else local_data_dir
vocab_dir = server_vocab_dir if on_server else local_vocab_dir
model_save_dir = server_model_save_dir if on_server else local_model_save_dir
test_save_dir = server_test_save_dir if on_server else local_test_save_dir

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--vocab_dir', type=str, default=vocab_dir)
parser.add_argument('--model_save_dir', type=str, default=model_save_dir)
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.add_argument('--test_save_dir', default=test_save_dir, type=str)
parser.set_defaults(lower=False)

parser.add_argument('--prune_k', default=-1, type=int,
                    help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max',
                    help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--test_confusion_save_file', default='')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

# parser.add_argument('--adj_type', type=str, default='regular')
parser.add_argument('--deprel_emb_dim', type=int, default=0)
parser.add_argument('--deprel_dropout', type=float, default=.5)

parser.add_argument('--num_tree_lstms', type=int, default=1)
parser.add_argument('--tree_x_dropout', type=float, default=.0)
parser.add_argument('--tree_h_dropout', type=float, default=.0)
parser.add_argument('--node_pooling', type=str2bool, default=False)
parser.add_argument('--emb_dropout', type=float, default=.0)
parser.add_argument('--use_bert_embeddings', type=str2bool, default=False)
parser.add_argument('--dataset', default='semeval', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
if opt['use_bert_embeddings']:
    embeddings_file = '/usr0/home/gis/data/bert_saves/id2embeddings.pkl'
else:
    embeddings_file = None
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False,
                         bert_embeddings=embeddings_file)
test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'], opt, vocab, evaluation=True,
                        bert_embeddings=embeddings_file)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['model_save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

test_save_dir = os.path.join(opt['test_save_dir'], opt['id'])
os.makedirs(test_save_dir, exist_ok=True)
test_save_file = os.path.join(test_save_dir, 'test_records.pkl')
test_confusion_save_file = os.path.join(test_save_dir, 'test_confusion_matrix.pkl')
train_confusion_save_file = os.path.join(test_save_dir, 'train_confusion_matrix.pkl')
deprel_save_file = os.path.join(test_save_dir, 'deprel_embs.pkl')
# Record saving over time
test_perf_save_file = os.path.join(test_save_dir, 'test_metrics.txt')
train_perf_save_file = os.path.join(test_save_dir, 'train_metrics.txt')
# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)

id2label = dict([(v, k) for k, v in label2id.items()])
train_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_train_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_train = defaultdict(lambda: -np.inf)

# start training
update_gap = max(int(50 / opt['batch_size']), 1)
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    # Training in-case of mini-batches
    trainer.model.train()
    trainer.optimizer.zero_grad()

    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        loss.backward()
        loss_val = loss.item()
        step_num = i + 1
        if step_num % update_gap == 0:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.opt['max_grad_norm'])
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        train_loss += loss_val
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], loss, duration, current_lr))
    # Update grads if needed
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.opt['max_grad_norm'])
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

    # print("Saving Deprel Embeddings...")
    # with open(deprel_save_file, 'wb') as handle:
    #     pickle.dump(trainer.get_deprel_emb(), handle)

    # eval on train
    print("Evaluating on train set...")
    train_predictions = []
    train_eval_loss = 0
    train_components = {'sentence': [], 'subject': [], 'object': [], 'labels': []}
    for i, batch in enumerate(train_batch):
        preds, _, loss, batch_components = trainer.predict(batch)
        train_predictions += preds
        train_eval_loss += loss
        sentence_enc, subject_enc, object_enc = batch_components

        train_components['sentence'] += sentence_enc
        train_components['subject'] += subject_enc
        train_components['object'] += object_enc

    train_predictions = [id2label[p] for p in train_predictions]
    train_eval_loss = train_eval_loss / train_batch.num_examples * opt['batch_size']
    train_components['labels'] = train_batch.gold()

    train_p, train_r, train_f1 = scorer.score(train_batch.gold(), train_predictions)
    print("epoch {}: train_loss = {:.6f}, train_eval_loss = {:.6f}, dev_f1 = {:.4f}".format(
        epoch, train_loss, train_eval_loss, train_f1))
    train_score = train_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, train_eval_loss, train_f1))
    current_train_metrics = {'f1': train_f1, 'precision': train_p, 'recall': train_r}

    # eval on test
    test_predictions = []
    test_components = {'sentence': [], 'subject': [], 'object': [], 'labels': []}
    for i, batch in enumerate(test_batch):
        preds, _, loss, batch_components = trainer.predict(batch)
        test_predictions += preds
        sentence_enc, subject_enc, object_enc = batch_components
        test_components['sentence'] += sentence_enc
        test_components['subject'] += subject_enc
        test_components['object'] += object_enc

    test_predictions = [id2label[p] for p in test_predictions]
    test_components['labels'] = test_batch.gold()

    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), test_predictions)
    test_metrics_at_current_train = {'f1': test_f1, 'precision': test_p, 'recall': test_r}

    if best_train_metrics['f1'] < current_train_metrics['f1']:
        best_train_metrics = current_train_metrics
        test_metrics_at_best_train = test_metrics_at_current_train
        trainer.save(os.path.join(model_save_dir, 'best_model.pt'), epoch)
        print("New best model saved")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}" \
                        .format(epoch, test_p * 100, test_r * 100, test_f1 * 100))

        # Compute Confusion Matrices over triples excluded in Training
        test_preds = np.array(test_predictions)
        test_gold = np.array(test_batch.gold())
        train_preds = np.array(train_predictions)
        train_gold = np.array(train_batch.gold())
        test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_gold,
                                                                  predictions=test_preds)
        train_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=train_gold,
                                                                 predictions=train_preds)
        print("Saving Excluded Triple Confusion Matrices...")
        with open(test_confusion_save_file, 'wb') as handle:
            pickle.dump(test_confusion_matrix, handle)

        component_save_dir = os.path.join(test_save_dir, 'components')
        os.makedirs(component_save_dir, exist_ok=True)

        test_components_save_dir = os.path.join(component_save_dir, 'test')
        os.makedirs(test_components_save_dir, exist_ok=True)
        print(f'Saving Test logits to: {test_components_save_dir}')
        for name, data in test_components.items():
            with open(os.path.join(test_components_save_dir, name + '.pkl'), 'wb') as handle:
                pickle.dump(data, handle)

        train_components_save_dir = os.path.join(component_save_dir, 'train')
        os.makedirs(train_components_save_dir, exist_ok=True)
        print(f'Saving Train logits to: {train_components_save_dir}')
        for name, data in train_components.items():
            with open(os.path.join(train_components_save_dir, name + '.pkl'), 'wb') as handle:
                pickle.dump(data, handle)

    print("Best Dev Metrics | F1: {} | Precision: {} | Recall: {}".format(
        best_train_metrics['f1'], best_train_metrics['precision'], best_train_metrics['recall']
    ))
    print("Test Metrics at Best Dev | F1: {} | Precision: {} | Recall: {}".format(
        test_metrics_at_best_train['f1'], test_metrics_at_best_train['precision'], test_metrics_at_best_train['recall']
    ))
    # Record all metrics in associated files
    helper.record_metrics(train_perf_save_file, train_p, train_r, train_f1)
    helper.record_metrics(test_perf_save_file, test_p, test_r, test_f1)

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or train_score > max(train_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}" \
                        .format(epoch, train_p * 100, train_r * 100, train_score * 100))
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(train_score_history) > opt['decay_epoch'] and train_score <= train_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    train_score_history += [train_score]
    print("")

print("Training ended with {} epochs.".format(epoch))

