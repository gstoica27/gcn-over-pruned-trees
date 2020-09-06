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

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import yaml

def generate_param_list(params, cfg_dict, prefix=''):
    param_list = prefix
    for param in params:
        if param_list == '':
            param_list += f'{cfg_dict[param]}'
        else:
            param_list += f'-{cfg_dict[param]}'
    return param_list

def create_model_name(cfg_dict):
    top_level_name = 'TACRED-{}-{}'.format(cfg_dict['data_type'].upper(), cfg_dict['version'].upper())
    approach_type = 'CGCN-JRRELP' if cfg_dict['link_prediction'] is not None else 'CGCN'
    optim_name = ['optim', 'lr', 'lr_decay', 'conv_l2', 'pooling_l2', 'max_grad_norm', 'seed']
    base_params = ['emb_dim', 'ner_dim', 'pos_dim', 'hidden_dim', 'num_layers', 'mlp_layers',
                   'input_dropout', 'gcn_dropout', 'word_dropout', 'lower', 'prune_k', 'no_adj']

    param_name_list = [top_level_name, approach_type]

    optim_name = generate_param_list(optim_name, cfg_dict, prefix='optim')
    param_name_list.append(optim_name)

    main_name = generate_param_list(base_params, cfg_dict, prefix='base')
    param_name_list.append(main_name)

    if cfg_dict['rnn']:
        rnn_params = ['rnn_hidden', 'rnn_layers', 'rnn_dropout']
        rnn_name = generate_param_list(rnn_params, cfg_dict, prefix='rnn')
        param_name_list.append(rnn_name)

    if cfg_dict['link_prediction'] is not None:
        kglp_task_cfg = cfg_dict['link_prediction']
        jrrelp_params = ['label_smoothing', 'lambda', 'free_network',
                       'with_relu', 'without_observed',
                       'without_verification', 'without_no_relation']
        jrrelp_name = generate_param_list(jrrelp_params, kglp_task_cfg, prefix='jrrelp')
        param_name_list.append(jrrelp_name)

        kglp_params = ['input_drop', 'hidden_drop', 'feat_drop', 'rel_emb_dim', 'use_bias', 'filter_channels', 'stride']
        lp_cfg = cfg_dict['link_prediction']['model']
        kglp_name = generate_param_list(kglp_params, lp_cfg, prefix='kglp')
        param_name_list.append(kglp_name)

    aggregate_name = os.path.join(*param_name_list)
    return aggregate_name


def add_kg_model_params(cfg_dict, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'link_prediction_configs.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = cfg_dict['link_prediction']['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    params['freeze_network'] = cfg_dict['link_prediction']['freeze_network']
    return params

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'{"server" if on_server else "local"}_config.yaml')
# config_path = os.path.join(cwd, 'configs', 'server_config.yaml')
with open(config_path, 'r') as file:
    cfg_dict = yaml.load(file)

cfg_dict['topn'] = float(cfg_dict['topn'])

opt = cfg_dict
torch.manual_seed(opt['seed'])
np.random.seed(opt['seed'])
random.seed(1234)
if opt['cpu']:
    opt['cuda'] = False
elif opt['cuda']:
    torch.cuda.manual_seed(opt['seed'])
init_time = time.time()

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
# Add subject/object indices
opt['subject_indices'] = vocab.subj_idxs
opt['object_indices'] = vocab.obj_idxs

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + f'/{opt["data_type"]}/train_{opt["version"]}.json', opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + f'/{opt["data_type"]}/dev_{opt["version"]}.json', opt['batch_size'], opt, vocab, evaluation=True,
                       kg_graph=train_batch.kg_graph)
test_batch = DataLoader(opt['data_dir'] + f'/{opt["data_type"]}/test_{opt["version"]}.json', opt['batch_size'], opt, vocab, evaluation=True,
                        kg_graph=dev_batch.kg_graph)

if opt['link_prediction'] is not None:
    opt['link_prediction']['model'] = add_kg_model_params(cfg_dict, cwd)
    opt['num_relations'] = len(constant.LABEL_TO_ID)
    opt['num_subjects'] = len(constant.SUBJ_NER_TO_ID) - 2
    opt['num_objects'] = len(constant.OBJ_NER_TO_ID) - 2
    opt['link_prediction']['model']['num_objects'] = cfg_dict['num_objects']

# model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_id = create_model_name(opt)

model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

test_save_dir = os.path.join(opt['test_save_dir'], opt['id'])
os.makedirs(test_save_dir, exist_ok=True)
test_save_file = os.path.join(test_save_dir, 'test_records.pkl')
test_confusion_save_file = os.path.join(test_save_dir, 'test_confusion_matrix.pkl')
dev_confusion_save_file = os.path.join(test_save_dir, 'dev_confusion_matrix.pkl')

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

id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_dev_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_dev = defaultdict(lambda: -np.inf)

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))
    
    # eval on train
    print("Evaluating on train set...")
    train_predictions = []
    train_eval_loss = 0
    for i, batch in enumerate(train_batch):
        preds, _, loss = trainer.predict(batch)
        train_predictions += preds
        train_eval_loss += loss
    train_predictions = [id2label[p] for p in train_predictions]
    train_eval_loss = train_eval_loss / train_batch.num_examples * opt['batch_size']

    train_p, train_r, train_f1 = scorer.score(train_batch.gold(), train_predictions)
    print("epoch {}: train_loss = {:.6f}, train_eval_loss = {:.6f}, dev_f1 = {:.4f}".format(
        epoch, train_loss, train_eval_loss, train_f1))
    train_score = train_f1
    # file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, train_eval_loss, train_f1))
    
    # eval on dev
    print("Evaluating on dev set...")
    dev_predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = trainer.predict(batch)
        dev_predictions += preds
        dev_loss += loss
    dev_predictions = [id2label[p] for p in dev_predictions]
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), dev_predictions)
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch, train_loss, dev_loss, dev_f1))
    dev_score = dev_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))
    current_dev_metrics = {'f1': dev_f1, 'precision': dev_p, 'recall': dev_r}

     # eval on test
    test_predictions = []
    for i, batch in enumerate(test_batch):
        preds, _, loss = trainer.predict(batch)
        test_predictions += preds
    test_predictions = [id2label[p] for p in test_predictions]

    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), test_predictions)
    test_metrics_at_current_dev = {'f1': test_f1, 'precision': test_p, 'recall': test_r}
    
    if best_dev_metrics['f1'] < current_dev_metrics['f1']:
        best_dev_metrics = current_dev_metrics
        test_metrics_at_best_dev = test_metrics_at_current_dev
        trainer.save(os.path.join(model_save_dir, opt['id'], 'best_model.pt'), epoch)
        print("New best model saved")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}" \
                        .format(epoch, test_p * 100, test_r * 100, test_f1 * 100))

        # Compute Confusion Matrices over triples excluded in Training
        test_preds = np.array(test_predictions)
        test_gold = np.array(test_batch.gold())
        dev_preds = np.array(dev_predictions)
        dev_gold = np.array(dev_batch.gold())
        test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_gold,
                                                                  predictions=test_preds)
        dev_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=dev_gold,
                                                                 predictions=dev_preds)
        print("Saving Excluded Triple Confusion Matrices...")
        with open(test_confusion_save_file, 'wb') as handle:
            pickle.dump(test_confusion_matrix, handle)

    print("Best Dev Metrics | F1: {} | Precision: {} | Recall: {}".format(
        best_dev_metrics['f1'], best_dev_metrics['precision'], best_dev_metrics['recall']
    ))
    print("Test Metrics at Best Dev | F1: {} | Precision: {} | Recall: {}".format(
        test_metrics_at_best_dev['f1'], test_metrics_at_best_dev['precision'], test_metrics_at_best_dev['recall']
    ))
    
    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))

