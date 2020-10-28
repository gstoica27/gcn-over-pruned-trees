"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import yaml
import time
import os
import numpy as np
from collections import Counter
import json

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
        jrrelp_params = ['label_smoothing', 'lambda', 'freeze_network',
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

def compute_ranks(probs, gold_labels, hits_to_compute=(1, 3, 5, 10, 20, 50)):
    gold_ids = np.array([constant.LABEL_TO_ID[label] for label in gold_labels])
    all_probs = np.stack(probs, axis=0)
    sorted_args = np.argsort(-all_probs, axis=-1)
    ranks = []
    assert len(sorted_args) == len(gold_labels)
    for row_args, gold_label in zip(sorted_args, gold_ids):
        if id2label[gold_label] == 'no_relation':
            continue
        rank = int(np.where(row_args == gold_label)[0]) + 1
        ranks.append(rank)
    # print(Counter(ranks))
    ranks = np.array(ranks)
    hits = {hits_level: [] for hits_level in hits_to_compute}
    name2ranks = {}
    for hit_level in hits_to_compute:
        valid_preds = np.sum(ranks <= hit_level)
        hits[hit_level] = valid_preds / len(ranks)

    for hit_level in hits:
        name = 'HITs@{}'.format(int(hit_level))
        name2ranks[name] = hits[hit_level]

    mr = np.mean(ranks)
    mrr = np.mean(1. / ranks)
    name2ranks['MRR'] = mrr
    name2ranks['MR'] = mr
    print('RANKS:')
    for name, metric in name2ranks.items():
        if 'HIT' in name or 'MRR' in name:
            value = round(metric * 100, 2)
        else:
            value = round(metric, 2)
        print('{}: {}'.format(name, value))
    return name2ranks

def compute_structure_parts(data):
    argdists = []
    sentlens = []
    for instance in data:
        ss, se = instance['subj_start'], instance['subj_end']
        os, oe = instance['obj_start'], instance['obj_end']
        sentlens.append(len(instance['token']))
        if ss > oe:
            argdist = ss - oe
        else:
            argdist = os - se
        argdists.append(argdist)
    return {'argdists': argdists, 'sentlens': sentlens}

def compute_structure_errors(parts, preds, gold_labels):
    structure_errors = {'argdist=1': [], 'argdist>10': [], 'sentlen>30': []}
    argdists = parts['argdists']
    sentlens = parts['sentlens']
    for i in range(len(argdists)):
        argdist = argdists[i]
        sentlen = sentlens[i]
        pred = preds[i]
        gold = gold_labels[i]
        is_correct = pred == gold

        if argdist <= 1:
            structure_errors['argdist=1'].append(is_correct)
        if argdist > 10:
            structure_errors['argdist>10'].append(is_correct)
        if sentlen > 30:
            structure_errors['sentlen>30'].append(is_correct)
    print('Structure Errors:')
    for structure_name, error_list in structure_errors.items():
        accuracy = round(np.mean(error_list) * 100., 4)
        print('{} | Accuracy: {} | Correct: {} | Wrong: {} | Total: {} '.format(
            structure_name, accuracy, sum(error_list), len(error_list) - sum(error_list), len(error_list)
        ))
    return structure_errors


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='best_model.pt', help='Name of the model file.')

parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--config_dir', type=str)
args = parser.parse_args()

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'{"server" if on_server else "local"}_config.yaml')

def add_kg_model_params(cfg_dict, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'link_prediction_configs.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = cfg_dict['link_prediction']['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    params['freeze_network'] = cfg_dict['link_prediction']['freeze_network']
    return params

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
opt['id'] = create_model_name(cfg_dict)
model_load_dir = opt['save_dir'] + '/' + opt['id']
print(model_load_dir)
# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
# Add subject/object indices
opt['subject_indices'] = vocab.subj_idxs
opt['object_indices'] = vocab.obj_idxs

# load opt
model_file = os.path.join(model_load_dir, args.model_path)
print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load data
if opt['eval_file'] is not None:
    data_file = opt['eval_file']
else:
    data_file = opt['data_dir'] +f'/{opt["data_type"]}/test_{opt["version"]}.json'
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

if opt['link_prediction'] is not None:
    opt['link_prediction']['model'] = add_kg_model_params(cfg_dict, cwd)
    opt['num_relations'] = len(constant.LABEL_TO_ID)
    opt['num_subjects'] = len(constant.SUBJ_NER_TO_ID) - 2
    opt['num_objects'] = len(constant.OBJ_NER_TO_ID) - 2
    opt['link_prediction']['model']['num_objects'] = cfg_dict['num_objects']

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]
metrics, other_data = scorer.score(batch.gold(), predictions, verbose=True)

compute_ranks(all_probs, batch.gold())

structure_parts = compute_structure_parts(batch.raw_data)
compute_structure_errors(structure_parts, preds=predictions, gold_labels=batch.gold())

ids = [instance['id'] for instance in batch.raw_data]
formatted_data = []
for instance_id, pred, gold in zip(ids, predictions, batch.gold()):
    formatted_data.append(
        {
            "id": instance_id.replace("'", '"'),
            "label_true": gold.replace("'", '"'),
            "label_pred": pred.replace("'", '"')
        }
    )

p = metrics['precision']
r = metrics['recall']
f1 = metrics['f1']

wrong_indices = other_data['wrong_indices']
correct_indices = other_data['correct_indices']
wrong_predictions = other_data['wrong_predictions']

raw_data = np.array(batch.raw_data)
wrong_data = raw_data[wrong_indices]
correct_data = raw_data[correct_indices]

wrong_ids = [d['id'] for d in wrong_data]
correct_ids = [d['id'] for d in correct_data]

dataset_name = 'tacred-{}-{}'.format(cfg_dict['data_type'], cfg_dict['version'])
data_save_dir = os.path.join(opt['test_save_dir'], dataset_name)
os.makedirs(data_save_dir, exist_ok=True)
print('saving to: {}'.format(data_save_dir))
np.savetxt(os.path.join(data_save_dir, 'correct_ids.txt'), correct_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_ids.txt'), wrong_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_predictions.txt'), wrong_predictions, fmt='%s')
json.dump(formatted_data, open(os.path.join(data_save_dir, 'cgcn_tacred.jsonl'), 'w'))

with open(os.path.join(data_save_dir, 'cgcn_retacred.jsonl'), 'w') as handle:
    for instance in formatted_data:
            line = "{}\n".format(instance)
            handle.write(line)

id2preds = {d['id']: pred for d, pred in zip(raw_data, predictions)}
json.dump(id2preds, open(os.path.join(data_save_dir, 'id2preds.json'), 'w'))

print("Result: {:.2f}\t{:.2f}\t{:.2f}".format(p,r,f1))
print(Counter([relation for relation in predictions]))
print("Evaluation ended.")

