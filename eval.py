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

data_save_dir = os.path.join(opt['test_save_dir'], 'unpatched_full')
os.makedirs(data_save_dir, exist_ok=True)
print('saving to: {}'.format(data_save_dir))
np.savetxt(os.path.join(data_save_dir, 'correct_ids.txt'), correct_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_ids.txt'), wrong_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_predictions.txt'), wrong_predictions, fmt='%s')

id2preds = {d['id']: pred for d, pred in zip(raw_data, predictions)}
json.dump(id2preds, open(os.path.join(data_save_dir, 'id2preds.json'), 'w'))

print("Result: {:.2f}\t{:.2f}\t{:.2f}".format(p,r,f1))
print(Counter([relation for relation in predictions]))
print("Evaluation ended.")

