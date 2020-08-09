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


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='best_model.pt', help='Name of the model file.')

parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--config_dir', type=str)
args = parser.parse_args()

cwd = os.getcwd()

def add_kg_model_params(cfg_dict, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'link_prediction_configs.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = cfg_dict['link_prediction']['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    params['freeze_network'] = cfg_dict['link_prediction']['freeze_network']
    return params

config_path = os.path.join(args.config_dir, 'config.json')
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


# load opt
model_file = os.path.join(cfg_dict['model_save_dir'], args.model_path)
print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = os.path.join(cfg_dict['model_save_dir'], 'vocab.pkl')
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + f'/{opt["data_type"]}_test.json'
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
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))
print(Counter([relation for relation in predictions]))
print("Evaluation ended.")

