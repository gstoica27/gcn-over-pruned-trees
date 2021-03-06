"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
import os
import json

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.',
        default='/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/saved_models/Regular-CGCN')
parser.add_argument('--model', type=str, default='checkpoint_epoch_100.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='/Volumes/External HDD/dataset/tacred/data/json')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
args.cpu = True
torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['cuda'] = torch.cuda.is_available()
opt['cpu'] = not torch.cuda.is_available()
trainer = GCNTrainer(opt)
trainer.load(model_file)
trainer.opt['cuda'] = torch.cuda.is_available()
trainer.opt['cpu'] = not torch.cuda.is_available()
# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = args.data_dir + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
incorrect_indices = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]
is_incorrect = np.array(predictions) != np.array(batch.gold())
incorrect_data = np.array(batch.raw_data)[is_incorrect]
save_file = os.path.join( args.data_dir, 'test_incorrect.json')
with open(save_file, 'w') as handle:
    json.dump(incorrect_data.tolist(), handle)

p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

