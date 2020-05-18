"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils
import os

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if len(batch) >= 10:
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch[:10]]
            labels = Variable(batch[10].cuda())
        else:
            inputs = [Variable(b) for b in batch[:10]]
            labels = Variable(batch[10])
        tokens = batch[0]
        head = batch[5]
        subj_pos = batch[6]
        obj_pos = batch[7]
        lens = batch[1].eq(0).long().sum(1).squeeze()
    else:
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch[:-2]]
            labels = Variable(batch[-2].cuda())
        else:
            inputs = [Variable(b) for b in batch[:-2]]
            labels = Variable(batch[-2])
        tokens = batch[0]
        head = batch[4]
        subj_pos = batch[5]
        obj_pos = batch[6]
        lens = batch[1].eq(0).long().sum(1).squeeze()

    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None, save_dir=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

        if save_dir is not None:
            self.sentence_save_file = os.path.join(save_dir, 'sentence.pkl')
            self.subject_save_file = os.path.join(save_dir, 'subject.pkl')
            self.object_save_file = os.path.join(save_dir, 'object.pkl')

    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        # self.model.train()
        # self.optimizer.zero_grad()
        logits, pooling_output, component_encs = self.model(inputs)
        subj_enc, obj_enc = component_encs
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()

        # loss /= update_gap
        # loss_val = loss.item()
        # backward
        # loss.backward()
        # if step_num % update_gap == 0:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        return loss

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, pooling_output, component_encs = self.model(inputs)
        subj_enc, obj_enc = component_encs
        pooling_output = pooling_output.data.cpu().numpy().tolist()
        subj_enc = subj_enc.data.cpu().numpy().tolist()
        obj_enc = obj_enc.data.cpu().numpy().tolist()

        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs, pooling_output, subj_enc, obj_enc = [list(t) for t in zip(*sorted(zip(orig_idx,
                    predictions, probs, pooling_output, subj_enc, obj_enc)))]
        return predictions, probs, loss.item(), (pooling_output, subj_enc, obj_enc)

    def get_deprel_emb(self):
        return self.model.get_deprel_emb()