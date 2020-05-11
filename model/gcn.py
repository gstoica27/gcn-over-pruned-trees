"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.deprel_side = opt['hidden_dim']
        if opt['adj_type'] in ['diagonal_deprel', 'concat_deprel']:
            deprel_emb_size = self.deprel_side
        elif opt['adj_type'] == 'full_deprel':
            deprel_emb_size = self.deprel_side ** 2
        # regular adjacency matrix, thus fill with dummy weight
        else:
            deprel_emb_size = 1
        self.deprel_weight = nn.Embedding(len(constant.DEPREL_TO_ID), deprel_emb_size, padding_idx=0)
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.deprel_weight)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, deprel):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            deprel = deprel.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i], deprel[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else Variable(adj)
            # return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, deprel.data)
        h, pool_mask = self.gcn(adj, inputs)
        
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        # subj_mask = subj_pos.eq(0).unsqueeze(2)
        # obj_mask = obj_pos.eq(0).unsqueeze(2)
        # pool_mask = torch.logical_xor(pool_mask.eq(0), (subj_mask + obj_mask))
        # subj_mask, obj_mask, pool_mask = subj_mask.eq(0), obj_mask.eq(0), pool_mask.eq(0)
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb, self.deprel_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn input encoding
        if opt['adj_type'] in ['diagonal_deprel', 'full_deprel']:
            self.preprocessor = nn.Linear(self.in_dim, self.mem_dim)
            self.in_dim = self.mem_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            if opt['adj_type'] == 'concat_deprel':
                input_dim += self.deprel_emb.weight.shape[1]
            self.W.append(nn.Linear(input_dim,  self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        seq_lens = masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        if len(seq_lens.shape) == 0:
            seq_lens = [seq_lens]
        else:
            seq_lens = list(seq_lens)
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'], use_cuda=self.opt['cuda'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        if self.opt['adj_type'] not in ['regular', 'concat_deprel']:
            # Encode parameters for deprel changes
            gcn_inputs = self.preprocessor(gcn_inputs)
        batch_size, max_len, encoding_dim = gcn_inputs.shape
        if self.opt['adj_type'] != 'regular':
            deprel_adj = self.deprel_emb(adj.type(torch.int64)) # [B,T,T,H/H^2]
            if self.opt['adj_type'] in ['diagonal_deprel', 'concat_deprel']:
                # [B,T,T,H]
                deprel_adj = deprel_adj.reshape((batch_size, max_len, max_len, -1))
            else:
                # [B,T,T,H,H]
                deprel_adj = deprel_adj.reshape((batch_size, max_len, max_len, encoding_dim, encoding_dim))
        else:
            batch_size, max_len, encoding_dim = gcn_inputs.shape
            deprel_adj = None

        # gcn layer
        adj_matrix = torch.where(adj != 0, torch.ones_like(adj), torch.zeros_like(adj)).type(torch.float32)
        denom = adj_matrix.sum(2).unsqueeze(2) + 1
        mask = (adj_matrix.sum(2) + adj_matrix.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj_matrix = torch.zeros_like(adj_matrix)

        for l in range(self.layers):
            if self.opt['adj_type'] == 'regular':
                # [B,T1,T2] x [B,T2,H] -> [B,T1,H]
                Ax = adj_matrix.bmm(gcn_inputs)
            elif self.opt['adj_type'] == 'diagonal_deprel':
                # [B,T1,H] -> [B,1,T2,H]
                layer_inputs = gcn_inputs.view((batch_size, 1, max_len, encoding_dim))
                # [B,T1,T2,H] x [B,1,T2,H] -> [B,T1,T2,H] -> [B,T1,H]
                Ax = (deprel_adj * layer_inputs).sum(2)
            elif self.opt['adj_type'] == 'full_deprel':
                # [B,T1,H] -> [B,1,T2,H,1]
                layer_inputs = gcn_inputs.view((batch_size, 1, max_len, encoding_dim, 1))
                # [B,1,T2,H,1] -> [B,T1,T2,H,1]
                layer_inputs = layer_inputs.repeat((1, max_len, 1, 1, 1))
                # [B,T1,T2,H,H] x [B,T1,T2,H,1] -> [B,T1,T2,H,1]
                deprel_attended = torch.einsum('abcde,abcef->abcdf', deprel_adj, layer_inputs)
                # [B,T1,T2,H,1] -> [B,T1,T2,H]
                deprel_attended = deprel_attended.squeeze(-1)
                # [B,T1,T2,H] -> [B,T1,H]
                Ax = deprel_attended.sum(2)
            elif self.opt['adj_type'] == 'concat_deprel':
                # [B,T1,H] -> [B,1,T2,H]
                layer_inputs = gcn_inputs.view((batch_size, 1, max_len, -1))
                layer_inputs = layer_inputs.repeat((1, max_len, 1, 1))
                # [B,1,T2,H] x [B,T1,T2,h] -> [B,T1,H]
                layer_inputs = torch.cat([layer_inputs, deprel_adj], dim=-1)
                Ax = layer_inputs.sum(2)
            else:
                raise ValueError('Adjacency aggregation type not supported.')

            AxW = self.W[l](Ax)
            if self.opt['adj_type'] != 'concat_deprel':
                AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

