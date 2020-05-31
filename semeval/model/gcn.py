"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from semeval.model.tree import Tree, head_to_tree, tree_to_adj
from semeval.utils import constant, torch_utils
from semeval.model.link_prediction_models import *

def initialize_link_prediction_model(params):
    name = params['name'].lower()
    if name == 'distmult':
        model = DistMult(params)
    elif name == 'conve':
        model = ConvE(params)
    elif name == 'complex':
        model = Complex(params)
    else:
        raise ValueError('Only, {distmult, conve, and complex}  are supported')
    return model


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        logits, pooling_output, supplemental_losses = self.gcn_model(inputs)
        return logits, pooling_output, supplemental_losses

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.incorrect_trees = 0
        self.incorrect_indices = {'train':[], 'test':[], 'dev': []}
        self.indice_type = 'train'
        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()
        # Store indices for emb lookup
        self.object_indices = self.opt['object_indices']
        self.subject_indices = self.opt['subject_indices']
        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])
        # LP Model
        if opt['link_prediction'] is not None:
            link_prediction_cfg = opt['link_prediction']['model']
            self.rel_emb = nn.Embedding(opt['num_relations'], link_prediction_cfg['rel_emb_dim'])
            self.register_parameter('rel_bias', torch.nn.Parameter(torch.zeros((opt['num_relations']))))
            self.object_indices = torch.from_numpy(np.array(self.object_indices))
            if opt['cuda']:
                self.object_indices = self.object_indices.cuda()
            self.lp_model = initialize_link_prediction_model(link_prediction_cfg)

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            if opt['link_prediction'] is None or _ < self.opt['mlp_layers'] - 2:
                output_dim = opt['hidden_dim']
                layers += [nn.Linear(opt['hidden_dim'], output_dim), nn.ReLU()]
            else:
                output_dim = opt['link_prediction']['model']['rel_emb_dim']
                layers += [nn.Linear(opt['hidden_dim'], output_dim)]
                if opt['link_prediction']['with_relu']:
                    layers += [nn.ReLU()]
            # layers += [nn.Linear(opt['hidden_dim'], output_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        # Classifier for baseline model
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])

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
        base_inputs = inputs['base']
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = base_inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            # trees = []

            # for i in range(len(l)):
            #     try:
            #         tree = head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i])
            #         trees.append(tree)
            #     except:
            #         self.incorrect_trees += 1
            #         self.incorrect_indices[self.indice_type].append(ids[i])


            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        # h_out = outputs = torch.zeros(head.shape[0], self.opt['link_prediction']['model']['rel_emb_dim'])

        if self.opt['link_prediction'] is not None:
            subjects, relations, known_objects = inputs['kg']
            object_embs = self.emb(self.object_indices)
            subject_embs = self.emb(subjects)
            relation_embs = self.rel_emb(relations)
            # Predict objects with LP model
            observed_preds = self.lp_model(subject_embs, relation_embs, object_embs)
            baseline_preds = self.lp_model(subject_embs, outputs, object_embs)
            # Compute loss terms
            if self.opt['link_prediction']['without_no_relation']:
                # Void loss from examples containing "no_relation"
                # TODO: It might be best to find a better approach for this b/c/ this can mess up the batch training.
                #  Wherein the relation embeddings are updated at a completely different pace than the rest of the network.
                #  (e.g. batch numbers are different every time)
                no_relation_blacklist = torch.eq(relations, constant.NO_RELATION_ID).eq(0).type(torch.float32).unsqueeze(-1)
                observed_loss = self.lp_model.loss(observed_preds, known_objects)
                baseline_loss = self.lp_model.loss(baseline_preds, known_objects)
                observed_loss = observed_loss * no_relation_blacklist
                baseline_loss = baseline_loss * no_relation_blacklist
                # Mean over column dimension
                observed_loss = observed_loss.mean(-1)
                baseline_loss = baseline_loss.mean(-1)
                # Total positive relation examples
                num_positives = no_relation_blacklist.sum()
                # Mean only over the positive examples
                observed_loss = observed_loss.sum() / num_positives
                baseline_loss = baseline_loss.sum() / num_positives
            else:
                observed_loss = self.lp_model.loss(observed_preds, known_objects).mean()
                baseline_loss = self.lp_model.loss(baseline_preds, known_objects).mean()

            supplemental_losses = {'observed': observed_loss, 'baseline': baseline_loss}
            # Relation extraction loss
            logits = torch.mm(outputs, self.rel_emb.weight.transpose(1, 0))
            logits += self.rel_bias.expand_as(logits)
        else:
            logits = self.classifier(outputs)
            # logits = outputs
            supplemental_losses = {}

        return logits, h_out, supplemental_losses

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        base_inputs = inputs['base']
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = base_inputs # unpack
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
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
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

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=torch.cuda.is_available()):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

