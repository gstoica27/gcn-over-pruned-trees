"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from model.tree_lstm import BatchedChildSumTreeLSTM
from utils import constant, torch_utils
from model.dropouts import EmbeddingDropout

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
        outputs, pooling_output, components = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output, components

    def get_deprel_emb(self):
        return self.gcn_model.get_deprel_embedding()

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        # self.deprel_side = opt['deprel_emb_dim'] ## set equal to hidden_dim mostly
        # self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), self.deprel_side, padding_idx=0)
        # embeddings = (self.emb, self.pos_emb, self.ner_emb, self.deprel_emb)
        if opt.get('deprel_emb_dim', 0) > 0:
            self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['deprel_emb_dim'], padding_idx=0)
        else:
            self.deprel_emb = None
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.deprel_emb)
        self.init_embeddings()

        # gcn layer
        self.tree_lstm_wrapper = TreeLSTMWrapper(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        if opt ['node_pooling']:
            multiplier = 3
        else:
            multiplier = 2
        in_dim = opt['hidden_dim']* multiplier
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def get_deprel_embedding(self):
        return self.deprel_emb.weight

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

    def find_adj_bounds(self, adj_matrix):
        """
        Find actual bounds over adjacency matrix. I.e. how many cols/rows are actually needed.
         Can significantly reduce memory load with this
        :param adj_matrix: np.array(max_len, max_len)
        :return: (row_top_offset, row_bottom_offset, col_right_offset)
        """
        if len(adj_matrix.shape) == 3:
            matrix = adj_matrix[0]
        else:
            matrix = adj_matrix
        col_right_offset = max([np.sum(row != 0) for row in matrix])
        is_pad_row = np.any(matrix, axis=1)
        non_pad_rows = np.where(is_pad_row)[0]
        row_top_offset = min(non_pad_rows)
        row_bottom_offset = max(non_pad_rows) + 1
        return (row_top_offset, row_bottom_offset, col_right_offset)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, deprel):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), \
                                             words.cpu().numpy(), \
                                             subj_pos.cpu().numpy(), \
                                             obj_pos.cpu().numpy()
            deprel = deprel.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune,
                                  subj_pos[i], obj_pos[i], deprel[i]) for i in range(len(l))]
            trees, subject_trees, object_trees = zip(*trees)
            # Maximum tree depth in batch. Used to know how many LSTM steps are needed
            max_depth = max(list(map(lambda node:node.depth, trees)))
            adj = [tree_to_adj(maxlen, tree,
                               batch_idx=idx,
                               directed=False,
                               self_loop=False).reshape(1, maxlen, maxlen) for idx, tree in enumerate(trees)]
            adj_bounds = [self.find_adj_bounds(adj_matrix) for adj_matrix in adj]
            adj = np.concatenate(adj, axis=0)

            row_top_offsets, row_bottom_offsets, col_right_offsets = zip(*adj_bounds)
            max_bottom_offset = max(row_bottom_offsets)
            max_col_offset = max(col_right_offsets)
            adj = adj[:, :, :max_col_offset]

            adj = torch.from_numpy(adj).type(torch.long) # [B,T1,T2]
            adj.requires_grad_(False)
            adj = Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
            component_trees = (trees, subject_trees, object_trees)
            return adj, max_depth, max_bottom_offset, component_trees

        tree_adj, max_depth, max_bottom_offset, component_trees = inputs_to_tree_reps(
            head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, deprel.data
        )
        tree_encodings, pool_mask = self.tree_lstm_wrapper(tree_adj, inputs, max_depth, max_bottom_offset)
        
        # pooling
        if self.opt.get('node_pooling', False):
            batch_size = words.shape[0]
            sentence_roots, subject_roots, object_roots = component_trees
            root_mask = self.tree_to_mask(sentence_roots, batch_size=batch_size, token_len=maxlen)
            subject_mask = self.tree_to_mask(subject_roots, batch_size=batch_size, token_len=maxlen)
            object_mask = self.tree_to_mask(object_roots, batch_size=batch_size, token_len=maxlen)
            # Extract respective root indices
            sentence_rep = (tree_encodings * root_mask).sum(1)
            subject_rep = (tree_encodings * subject_mask).sum(1)
            object_rep = (tree_encodings * object_mask).sum(1)
            outputs = torch.cat([sentence_rep, subject_rep, object_rep], dim=1)

        else:
            subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
            # subj_mask = subj_pos.eq(0).unsqueeze(2)
            # obj_mask = obj_pos.eq(0).unsqueeze(2)
            # pool_mask = torch.logical_xor(pool_mask.eq(0), (subj_mask + obj_mask))
            # subj_mask, obj_mask, pool_mask = subj_mask.eq(0),  obj_mask.eq(0), pool_mask.eq(0)
            pool_type = self.opt['pooling']
            sentence_rep = pool(tree_encodings, pool_mask, type=pool_type)
            subject_rep = pool(tree_encodings, subj_mask, type=pool_type)
            object_rep = pool(tree_encodings, obj_mask, type=pool_type)
            # outputs = torch.cat([sentence_rep, subject_rep, object_rep], dim=1)
            outputs = torch.cat([subject_rep, object_rep], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, sentence_rep, (subject_rep, object_rep)

    def tree_to_mask(self, trees, batch_size, token_len):
        token_indices = np.array([tree.idx for tree in trees])
        batch_indices = np.arange(batch_size)
        # indices = list(zip(batch_indices, token_indices))
        # [B,T]
        mask = torch.zeros((batch_size, token_len), dtype=torch.float32)
        # Replace tree index elements in each batch sample with 1
        mask[batch_indices, token_indices] = torch.ones_like(torch.from_numpy(batch_indices), dtype=torch.float32)
        # Expand dims for masking
        mask.unsqueeze_(-1)
        if self.opt['cuda']:
            mask = mask.cuda()
        return mask

class TreeLSTMWrapper(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(TreeLSTMWrapper, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb, self.deprel_emb = embeddings
        # self.emb, self.pos_emb, self.ner_emb = embeddings
        self.emb_dropout = EmbeddingDropout(opt['emb_dropout'])
        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True,
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])
        self.tree_lstms = nn.ModuleList()
        for layer in range(opt.get('num_tree_lstms', 1)):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.tree_lstms.append(BatchedChildSumTreeLSTM(in_dim=input_dim,
                                                           mem_dim=self.mem_dim,
                                                           on_cuda=opt['cuda'],
                                                           x_dropout=opt['tree_x_dropout'],
                                                           h_dropout=opt['tree_h_dropout'],
                                                           deprel_emb=opt.get('deprel_emb_dim', 0)))

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

    def forward(self, trees, inputs, max_depth, max_bottom_offset):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        word_embs = self.emb_dropout(self.emb, words)
        # word_embs = self.emb(words)
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

        # [B,T1,T2] -> [B,T1,T2,1]. This is effectively a type of adjacency matrix
        mask = trees.eq(0).eq(0).unsqueeze(-1).type(torch.float32)
        # mask = torch.where(trees != 0, torch.ones_like(trees), torch.zeros_like(trees)).type(torch.float32)
        # Adjacency matrix is 2-indexed, so anything less than 2 is treated as a "PAD" value, and masked out
        # in sentence pooling downstream
        sentence_mask = torch.where(trees > 1, torch.ones_like(trees), torch.zeros_like(trees)).type(torch.float32)
        directed = False
        if not directed:
            # Make adjacency matrix square for undirectedness if necessary
            sentence_mask = self.maybe_squarify_matrix(sentence_mask)
            sentence_mask += sentence_mask.permute(0, 2, 1)
        sentence_mask = (sentence_mask.sum(2) + sentence_mask.sum(1)).eq(0).unsqueeze(2)

        lstm_inputs = gcn_inputs
        for tree_lstm in self.tree_lstms:
            if self.deprel_emb is not None:
                batch_deprel_embs = self.deprel_emb(deprel)
                batch_size, token_len, emb_dim = batch_deprel_embs.shape
                flat_deprel_embs = batch_deprel_embs.reshape((batch_size * token_len, emb_dim))
                child_deprel_embs = F.embedding(trees.type(torch.long), flat_deprel_embs, 0, 2, False, False)
            else:
                child_deprel_embs = None
            lstm_inputs = tree_lstm(lstm_inputs, trees, mask, max_depth, child_deprel_embs)
        hidden_state = lstm_inputs

        return hidden_state, sentence_mask

    def maybe_squarify_matrix(self, matrix):
        batch_size, height, width = matrix.shape
        pad_amount = abs(height - width)
        if width < height:
            pad_tensor = torch.zeros((batch_size, height, pad_amount))
            if self.opt['cuda']: pad_tensor = pad_tensor.cuda()
            padded_matrix = torch.cat([matrix, pad_tensor], dim=2)
        elif height < width:
            pad_tensor = torch.zeros((batch_size, pad_amount, width))
            if self.opt['cuda']: pad_tensor = pad_tensor.cuda()
            padded_matrix = torch.cat([matrix, pad_tensor], dim=1)
        else:
            padded_matrix = matrix
        return padded_matrix

    def init_zero_state(self, batch_size, token_size, hidden_dim, use_cuda=False):
        # token_size +1 necessary because Adjacencies are 2 indexed. 0 is the padding
        # index, and 1 is the actual "zero" hidden state used by the leaf nodes
        hidden_shape = (batch_size, token_size + 2, hidden_dim)
        cell_shape = (batch_size, token_size + 2, hidden_dim)
        h0 =  Variable(torch.zeros(*hidden_shape), requires_grad=False)
        c0 =  Variable(torch.zeros(*cell_shape), requires_grad=False)
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

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

