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
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

    def get_deprel_emb(self):
        return self.gcn_model.get_deprel_embedding()

    def get_gcn_parameters(self):
        return self.gcn_model.get_gcn_parameters()

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.deprel_side = opt['deprel_emb_dim'] ## set equal to hidden_dim mostly
        if opt['adj_type'] != 'regular' or opt['deprel_attn']:
            deprel_emb_dim = self.deprel_side
        # regular adjacency matrix, thus fill with dummy weight
        else:
            deprel_emb_dim = 1
        self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), deprel_emb_dim, padding_idx=0)
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.deprel_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
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

    def forward(self, inputs):
        if self.opt['dataset'] == 'tacred':
            words, masks, pos, ner, deprel, head, subj_pos, obj_pos = inputs  # unpack
        else:
            words, masks, pos, deprel, head, subj_pos, obj_pos = inputs  # unpack

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        # print('words: {} | masks: {} | pos: {} | ner: {} | head: {} | maxlen: {}'.format(
        #     words.shape, masks.shape, pos.shape, ner.shape, head.shape, maxlen))

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos, deprel):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            deprel = deprel.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i], deprel[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            # return adj.cuda() if self.opt['cuda'] else Variable(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data, deprel.data)
        # print('adj: {}'.format(adj.shape))
        h, pool_mask = self.gcn(adj, inputs)
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out

    def get_gcn_parameters(self):
        return self.gcn.get_gcn_parameters()

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + (opt['ner_dim'] * int(self.opt['dataset'] == 'tacred'))

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
        self.token_dropout = EmbeddingDropout(opt['emb_dropout'])

        # gcn input encoding
        if opt['adj_type'] in ['diagonal_deprel']:
            self.preprocessor = nn.Linear(self.in_dim, self.mem_dim)
            self.in_dim = self.mem_dim
        elif opt['adj_type'] in ['full_deprel', 'regular']:
            self.W = nn.ModuleList()
            for layer in range(self.layers):
                input_dim = opt['deprel_emb_dim']
                output_input_dim = self.in_dim if layer == 0 else self.mem_dim
                output_dim = input_dim * self.mem_dim
                self.W.append(nn.Linear(output_input_dim, output_dim, bias=True))
        else:
            # gcn layer
            self.W = nn.ModuleList()
            for layer in range(self.layers):
                input_dim = self.in_dim if layer == 0 else self.mem_dim
                if opt['adj_type'] == 'concat_deprel' and layer == 0:
                    input_dim += self.deprel_emb.weight.shape[1]
                    self.deprel_dropout = nn.Dropout(opt.get('deprel_dropout', 0.0))
                self.W.append(nn.Linear(input_dim,  self.mem_dim))

        if opt['deprel_attn']:
            self.attn_proj = nn.Linear(opt['deprel_emb_dim'], 1, bias=False)

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
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        In the case that the input vector is completely masked, the return value of this function is
        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
        that we deal with this case relies on having single-precision floats; mixing half-precision
        floats with fully-masked vectors will likely give you ``nans``.
        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
        lower), the way we handle masking here could mess you up.  But if you've got logit values that
        extreme, you've got bigger problems than this.
        """
        if mask is not None:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.
            vector = vector + (mask + 1e-45).log()
        return torch.nn.functional.softmax(vector, dim=dim)

    def forward(self, adj, inputs):
        if self.opt['dataset'] == 'tacred':
            words, masks, pos, ner, deprel, head, subj_pos, obj_pos = inputs  # unpack
        else:
            words, masks, pos, deprel, head, subj_pos, obj_pos = inputs  # unpack

        if len(words.shape) > 2:
            word_embs = words
        else:
            # word_embs = self.emb(words)
            word_embs = self.token_dropout(self.emb, words)
            # word_embs = words
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0 and self.opt['dataset'] == 'tacred':
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        if self.opt['adj_type'] == 'diagonal_deprel':
            # Encode parameters for deprel changes
            gcn_inputs = self.preprocessor(gcn_inputs)
        batch_size, max_len, encoding_dim = gcn_inputs.shape
        if self.opt['adj_type'] == 'full_deprel' or self.opt['deprel_attn']:
            deprel_emb = self.deprel_emb(deprel)
        elif self.opt['adj_type'] != 'regular':
            deprel_adj = self.deprel_emb(adj.type(torch.int64)) # [B,T,T,H]
            deprel_adj = deprel_adj.reshape((batch_size, max_len, max_len, -1))
        else:
            batch_size, max_len, encoding_dim = gcn_inputs.shape
            deprel_adj = None
        # gcn layer
        adj_matrix = torch.where(adj != 0, torch.ones_like(adj), torch.zeros_like(adj)).type(torch.float32)
        denom = adj_matrix.sum(2).unsqueeze(2) + 1
        mask = (adj_matrix.sum(2) + adj_matrix.sum(1)).eq(0).unsqueeze(2)
        if self.opt['deprel_attn']:
            # [B,T,D]x[D,1]->[B,T,1]->[B,T]->[B,1,T]->[B,T,T]
            deprel_attn = self.attn_proj(deprel_emb).permute(0, 2, 1).repeat((1, max_len, 1))
            adj_matrix = self.masked_softmax(deprel_attn, adj_matrix) * adj_matrix
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj_matrix = torch.zeros_like(adj_matrix)
        elif self.opt['adj_type'] == 'concat_deprel':
            deprel_embs = self.deprel_dropout(self.deprel_emb(deprel))
            gcn_inputs = torch.cat([gcn_inputs, deprel_embs], dim=-1)
        for l in range(self.layers):
            if self.opt['adj_type'] == 'regular':
                # [B,T1,T2] x [B,T2,H] -> [B,T1,H]
                # Ax = adj_matrix.bmm(gcn_inputs)
                # AxW = self.W[l](Ax)
                # AxW = AxW + self.W[l](gcn_inputs)  # self loop

                ########################################################################################################
                ########################################## Weight Extractions ##########################################
                ########################################################################################################
                # [D,T,H]
                weight_l = self.W[l].weight.reshape((self.opt['deprel_emb_dim'], -1, self.mem_dim))
                # [D, H]
                bias_l = self.W[l].bias.reshape((self.opt['deprel_emb_dim'], self.mem_dim))
                ########################################################################################################
                ######################### Forward Dependency Relation Traversal and Aggregation ########################
                ########################################################################################################
                # Extract all dependency relations that are children of current node
                forward_adj_matrix = torch.where((0 < adj) * (adj < constant.DEPREL_FORWARD_BOUND),
                                                 torch.ones_like(adj),
                                                 torch.zeros_like(adj)). \
                    type(torch.float32)
                # [B,N,D]
                forward_deprel_embs = torch.ones((batch_size, max_len, self.opt['deprel_emb_dim']))
                if self.opt['cuda']:
                    forward_deprel_embs = forward_deprel_embs.cuda()
                # [B,N,H]
                forward_encs = self.traverse_deprel(token_encs=gcn_inputs,
                                                    deprel_embs=forward_deprel_embs,
                                                    weight=weight_l,
                                                    bias=bias_l)
                # [B,N,N]x[B,N,H]->[B,N,H]
                forward_combined = forward_adj_matrix.bmm(forward_encs)
                AxW = forward_combined
                ########################################################################################################
                ######################### Reverse Dependency Relation Traversal and Aggregation ########################
                ########################################################################################################
                # Extract all dependency relations that are parents of current node. These are reverse connections
                if not self.opt['deprel_directed']:
                    reverse_adj_matrix = torch.where(
                        (constant.DEPREL_FORWARD_BOUND < adj) * (adj < constant.DEPREL_REVERSE_BOUND),
                        torch.ones_like(adj),
                        torch.zeros_like(adj)). \
                        type(torch.float32)
                    # [B,N,D]
                    # reverse_deprel_embs = self.deprel_emb(deprel + constant.DEPREL_FORWARD_BOUND)
                    reverse_deprel_embs = torch.ones((batch_size, max_len, self.opt['deprel_emb_dim']))
                    if self.opt['cuda']:
                        reverse_deprel_embs = reverse_deprel_embs.cuda()
                    # [B,N,H]
                    reverse_encs = self.traverse_deprel(token_encs=gcn_inputs,
                                                        deprel_embs=reverse_deprel_embs,
                                                        weight=weight_l,
                                                        bias=bias_l)
                    # [B,N,N]x[B,N,H]->[B,N,H]
                    reverse_commbined = reverse_adj_matrix.bmm(reverse_encs)
                    AxW += reverse_commbined
                ########################################################################################################
                ########################################## Self Loop Traversal #########################################
                ########################################################################################################
                if self.opt['deprel_self_loop']:
                    # [1,D]
                    # self_loop_lookup = torch.ones((1, 1)).type(torch.LongTensor) * constant.SELF_LOOP_INDEX
                    # if self.opt['cuda']:
                    #     self_loop_lookup = self_loop_lookup.cuda()
                    # self_loop_emb = self.deprel_emb(self_loop_lookup)
                    self_loop_emb = torch.ones((1, 1, self.opt['deprel_emb_dim']))
                    if self.opt['cuda']:
                        self_loop_emb = self_loop_emb.cuda()
                    # [B,N,H]
                    self_loop_encs = self.traverse_self_loop(token_encs=gcn_inputs,
                                                             self_loop_emb=self_loop_emb,
                                                             weight=weight_l,
                                                             bias=bias_l)
                    AxW += self_loop_encs
            elif self.opt['adj_type'] == 'diagonal_deprel':
                # [B,T1,H] -> [B,1,T2,H]
                layer_inputs = gcn_inputs.view((batch_size, 1, max_len, encoding_dim))
                # [B,T1,T2,H] x [B,1,T2,H] -> [B,T1,T2,H] -> [B,T1,H]
                Ax = (deprel_adj * layer_inputs).sum(2)
                AxW = self.W[l](Ax)
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
            elif self.opt['adj_type'] == 'full_deprel':
                ########################################################################################################
                ########################################## Weight Extractions ##########################################
                ########################################################################################################
                # [D,T,H]
                weight_l = self.W[l].weight.reshape((self.opt['deprel_emb_dim'], -1, self.mem_dim))
                # [D, H]
                bias_l = self.W[l].bias.reshape((self.opt['deprel_emb_dim'], self.mem_dim))
                ########################################################################################################
                ######################### Forward Dependency Relation Traversal and Aggregation ########################
                ########################################################################################################
                # Extract all dependency relations that are children of current node
                forward_adj_matrix = torch.where((0 < adj) * (adj < constant.DEPREL_FORWARD_BOUND),
                                                 torch.ones_like(adj),
                                                 torch.zeros_like(adj)).\
                    type(torch.float32)
                # Maybe randomly drop edges
                forward_adj_matrix = self.maybe_drop_edges(forward_adj_matrix)
                # [B,N,D]
                forward_deprel_embs = self.deprel_emb(deprel)
                # Maybe forget dependency relation embeddings
                forward_deprel_embs = self.maybe_forget_deprels(forward_deprel_embs)
                # Mix between dependency relation and no-relation on all edges
                forward_ones = torch.ones((batch_size, max_len, self.opt['deprel_emb_dim']))
                if self.opt['cuda']:
                    forward_ones = forward_ones.cuda()
                deprel_alpha = self.opt.get('deprel_alpha', 1.0)

                if l >= self.opt['deprel_max_depth']:
                    forward_deprel_embs = forward_ones

                # forward_deprel_embs = forward_deprel_embs * deprel_alpha + (1 - deprel_alpha) * forward_ones
                # [B,N,H]
                forward_encs = self.traverse_deprel(token_encs=gcn_inputs,
                                                    deprel_embs=forward_deprel_embs,
                                                    weight=weight_l,
                                                    bias=bias_l)
                # [B,N,N]x[B,N,H]->[B,N,H]
                forward_combined = forward_adj_matrix.bmm(forward_encs)
                AxW = forward_combined
                ########################################################################################################
                ######################### Reverse Dependency Relation Traversal and Aggregation ########################
                ########################################################################################################
                # Extract all dependency relations that are parents of current node. These are reverse connections
                if not self.opt['deprel_directed']:
                    reverse_adj_matrix = torch.where(
                        (constant.DEPREL_FORWARD_BOUND < adj) * (adj < constant.DEPREL_REVERSE_BOUND),
                        torch.ones_like(adj),
                        torch.zeros_like(adj)).\
                            type(torch.float32)
                    # Maybe drop edges randomly
                    reverse_adj_matrix = self.maybe_drop_edges(reverse_adj_matrix)
                    # [B,N,D]
                    reverse_deprel_embs = self.deprel_emb(deprel + constant.DEPREL_FORWARD_BOUND)
                    # Maybe forget dependency relation embeddings
                    reverse_deprel_embs = self.maybe_forget_deprels(reverse_deprel_embs)
                    reverse_ones = torch.ones((batch_size, max_len, self.opt['deprel_emb_dim']))
                    if self.opt['cuda']:
                        reverse_ones = reverse_ones.cuda()
                    if l >= self.opt['deprel_max_depth']:
                        reverse_deprel_embs = reverse_ones
                    # reverse_deprel_embs = reverse_deprel_embs * deprel_alpha + (1 - deprel_alpha) * reverse_ones
                    # [B,N,H]
                    reverse_encs = self.traverse_deprel(token_encs=gcn_inputs,
                                                        deprel_embs=reverse_deprel_embs,
                                                        weight=weight_l,
                                                        bias=bias_l)
                    # [B,N,N]x[B,N,H]->[B,N,H]
                    reverse_combined = reverse_adj_matrix.bmm(reverse_encs)
                    AxW += reverse_combined
                ########################################################################################################
                ########################################## Self Loop Traversal #########################################
                ########################################################################################################
                if self.opt['deprel_self_loop']:
                    # [1,D]
                    self_loop_lookup = torch.ones((1, 1)). type(torch.LongTensor) * constant.SELF_LOOP_INDEX
                    if self.opt['cuda']:
                        self_loop_lookup = self_loop_lookup.cuda()
                    self_loop_emb = self.deprel_emb(self_loop_lookup)

                    if l >= self.opt['deprel_max_depth']:
                        self_loop_emb = torch.ones((1, 1, self.opt['deprel_emb_dim']))
                        if self.opt['cuda']:
                            self_loop_emb = self_loop_emb.cuda()

                    # [B,N,H]
                    self_loop_encs = self.traverse_self_loop(token_encs=gcn_inputs,
                                                             self_loop_emb=self_loop_emb,
                                                             weight=weight_l,
                                                             bias=bias_l)
                    AxW += self_loop_encs

            elif self.opt['adj_type'] == 'concat_deprel':
                # [B,T1,H]
                Ax = adj_matrix.bmm(gcn_inputs)
                AxW = self.W[l](Ax)
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
            elif self.opt['adj_type'] == 'only_deprel':
                # [B,T1,T2,H] -> [B,T1,H]
                current_connections = deprel_adj.sum(2)
                layer_inputs = gcn_inputs + current_connections
                Ax = layer_inputs
                AxW = self.W[l](Ax)
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
            else:
                raise ValueError('Adjacency aggregation type not supported.')

            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask

    def get_gcn_parameters(self):
        return self.W

    def traverse_deprel(self, token_encs, deprel_embs, weight, bias):
        """
        Traverse dependency relation edges simultaneously.
        :param token_encs: Token embeddings | [B,N,T]
        :param deprel_embs: Deprel embeddings | [B,N,D]
        :param weight: Weight tensor | [D,T,H]
        :param bias: Bias tensor | [D,H]
        :return: deprel transformed encodings | [B,N,H]
        """
        # [B,N,D]o[B,N,T]->[B,N,D,T]
        deprel_op = torch.einsum('ijk,ijl->ijkl', deprel_embs, token_encs)
        # [B,N,D,T]x[D,T,H]->[B,N,H]
        deprel_transformed = torch.einsum('abcd,cde->abe', deprel_op, weight)
        deprel_bias = torch.einsum('ijk,kl->ijl', deprel_embs, bias)
        deprel_traversed = deprel_transformed + deprel_bias
        return deprel_traversed

    def traverse_self_loop(self, token_encs, self_loop_emb, weight, bias):
        """
        Traverse self loop relation
        :param token_encs:  Token embeddings | [B,N,T]
        :param self_loop_emb: Self loop embedding | [1,D]
        :param weight: Weight tensor | [D,T,H]
        :param bias: Bias tensor | [D,H]
        :return: Self loop traversed tensor | [B,N,H]
        """
        self_loop_emb = self_loop_emb.squeeze(0)
        # [1,D]x[D,T,H]->[1,T,H]->[T,H]
        sl_weight = torch.einsum('ij,jkl->ikl', self_loop_emb, weight).squeeze(0)
        # [B,N,T]x[T,H]->[B,N,H]
        sl_transformed = torch.einsum('ijk,kl->ijl', token_encs, sl_weight)
        # [1,D]x[D,H]->[1,H]->[1,1,H]
        sl_bias = torch.einsum('ij,jk->ik', self_loop_emb, bias).unsqueeze(0)
        sl_traversed = sl_transformed + sl_bias
        return sl_traversed

    def maybe_drop_edges(self, adj_matrix):
        """
        Randomly mask out dependency tree connections if desired,
        :param adj_matrix: Adjacency matrix representing parent to child connections | [B,N,N]
        :return: Masked adjacency matrix | [B,N,N]
        """
        if self.training and self.opt.get('edge_keep_prob', 1.0) < 1.0:
            keep_edges = torch.empty_like(adj_matrix, requires_grad=False).bernoulli_(self.opt['edge_keep_prob'])
            if self.opt['cuda']:
                keep_edges = keep_edges.cuda()
            remain_adj_matrix = keep_edges * adj_matrix
            return remain_adj_matrix
        else:
            return adj_matrix

    def maybe_forget_deprels(self, deprel_embs):
        """
        Randomly remove dependency relations if desired. Note: removing a dependency relation does not
        imply zeroing out its embeddings. Instead, it means the masked out relations are replaced with
        a 1-vector. This is because 0-vectors would also remove the edge.
        :param deprel_embs: Dependency relation embeddings | [B,N,D]
        :return: Masked out dependency relation embeddings | [B,N,D]
        """
        keep_prop = self.opt.get('deprel_keep_prop', 1.0)
        if self.training and keep_prop < 1.0:
            batch_size, num_token, emb_dim = deprel_embs.shape
            po_deprels = torch.empty((batch_size, num_token, 1), requires_grad=False).\
                bernoulli_(keep_prop).\
                repeat(1, 1, emb_dim)
            if self.opt['cuda']:
                po_embs = torch.where((po_deprels == 1).cuda(), deprel_embs, torch.ones_like(deprel_embs).cuda())
            else:
                po_embs = torch.where(po_deprels == 1, deprel_embs, torch.ones_like(deprel_embs))
            return po_embs
        return deprel_embs


def pool(h, mask, type='max'):
    if type == 'max':
        # print('h: {} | mask: {}'.format(h.shape, mask.shape))
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

