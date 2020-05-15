import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchnlp.nn import *
from model.dropouts import VariationalDropout


# Copied from https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, x_dropout=0.0, h_dropout=0.0):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.x_dropout = x_dropout
        self.h_dropout = h_dropout
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = WeightDropLinear(self.mem_dim, 3 * self.mem_dim, weight_dropout=self.h_dropout)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = WeightDropLinear(self.mem_dim, self.mem_dim, weight_dropout=self.h_dropout)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

# Batched Version for Child-Sum Tree-LSTM. Model logic details are described here:
# https://www.overleaf.com/project/5ebc5ed89e56a600019484d2
class BatchedChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, on_cuda, x_dropout=0.0, h_dropout=0.0):
        super(BatchedChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.on_cuda = on_cuda
        self.x_dropout = x_dropout
        self.h_dropout = h_dropout
        self.x_iouf = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.h_iou = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.h_f = nn.Linear(self.mem_dim, self.mem_dim)

        self.input_dropout = VariationalDropout(dropout=self.x_dropout)
        self.output_dropout = VariationalDropout(dropout=self.x_dropout)

    def step(self, inputs, child_hidden, child_cell, child_mask):
        """
        Forward cell of batched child sum lstm
        :param inputs: token encodings. Size                [B,T1,H]
        :param child_hidden: hidden states of all children  [B,T1,T2,H]
        :param child_cell: cell state of tree nodes         [B,T1,T2,H]
        :param child_mask: mask, 1=child, 0=don't use       [B,T1,T2,1]
        :return:
            - hidden states for each leaf node              [B,T1,H]
            - cell states for each leaf node                [B,T1,H]
        """
        num_children = child_hidden.shape[2]
        h_j = (child_hidden * child_mask).sum(2)          # [B,T1,H]
        x_iouf = self.x_iouf(inputs)                      # [B,T1,4H]
        h_iou = self.h_iou(h_j)                           # [B,T1,3H]
        x_i, x_o, x_u, x_f = torch.split(x_iouf, int(x_iouf.shape[-1] / 4), dim=2)
        h_i, h_o, h_u = torch.split(h_iou, int(h_iou.shape[-1] / 3), dim=2)

        i_j = torch.sigmoid(x_i + h_i)
        o_j = torch.sigmoid(x_o + h_o)
        u_j = torch.tanh(x_u + h_u)

        h_f = self.h_f(child_hidden)  # [B,T1,T2,H]
        f_jk = torch.sigmoid(x_f[:, :num_children, :].unsqueeze(1) + h_f)        # [B,T1,T2,H]
        # [B,T1,T2,H]x[B,T1,T2,H]x[B,T1,T2,1] -> [B,T1,H]
        c_j_rhs = (f_jk * child_cell * child_mask).sum(2)
        c = i_j * u_j + c_j_rhs                         # [B,T1,H]
        # [B,T1,H]x[B,T1,H]
        h = o_j * torch.tanh(c)
        return h, c

    def hidden_dropout(self):
        self.h_iou.weight = torch.nn.functional.dropout(self.h_iou.weight,
                                                             p=self.h_dropout,
                                                             training=self.training).contiguous()
        self.h_f.weight.data = torch.nn.functional.dropout(self.h_f.weight.data,
                                                             p=self.h_dropout,
                                                             training=self.training).contiguous()

    def forward(self, token_encodings, trees, child_mask, max_depth):
        # self.hidden_dropout()
        batch_size, token_size, hidden_dim = token_encodings.shape
        # hidden: [B,T1+2,H] cell: [B,T1+2,H]
        hidden_state, cell_state = self.init_zero_state(
            batch_size=batch_size, token_size=token_size,
            hidden_dim=self.mem_dim, use_cuda=self.on_cuda
        )
        dropped_encodings = self.input_dropout(token_encodings)
        for level in range(max_depth):
            # Flatten hidden/cell state in order perform lookup
            # [B,T1,H] -> [BxT1,H]
            flat_hidden_state = hidden_state.reshape((-1, hidden_state.shape[-1]))
            flat_cell_states = cell_state.reshape((-1, hidden_state.shape[-1]))
            # [BxT1,H] ([B,T1,T2]) -> [B,T1,T2,H]
            child_hidden_states = F.embedding(trees.type(torch.long), flat_hidden_state, 0, 2, False,
                                              False)  # [B,T1,T2,H]
            child_cell_states = F.embedding(trees.type(torch.long), flat_cell_states, 0, 2, False, False)

            new_hidden_states, new_cell_states = self.step(
                dropped_encodings,  # [:,:max_bottom_offset,:],
                child_hidden_states,
                child_cell_states,
                child_mask
            )
            # Add "padded" cells to hidden and cell states so that 2-indexed Adjacency
            # matrix works + we preserve same cell/hidden state for precomputed nodes
            # throughout depth iterations.
            # [B,2,H]
            hidden_pad = Variable(torch.zeros((batch_size, 2, hidden_state.shape[-1]),
                                              dtype=torch.float32),
                                  requires_grad=False)
            cell_pad = Variable(torch.zeros((batch_size, 2, hidden_state.shape[-1]),
                                            dtype=torch.float32),
                                requires_grad=False)

            if self.on_cuda:
                hidden_pad = hidden_pad.cuda()
                cell_pad = cell_pad.cuda()
            # Haven't reached tree root yet, pad for indexing again
            if level < max_depth - 1:
                # [[B,2,H];[B,T1,H]] -> [B,T1+2,H]
                hidden_state = torch.cat([hidden_pad, new_hidden_states], dim=1)
                cell_state = torch.cat([cell_pad, new_cell_states], dim=1)
            else:
                hidden_state = new_hidden_states
                cell_state = new_cell_states

        hidden_state = self.output_dropout(hidden_state)
        return hidden_state

    def init_zero_state(self, batch_size, token_size, hidden_dim, use_cuda=False):
        # token_size +1 necessary because Adjacencies are 2 indexed. 0 is the padding
        # index, and 1 is the actual "zero" hidden state used by the leaf nodes
        hidden_shape = (batch_size, token_size + 2, hidden_dim)
        cell_shape = (batch_size, token_size + 2, hidden_dim)
        h0 = Variable(torch.zeros(*hidden_shape), requires_grad=False)
        c0 = Variable(torch.zeros(*cell_shape), requires_grad=False)
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0
