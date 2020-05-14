import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
########################################################################################################################
#             Code Copied from https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py             #
########################################################################################################################

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

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

class BatchedChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(BatchedChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.x_iouf = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.h_iou = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.h_f = nn.Linear(self.mem_dim, self.mem_dim)

    def forward(self, inputs, child_hidden, child_cell, child_mask):
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