import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

# Copied from: https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
class VariationalDropout(nn.Module):
    def __init__(self, dropout=.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, inputs):
        if not self.training or self.dropout <= 0.:
            return inputs
        batch_size, num_tokens, encoding_dim = inputs.shape
        mask = inputs.new_empty(batch_size, 1, encoding_dim, requires_grad=False).bernoulli_(1 - self.dropout)
        if inputs.is_cuda:
            mask = mask.cuda()
        masked_inputs = inputs.masked_fill(mask == 0, 0) / (1 - self.dropout)
        return masked_inputs


class EmbeddingDropout(nn.Module):
    def __init__(self, dropout=.5):
        super(EmbeddingDropout, self).__init__()
        self.dropout = dropout

    def forward(self, emb_matrix, input_values):
        if not self.training or self.dropout <= 0.:
            return emb_matrix(input_values)
        unique_values = torch.unique(input_values)
        dropout_mask = torch.zeros((input_values.shape[0], emb_matrix.weight.shape[0]), dtype=torch.float32)
        unique_mask = torch.empty(input_values.shape[0], unique_values.shape[0],  requires_grad=False).bernoulli_(1 - self.dropout)
        dropout_mask[:, unique_values] = unique_mask  # [B, V]
        if input_values.is_cuda:
            dropout_mask = dropout_mask.cuda()
        input_embs = F.embedding(input_values.weight, emb_matrix, padding_idx=0, sparse=True)                 # [B, L, E]
        input_embs_mask = torch.gather(dropout_mask, 1, input_values, sparse_grad=True).unsqueeze(-1)  # [B, L, 1]
        return input_embs * input_embs_mask / (1 - self.dropout)


