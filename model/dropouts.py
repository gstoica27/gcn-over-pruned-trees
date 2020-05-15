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
        emb_mask = torch.zeros((emb_matrix.weight.shape[0], 1), dtype=torch.float32)
        unique_mask = torch.empty(unique_values.shape[0], 1,  requires_grad=False).bernoulli_(1 - self.dropout)
        emb_mask[unique_values, :] = unique_mask
        if input_values.is_cuda:
            emb_mask = emb_mask.cuda()
        masked_emb_matrix = emb_matrix.weight * emb_mask
        input_embs = F.embedding(input_values, masked_emb_matrix, 0, 2, False, False) / (1 - self.dropout)
        return input_embs


