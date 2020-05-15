import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

# Copied from: https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout, batch_first=False):
        super(VariationalDropout).__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class EmbeddingDropout(nn.Module):
    def __init__(self, dropout=.5):
        super(EmbeddingDropout, self).__init__()
        self.dropout = dropout

    def forward(self, emb_matrix, input_values):
        unique_values = torch.unique(input_values)
        emb_mask = torch.zeros((emb_matrix.weight.shape[0], 1), dtype=torch.float32)
        unique_mask = torch.empty(unique_values.shape[0], 1,  requires_grad=False).bernoulli_(1 - self.dropout)
        emb_mask[unique_values, :] = unique_mask
        if input_values.is_cuda:
            emb_mask = emb_mask.cuda()
        masked_emb_matrix = emb_matrix.weight * emb_mask
        input_embs = F.embedding(input_values, masked_emb_matrix, 0, 2, False, False) / (1 - self.dropout)
        return input_embs


