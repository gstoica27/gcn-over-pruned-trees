import torch
import torch.nn as nn
import numpy as np


def test_full_deprel():
    deprel_emb_dim = 5
    token_num = 4
    batch_size = 5
    token_emb_dim = 6
    num_deprel = 10
    unique_tokens = 15
    hidden_dim = 8

    deprel_emb = nn.Embedding(num_deprel, deprel_emb_dim)
    token_emb = nn.Embedding(unique_tokens, token_emb_dim)

    token_indices = torch.from_numpy(np.random.randint(0, unique_tokens, size=20).reshape((batch_size, token_num)))
    deprel_indices = torch.from_numpy(np.random.randint(0, num_deprel, size=20).reshape((batch_size, token_num)))

    projection = nn.Linear(token_emb_dim, deprel_emb_dim * hidden_dim)
    weight = projection.weight.reshape((deprel_emb_dim, token_emb_dim, hidden_dim)) # [D,T,H]
    bias = projection.bias.reshape((deprel_emb_dim, hidden_dim))                  # [D,H]

    batch_deprel_emb = deprel_emb(deprel_indices)  # [B,N,T]
    batch_token_emb = token_emb(token_indices)  # [B,N,D]
    ##################################### Einsum Method ######################################
    # [B,N,D]x[B,N,T]->[B,N,D,T]
    outer_product = torch.einsum('ijk,ija->ijka', batch_deprel_emb, batch_token_emb)
    # [B,N,D,T]x_{12}[D,T,H]->[B,N,H]
    deprel_gcn_vector = torch.einsum('abcd,cde->abe', outer_product, weight)
    # [B,N,D]x[D,H]->[B,N.H]
    deprel_gcn_bias = torch.einsum('abc,ce->abe', batch_deprel_emb, bias)
    ###################################### Tiling Method ######################################
    weight2 = weight.permute(0, 2, 1)
    deprel_weight = torch.matmul(batch_deprel_emb.reshape(-1, deprel_emb_dim),
                                 weight2.reshape(deprel_emb_dim, -1))
    deprel_weight = deprel_weight.reshape(-1, hidden_dim, token_emb_dim)
    batch_token_emb = batch_token_emb.reshape(-1, token_emb_dim).unsqueeze(-1)
    gcn_vector = torch.bmm(deprel_weight, batch_token_emb).squeeze(-1).reshape(batch_size, token_num, hidden_dim)

    difference = torch.pow(deprel_gcn_vector - gcn_vector, 2)
    summed_difference = torch.sum(difference)
    assert abs(summed_difference.detach().numpy() - 0) < .0001


if __name__ == '__main__':
    test_full_deprel()