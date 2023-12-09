import numpy as np
import torch
import torch.nn as nn


def get_positional_encoding(max_seq_len, embed_dim):

    positional_encoding=torch.zeros(max_seq_len, embed_dim).cuda()
    for pos in range(max_seq_len):
        for i in range(embed_dim):
            if(i%2==0):
                positional_encoding[pos,i]=torch.sin(pos / torch.tensor(10000**(2 * i / embed_dim)))
            else:
                positional_encoding[pos,i]=torch.cos(pos / torch.tensor(10000**(2 * i /embed_dim)))
    return positional_encoding
