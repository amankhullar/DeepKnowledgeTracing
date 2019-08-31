import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def masked_softmax(logits, mask, dim=-1):
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    probs = F.softmax(masked_logits, dim)
    return probs

class SkillLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, drop_prob=0.):
        super(SkillLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=self.drop_prob if num_layers > 1 else 0.)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # convert python list into tensor
        lengths = torch.Tensor(lengths)

        # Generating mask
        mask = torch.zeros(x.size(0), x.size(1), x.size(2)//2)
        for idx in range(mask.size(0)):
            mask[idx][:int(lengths[idx])] = 1

        # save the original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        #sort by lengths and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]         # (batch_size, ques_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.lstm(x)     # (batch_size, ques_len, hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]       # (batch_size, ques_len, hidden_size)

        # Apply dropout
        x = F.dropout(x, self.drop_prob, self.training)     # (batch_size, ques_len, hidden_size)

        # Apply output layer and softmax function
        out_dist = self.out(x)       # (batch_size, ques_len, output_size)
        out_dist = masked_softmax(out_dist, mask)      # (batch_size, ques_len, output_size)

        return out_dist
