import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SkillLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.):
        super(SkillLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=self.drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # convert python list into tensor
        lengths = torch.Tensor(lengths)

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
        x = F.dropout(x, self.drop_prob, self.training)

        return x
