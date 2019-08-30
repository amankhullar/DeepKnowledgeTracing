import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SkillLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.):
        super(SkillLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=self.drop_prob if num_layers > 1 else 0.)

    def forward(self, student_questions, original_ques):
        
