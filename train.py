import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import data

from datasets import SkillDataset

def main(train_file_path, batch_size):
    skill_dataset = SkillDataset(train_file_path)
    train_skill_loader = torch.utils.data.DataLoader(skill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    

if __name__ == "__main__":
    train_file_path = ""
    batch_size = 1
    main(train_file_path, batch_size)
