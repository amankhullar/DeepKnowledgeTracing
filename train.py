import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import SkillDataset

def main(train_file_path, batch_size):
    skill_dataset = SkillDataset(train_file_path)
    train_skill_loader = torch.utils.data.DataLoader(skill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    for (student, student_orig) in train_skill_loader:
        print("Student size is : {}".format(student.size()))
        print("Student padded value is  : {}".format(student))
        print("Original student values are : {}".format(student_orig))
        print("Original student type is : {}".format(type(student_orig)))
        break

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(base_dir, 'data', '0910_a_test.csv')
    batch_size = 3
    main(train_file_path, batch_size)
