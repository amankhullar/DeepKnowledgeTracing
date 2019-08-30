import csv
import os
import sys

import torch
from torch.utils.data import Dataset

class SkillDataset(Dataset):
    """
    A Pytorch dataset class to be used in the Pytorch Dataloader to create text batches
    """
    def __init__(self, data_file):
        super(SkillDataset, self).__init__()
        try:
            with open(data_file, 'r') as f:
                rows = csv.reader(f, delimiter=',')
                idx = 0
                self.max_q = 0
                self.max_skill = 0
                self.students_data = []
                curr_q = curr_skill = curr_res = ''
                for row in rows:
                    if idx % 3 == 0:
                        curr_q = row
                        self.max_q = max(self.max_q, int(row[0]))
                    elif idx % 3 == 1:
                        curr_skill = row
                        self.max_skill = max(self.max_skill, max(map(int, row)))
                    else:
                        curr_res = row
                        self.students_data.append((curr_q, curr_skill, curr_res))
                    idx += 1
        except Exception as e:
            print("Could not find dataset file : " + str(e))
            sys.exit()
        else:
            assert idx % 3 == 0, "Incomplete dataset"

    def __len__(self):
        return len(self.students_data)

    def __getitem__(self, idx):
        print(self.students_data[idx][0])
        student_ques = torch.zeros(self.max_q, 2*(self.max_skill+1))                  # (Padded tensor : (max_ques,2 * (max_skills+1)) of 0s)
        for ques in range(int(self.students_data[idx][0][0])):
            student_ques[ques][int(self.students_data[idx][1][ques])] = 1         # (one-hot encoding of skills)
            if int(self.students_data[idx][-1][ques]) == 1:
                student_ques[ques][int(self.students_data[idx][1][ques]) + self.max_skill] = 1          # (correct answer)
        return student_ques, self.students_data[idx]                    # student_ques : (max_ques, 2*(max_skills+1))
        