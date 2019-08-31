import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import SkillDataset
from models import SkillLSTM

from args import get_train_args
from tqdm import tqdm

def save_checkpoint(model, save_path, device=torch.device('cpu')):
    ckpt_dict = {
        'model_name': model.__class__.__name__,
        'model_state': model.cpu().state_dict(),
    }
    model.to(device)

    checkpoint_path = save_path

    torch.save(ckpt_dict, checkpoint_path)
    print("Saved checkpoint")

def main(args):
    skill_dataset = SkillDataset(args.train_path)
    # TODO : add collator function to make it batch-wise padding
    train_skill_loader = torch.utils.data.DataLoader(skill_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SkillLSTM(2*args.input_size, args.hidden_size, args.input_size)     # the last argument is output_size

    model.train()

    # Get optimizer and scheduler can be used if given in paper
    optimizer = optim.SGD(model.parameters(), args.lr)

    # Get loss function
    loss_fn = nn.BCELoss()

	# training preparation
    steps_till_eval = args.eval_steps
    epoch = 0

    while epoch != args.num_epochs:
        epoch += 1
        epoch_loss = 0
        print("Entering epoch number")
        with torch.enable_grad(), tqdm(total=len(skill_dataset)) as progress_bar:
            for (student, student_orig) in train_skill_loader:
                # Setup forward
                optimizer.zero_grad()

                # Forward
                print("Starting forward pass")
                lengths = [int(q[0]) for q in student_orig]
                batch_out = model(student, lengths)
                loss = loss_fn(batch_out, student[:,:,args.input_size:])
                epoch_loss = epoch_loss + loss.item()

                # Backward
                print("Starting backward pass")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)        # To tackle exploding gradients
                optimizer.step()

                # Log info
                progress_bar.update(args.batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                        BCELoss=loss)

                steps_till_eval -= args.batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    model.eval()

                    # TODO : evaluate function

                    model.train()
                    save_checkpoint(model, args.save_path)
            print("Epoch loss is : {}".format(epoch_loss))

if __name__ == "__main__":
    args = get_train_args()
    main(args)
