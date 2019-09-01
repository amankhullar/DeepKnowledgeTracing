import os
import sys

import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import sqrt
from datasets import SkillDataset
from models import SkillLSTM

from args import get_train_args
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

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
    loss_list = []

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
                batch_out = model(student, lengths)                     # (batch_size, num_ques, skill_size)
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
                    print("Evaluating the trained model")

                    res_rmse, res_auc, res_r2 = evaluate(model)
                    print("RMSE is : {}".format(res_rmse))
                    print("AUC is : {}".format(res_auc))
                    print("R2 is : {}".format(res_r2))

                    model.train()
                    save_checkpoint(model, args.save_path)
                    print("Evaluation complete")
                    
            loss_list.append(epoch_loss)
            print("Epoch loss is : {}".format(epoch_loss))

    # output plot for loss visualization
    plt.figure(loss_list)
    plt.savefig(args.save_loss_plot)

def evaluate(model):
    model.eval()
    eval_dataset = SkillDataset(args.eval_path)

    dev_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rmse = auc = r2 = 0
    with torch.no_grad(), tqdm(total=len(eval_dataset)) as progress_bar:
        for (student, student_orig) in dev_loader:
            # Forward
            lengths = [int(q[0]) for q in student_orig]
            preds = model(student, lengths)         # (batch_size, num_ques, skill_size)
            preds_idxs = torch.argmax(preds, dim=-1)
            for batch_idx in range(args.batch_size):
                y_true = []
                y_pred = []
                for ques_idx in range(lengths[batch_idx]):
                    tag = int((student[batch_idx][ques_idx] == 1).nonzero()[0])
                    # print("Tag is : {}".format(tag))
                    # print("Student tag is : {}".format(student[batch_idx][ques_idx]))
                    y_true.append(int(student[batch_idx][ques_idx][args.input_size+tag-1]))       # true tag taken from the concatenated output
                    y_pred.append(int(tag==int(preds_idxs[batch_idx][ques_idx])))                 # predicited tag calculated from the given question and answer
                
                # RMSE 
                batch_rmse = sqrt(mean_squared_error(y_true, y_pred))
                # print("True val: {}".format(y_true))
                # print("Pred val: {}".format(y_pred))
                rmse += batch_rmse

                # AUC
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
                auc += metrics.auc(fpr, tpr)

                # R^2 metric
                r2 += r2_score(y_true, y_pred)

                # Generate heatmap of output predictions
                ax = sns.heatmap(preds[batch_idx,:lengths[batch_idx],:].numpy())         # Ignoring the padded predictions
                fig = ax.get_figure()
                fig.savefig(args.save_fig)

            rmse /= args.batch_size
            auc /= args.batch_size
            r2 /= args.batch_size

            progress_bar.update(args.batch_size)

    return rmse, auc, r2


if __name__ == "__main__":
    args = get_train_args()
    main(args)
