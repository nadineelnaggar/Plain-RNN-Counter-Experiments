import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
import argparse
from argparse import ArgumentParser


seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
# parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification')
# parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
# parser.add_argument('--hidden_size', type=int, help='hidden size')
# parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
# parser.add_argument('--batch_size', type=int, help='batch size', default=1)
# parser.add_argument('--learning_rate', type=float, help='learning rate')
# parser.add_argument('--lr_scheduler_step',type=int, help='number of epochs before reducing', default=100)
# parser.add_argument('--lr_scheduler_gamma',type=float, help='multiplication factor for lr scheduler', default=1.0)
# parser.add_argument('--num_epochs', type=int, help='number of training epochs')
# parser.add_argument('--num_runs', type=int, help='number of training runs')
# # parser.add_argument('--best_run',type=int,help='run with the lowest loss and highest accuracy',default=-1)
# parser.add_argument('--checkpoint_step', type=int, help='checkpoint step', default=0)
# parser.add_argument('--shuffle_dataset',type=bool,default=False)
# parser.add_argument('--num_checkpoints', type=int,default=100, help='number of checkpoints we want to include if we dont need all of them (e.g., first 5 checkpoints only), stop after n checkpoints')
# parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or appended')
# parser.add_argument('--runtime',type=str, help='local or colab', default='colab')
#
#
# args = parser.parse_args()
#
# model_name = args.model_name
# task = args.task
# feedback = args.feedback
# hidden_size = args.hidden_size
# num_layers = args.num_layers
# learning_rate = args.learning_rate
# num_epochs = args.num_epochs
# num_runs = args.num_runs
# batch_size = args.batch_size
# # load_model = args.load_model
# lr_scheduler_gamma = args.lr_scheduler_gamma
# lr_scheduler_step = args.lr_scheduler_step
# num_checkpoints = args.num_checkpoints
# shuffle_dataset = args.shuffle_dataset
# checkpoint_step = args.checkpoint_step
# dataset_type = args.dataset_type
# runtime = args.runtime
#
# use_optimiser='Adam'
#
# num_bracket_pairs = 25
#
#
#
# if runtime=='local':
#     path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
#            +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
#            +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
#            +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
# elif runtime=='colab':
#     path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
#            +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
#            +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
#            +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
#
#
#
# prefix = path+'INFERENCE_'+dataset_type+'_'+str(checkpoint_step)+'checkpoint_step_upto'+str(num_checkpoints)+'checkpoints_'
#
#
# # excel_name_inference = path+ 'Dyck1_' + task + '_' + str(
# #         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
# #         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
# #         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + 'INFERENCE.xlsx'
#
# excel_name_inference=prefix+'EXCEL INFERENCE.xlsx'
#
# def read_sheets():
#     sheet_name='Sheet1'
#     df = pd.read_excel(excel_name_inference, sheet_name=sheet_name)
#     return df


# x = [1,2,3,4,5,6,7,8,9,10]
# y = [2,4,6,8,10,12,14,16,18,20]

rng = np.random.default_rng()
x = rng.random(10)
y = 1.6*x + rng.random(10)

res = stats.linregress(x, y)

print(f"R-squared: {res.rvalue**2:.6f}")
print('p value = ',res.pvalue)
print('res = ',res)
print('res slope (coefficient) = ',res.slope)
# print('coefficient = ',res.coefficient)


plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()