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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification')
parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
parser.add_argument('--hidden_size', type=int, help='hidden size')
parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--lr_scheduler_step',type=int, help='number of epochs before reducing', default=100)
parser.add_argument('--lr_scheduler_gamma',type=float, help='multiplication factor for lr scheduler', default=1.0)
parser.add_argument('--num_epochs', type=int, help='number of training epochs')
parser.add_argument('--num_runs', type=int, help='number of training runs')
# parser.add_argument('--best_run',type=int,help='run with the lowest loss and highest accuracy',default=-1)
parser.add_argument('--checkpoint_step', type=int, help='checkpoint step', default=0)
parser.add_argument('--shuffle_dataset',type=bool,default=False)
parser.add_argument('--num_checkpoints', type=int,default=100, help='number of checkpoints we want to include if we dont need all of them (e.g., first 5 checkpoints only), stop after n checkpoints')
parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or appended')
parser.add_argument('--runtime',type=str, help='local or colab', default='colab')


args = parser.parse_args()

model_name = args.model_name
task = args.task
feedback = args.feedback
hidden_size = args.hidden_size
num_layers = args.num_layers
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_runs = args.num_runs
batch_size = args.batch_size
# load_model = args.load_model
lr_scheduler_gamma = args.lr_scheduler_gamma
lr_scheduler_step = args.lr_scheduler_step
num_checkpoints = args.num_checkpoints
shuffle_dataset = args.shuffle_dataset
checkpoint_step = args.checkpoint_step
dataset_type = args.dataset_type
runtime = args.runtime

use_optimiser='Adam'

num_bracket_pairs = 25



if runtime=='local':
    path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
           +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
           +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
           +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"
elif runtime=='colab':
    path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
           +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
           +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
           +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"



prefix = path+'INFERENCE_'+dataset_type+'_'+str(checkpoint_step)+'checkpoint_step_upto'+str(num_checkpoints)+'checkpoints_'

# plot_train_loss = prefix+'LINEAR_REGRESSION_log_inverse_avg_train_loss_FPF.png'
# plot_val_loss = prefix+'LINEAR_REGRESSION_log_inverse_avg_val_loss_FPF.png'
# plot_long_loss = prefix+'LINEAR_REGRESSION_inverse_avg_long_loss_FPF.png'

plot_train_loss = prefix+'LINEAR_REGRESSION_log_inverse_avg_train_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY.png'
plot_val_loss = prefix+'LINEAR_REGRESSION_log_inverse_avg_val_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY.png'
plot_long_loss = prefix+'LINEAR_REGRESSION_inverse_avg_long_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY.png'


plot_train_loss_large_font = prefix+'LINEAR_REGRESSION_log_inverse_avg_train_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY_LARGE_FONT.png'
plot_val_loss_large_font = prefix+'LINEAR_REGRESSION_log_inverse_avg_val_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY_LARGE_FONT.png'
plot_long_loss_large_font = prefix+'LINEAR_REGRESSION_inverse_avg_long_loss_FPF_EVERY_5_EPOCHS_GOOD_MODELS_ONLY_LARGE_FONT.png'


# excel_name_inference = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + 'INFERENCE.xlsx'

# excel_name_inference=prefix+'EXCEL INFERENCE.xlsx'
excel_name_inference=prefix+'EXCEL INFERENCE CHECKPOINTS ONLY GOOD MODELS.xlsx'

def read_sheets():
    sheet_name='Sheet1'
    df = pd.read_excel(excel_name_inference, sheet_name=sheet_name)
    return df


# x = [1,2,3,4,5,6,7,8,9,10]
# y = [2,4,6,8,10,12,14,16,18,20]

# rng = np.random.default_rng()
# x = rng.random(10)
# y = 1.6*x + rng.random(10)
#
# res = stats.linregress(x, y)
#
# print(f"R-squared: {res.rvalue**2:.6f}")
# print('p value = ',res.pvalue)
# print('res = ',res)
# print('res slope (coefficient) = ',res.slope)
# # print('coefficient = ',res.coefficient)
#
#
# plt.plot(x, y, 'o', label='original data')
# plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
# plt.legend()
# plt.show()


def plot_linear_regression():
    # different models, all sequences (one histogram per model)

    df = read_sheets()
    # max_depth = df['max depth for incorrect sequences (2000 tokens)'][0]
    # print(type(max_depth))
    # print(max_depth[:20])
    # txt = df['first point of failure for each incorrect sequence'][0]
    # all_fpf=[int(s) for s in txt.split(', ') if s.isdigit()]

    avg_fpf = df['average first point of failure (2000 tokens)']



    fpfs = []

    log_inverse_val_loss = df['log of inverse avg validation losses']
    log_inverse_train_loss = df['log of inverse avg train losses']
    log_inverse_long_loss = df['log of inverse avg long validation losses']
    for i in range(len(df)):
        max_depth = df['max depth for incorrect sequences (2000 tokens)'][i]
        print(type(max_depth))
        print(max_depth[:20])
        txt = df['first point of failure for each incorrect sequence'][i]
        txt = txt.replace('[', '')
        txt = txt.replace(']', '')
        all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
        # plt.subplots()


        fpf = avg_fpf[i]
        fpfs.append(fpf)


    res_train = stats.linregress(log_inverse_train_loss,fpfs)
    print('r value for train = ',res_train.rvalue)
    print('r squared for train = ', res_train.rvalue*res_train.rvalue)
    print('p value for train = ', res_train.pvalue)
    print('coefficient value for train = ', res_train.slope)
    print(res_train)
    plt.subplots()
    plt.rcParams['font.size'] = '12'
    # plt.plot(log_inverse_train_loss, fpfs, 'o', label='Negative log train loss compared to Avg FPF')
    plt.plot(log_inverse_train_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_train_loss, res_train.intercept + res_train.slope*log_inverse_train_loss, 'r', label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log train loss', fontsize=14)
    plt.ylabel('Average FPF', fontsize=14)
    # plt.title('Correlation and linear regression for the', fontsize=16)
    plt.legend(prop={'size': 12})
    plt.savefig(plot_train_loss)
    plt.show()
    plt.close()

    plt.subplots()
    plt.rcParams['font.size'] = '16'
    # plt.plot(log_inverse_train_loss, fpfs, 'o', label='Negative log train loss compared to Avg FPF')
    plt.plot(log_inverse_train_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_train_loss, res_train.intercept + res_train.slope * log_inverse_train_loss, 'r',
             label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log train loss', fontsize=16)
    plt.ylabel('Average FPF', fontsize=16)
    # plt.title('Correlation and linear regression for the', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(plot_train_loss_large_font, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    res_val = stats.linregress(log_inverse_val_loss, fpfs)
    print('r value for val = ', res_val.rvalue)
    print('r squared for val = ', res_val.rvalue*res_val.rvalue)
    print('p value for val = ', res_val.pvalue)
    print('coefficient value for val = ', res_val.slope)
    print(res_val)
    plt.subplots()
    plt.rcParams['font.size'] = '12'
    # plt.plot(log_inverse_val_loss, fpfs, 'o', label='Negative log val loss compared to Avg FPF')
    plt.plot(log_inverse_val_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_val_loss, res_val.intercept + res_val.slope * log_inverse_val_loss, 'r',
             label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log validation loss', fontsize=14)
    plt.ylabel('Average FPF', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.savefig(plot_val_loss)
    plt.show()
    plt.close()

    plt.subplots()
    plt.rcParams['font.size'] = '16'
    # plt.plot(log_inverse_val_loss, fpfs, 'o', label='Negative log val loss compared to Avg FPF')
    plt.plot(log_inverse_val_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_val_loss, res_val.intercept + res_val.slope * log_inverse_val_loss, 'r',
             label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log validation loss', fontsize=16)
    plt.ylabel('Average FPF', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(plot_val_loss_large_font,bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    res_long = stats.linregress(log_inverse_long_loss, fpfs)
    print('r value for long = ', res_long.rvalue)
    print('r squared for long = ', res_long.rvalue * res_long.rvalue)
    print('p value for long = ', res_long.pvalue)
    print('coefficient value for long = ', res_long.slope)
    print(res_long)
    plt.subplots()
    plt.rcParams['font.size'] = '12'
    # plt.plot(log_inverse_long_loss, fpfs, 'o', label='negative log validation loss compared to Avg FPF')
    plt.plot(log_inverse_long_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_long_loss, res_long.intercept + res_long.slope * log_inverse_long_loss, 'r',
             label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log long validation loss', fontsize=14)
    plt.ylabel('Average FPF', fontsize=14)
    plt.legend(prop={'size': 12})
    plt.savefig(plot_long_loss)
    plt.show()
    plt.close()

    plt.subplots()
    plt.rcParams['font.size'] = '16'
    # plt.plot(log_inverse_long_loss, fpfs, 'o', label='negative log validation loss compared to Avg FPF')
    plt.plot(log_inverse_long_loss, fpfs, 'o', label='Models at different stages of training')
    plt.plot(log_inverse_long_loss, res_long.intercept + res_long.slope * log_inverse_long_loss, 'r',
             label='fitted line')
    # plt.legend()
    plt.xlabel('Negative log long validation loss', fontsize=16)
    plt.ylabel('Average FPF', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(plot_long_loss_large_font, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
plot_linear_regression()