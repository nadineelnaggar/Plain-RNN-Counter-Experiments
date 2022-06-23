import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from Dyck1_Datasets import NextTokenPredictionDataset2000tokens_nested, NextTokenPredictionDataset2000tokens_zigzag, NextTokenPredictionDataset2000tokens
from models_batch import VanillaLSTM, VanillaRNN, VanillaGRU, VanillaReLURNN


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


excel_name = path+''+model_name+'_METRICS.xlsx'

def read_sheets():
    sheet_name='Sheet1'
    df = pd.read_excel(excel_name, sheet_name=sheet_name)
    # df = pd.read_excel(path_name,sheet_name=sheet_names)
    # dfs = []
    # for i in range(num_runs):
    #     dfs.append(df.get(sheet_names[i]))
    # return dfs
    return df

frames=read_sheets()
# # print(frames.head())
# print(frames['first point of failure for each incorrect sequence'].head())
# print(len(frames['first point of failure for each incorrect sequence']))

def create_histograms_lstm_metrics():
    # different models, all sequences (one histogram per model)

    df = read_sheets()
    # max_depth = df['max depth for incorrect sequences (2000 tokens)'][0]
    # print(type(max_depth))
    # print(max_depth[:20])
    # txt = df['first point of failure for each incorrect sequence'][0]
    # all_fpf=[int(s) for s in txt.split(', ') if s.isdigit()]

    # avg_fpf = df['average first point of failure (2000 tokens)']
    
    metrics_ft_best_case = df['metrics_ft_best_case']
    metrics_ft_worst_case = df['metrics_ft_worst_case']
    metrics_it_best_case = df['metrics_it_1_best_case']
    metrics_it_worst_case = df['metrics_it_1_worst_case']
    metrics_ctilde_open_best_case = df['metrics_ctilde_open_best_case']
    metrics_ctilde_open_worst_case = df['metrics_ctilde_open_worst_case']
    metrics_ctilde_close_best_case = df['metrics_ctilde_close_best_case']
    metrics_ctilde_close_worst_case = df['metrics_ctilde_close_worst_case']
    metrics_ot = df['metrics_ot']
    sigmoid_metrics_ft_best_case = df['sigmoid_metrics_ft_best_case']
    sigmoid_metrics_ft_worst_case = df['sigmoid_metrics_ft_worst_case']
    sigmoid_metrics_it_best_case = df['sigmoid_metrics_it_best_case']
    sigmoid_metrics_it_worst_case = df['sigmoid_metrics_it_worst_case']
    sigmoid_metrics_ot = df['sigmoid_metrics_ot']
    tanh_metrics_ctilde_open_best_case = df['tanh_metrics_ctilde_open_best_case']
    tanh_metrics_ctilde_open_worst_case = df['tanh_metrics_ctilde_open_worst_case']
    tanh_metrics_ctilde_close_best_case = df['tanh_metrics_ctilde_close_best_case']
    tanh_metrics_ctilde_close_worst_case = df['tanh_metrics_ctilde_close_worst_case']
    

    # hist_bins = []
    # for i in range(1000):
    #     hist_bins.append(i)

    plt.subplots()
    plt.hist(metrics_ft_best_case, alpha=0.5, label='best case')
    plt.hist(metrics_ft_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path+"_"+model_name+'_'+'histogram_metrics_ft.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(metrics_it_best_case, alpha=0.5, label='best case')
    plt.hist(metrics_it_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_metrics_it.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(metrics_ctilde_open_best_case, alpha=0.5, label='best case')
    plt.hist(metrics_ctilde_open_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_metrics_ctilde_open.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(metrics_ctilde_close_best_case, alpha=0.5, label='best case')
    plt.hist(metrics_ctilde_close_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_metrics_ctilde_close.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(metrics_ot, alpha=0.5)
    # plt.hist(metrics_ctilde_close_worst_case, alpha=0.5)
    # plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_metrics_ot.png')
    plt.show()
    plt.close()
#################################
    plt.subplots()
    plt.hist(sigmoid_metrics_ft_best_case, alpha=0.5, label='best case')
    plt.hist(sigmoid_metrics_ft_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_sigmoid_metrics_ft.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(sigmoid_metrics_it_best_case, alpha=0.5, label='best case')
    plt.hist(sigmoid_metrics_it_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_sigmoid_metrics_it.png')
    plt.show()
    plt.close()
    
    plt.subplots()
    plt.hist(sigmoid_metrics_ot, alpha=0.5)
    # plt.hist(sigmoid_metrics_ctilde_close_worst_case, alpha=0.5)
    # plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_sigmoid_metrics_ot.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(tanh_metrics_ctilde_open_best_case, alpha=0.5, label='best case')
    plt.hist(tanh_metrics_ctilde_open_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_tanh_metrics_ctilde_open.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.hist(tanh_metrics_ctilde_close_best_case, alpha=0.5, label='best case')
    plt.hist(tanh_metrics_ctilde_close_worst_case, alpha=0.5, label='worst case')
    plt.legend(loc='upper right')
    plt.savefig(path + "_" + model_name + '_' + 'histogram_tanh_metrics_ctilde_close.png')
    plt.show()
    plt.close()



def create_histograms_relu_metrics():
    pass


if model_name=='VanillaLSTM':
    create_histograms_lstm_metrics()
elif model_name=='VanillaReLURNN':
    create_histograms_relu_metrics()
