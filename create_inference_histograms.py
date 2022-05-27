import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# RERUN THE CODE TO GENERATE THE EXCEL SHEETS NOW THAT THE SHUFFLE DATA IS FALSE IN THE DATALOADER,
# THEN GENERATE HISTOGRAMS


# to read the excel generated from the inference code
# read the excel from the inference
# make the histograms to reflect the distribution of the fpf (single model different sequences + max depth plot)
# make the histograms to reflect the distribution of the fpf (single sequence different models + timestep plot)



seed = 10
torch.manual_seed(seed)
np.random.seed(seed)


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

use_optimiser='Adam'

num_bracket_pairs = 25




# path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
#        +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
#        +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
#        +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"

path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"

excel_name_inference = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + 'INFERENCE.xlsx'


def read_sheets():
    sheet_name='Sheet1'
    df = pd.read_excel(excel_name_inference, sheet_name=sheet_name)
    # df = pd.read_excel(path_name,sheet_name=sheet_names)
    # dfs = []
    # for i in range(num_runs):
    #     dfs.append(df.get(sheet_names[i]))
    # return dfs
    return df

frames=read_sheets()
# print(frames.head())
print(frames['first point of failure for each incorrect sequence'].head())
print(len(frames['first point of failure for each incorrect sequence']))

def create_histogram():
    # different models, all sequences (one histogram per model)

    df = read_sheets()
    # max_depth = df['max depth for incorrect sequences (2000 tokens)'][0]
    # print(type(max_depth))
    # print(max_depth[:20])
    # txt = df['first point of failure for each incorrect sequence'][0]
    # all_fpf=[int(s) for s in txt.split(', ') if s.isdigit()]

    avg_fpf = df['average first point of failure (2000 tokens)']

    hist_bins = []
    for i in range(1000):
        hist_bins.append(i)

    fpfs = []
    for i in range(5):
        max_depth = df['max depth for incorrect sequences (2000 tokens)'][i]
        print(type(max_depth))
        print(max_depth[:20])
        txt = df['first point of failure for each incorrect sequence'][i]
        all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
        plt.subplots()
        bins = []

        fpf = avg_fpf[i]
        fpfs.append(fpf)
        max_depths = max_depth[i]
        # for k in range(len(fpfs)):
        #     bins.append(k)
        # for j in range(min(max_depth[j]), max(max_depth[j])+1):
        #     if max_depth[j] not in bins:
        #         bins.append(max_depth[j])
        # plt.hist(x=max_depth[i], range=[0,1000])
        # plt.hist(fpfs, bins=hist_bins,range=[0,max(max_depth[i])])
        plt.hist(all_fpf, bins=range(0,2001,50))
        plt.xlabel('First point of failure for each incorrect sequence')
        plt.ylabel('Number of incorrect sequences')
        plt.savefig(path+'histogram one model multiple sequences '+str(i)+'.png')
        plt.show()
        plt.close()

# create_histogram()

def create_histogram_one_sequence_multiple_models():
    # choose a sequence (maybe 5, and create one histogram per sequence)
    # extract the max_depth of the sequence
    # extract the fpf of this particular sequence on the different models
    # plot histogram of the fpf
    # plot the timestep depths if possible

    df = read_sheets()
    num_models = len(df) #number of rows in the dataframe = number of models

    #read relevant columns from the dataframe

    avg_fpf = df['average first point of failure (2000 tokens)']
    # all_fpfs = df['first point of failure for each incorrect sequence']





    for i in range(5):
        fpfs = []
        for j in range(num_models):
            txt = df['first point of failure for each incorrect sequence'][j]
            all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
            fpfs.append(all_fpf[i])
        plt.subplots()
        plt.hist(fpfs, bins=range(0, 2001, 50))
        plt.xlabel('First point of failure for each incorrect sequence')
        plt.ylabel('Number of incorrect sequences')
        plt.savefig(path + 'histogram one sequence multiple models ' + str(i) + '.png')
        plt.show()
        plt.close()



    # for i in range(5):
    #     max_depth = df['max depth for incorrect sequences (2000 tokens)'][i]
    #     # print(type(max_depth))
    #     # print(max_depth[:20])
    #     txt = df['first point of failure for each incorrect sequence'][i]
    #     all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
    #     fpfs.append(all_fpf[0])
    #     plt.subplots()
    #     bins = []
    #
    #     fpf = avg_fpf[i]
    #     fpfs.append(fpf)
    #     max_depths = max_depth[i]
    #
    #     plt.hist(all_fpf, bins=range(0, 1001, 50))
    #     plt.xlabel('First point of failure for each incorrect sequence')
    #     plt.ylabel('Number of incorrect sequences')
    #     plt.savefig(path+'histogram one sequence multiple models ' + str(i) + '.png')
    #     plt.show()
    #     plt.close()

create_histogram()
create_histogram_one_sequence_multiple_models()