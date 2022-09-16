import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from Dyck1_Datasets import NextTokenPredictionDataset2000tokens_nested, NextTokenPredictionDataset2000tokens_zigzag, NextTokenPredictionDataset2000tokens
from models_batch import VanillaLSTM, VanillaRNN, VanillaGRU, VanillaReLURNN

# RERUN THE CODE TO GENERATE THE EXCEL SHEETS NOW THAT THE SHUFFLE DATA IS FALSE IN THE DATALOADER,
# THEN GENERATE HISTOGRAMS


# to read the excel generated from the inference code
# read the excel from the inference
# make the histograms to reflect the distribution of the fpf (single model different sequences + max depth plot)
# make the histograms to reflect the distribution of the fpf (single sequence different models + timestep plot)



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



prefix = path+model_name+'_INFERENCE_'+dataset_type+'_'+str(checkpoint_step)+'checkpoint_step_upto'+str(num_checkpoints)+'checkpoints_'


# excel_name_inference = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + 'INFERENCE.xlsx'

excel_name_inference=prefix+'EXCEL INFERENCE.xlsx'

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
    # df = read_sheets()

    if model_name == 'VanillaLSTM':
        # print(df['avg training losses'][20].dtype)
        # print(df['avg training losses'][20])
        df.drop(df[df['avg training losses'] > 10e-10].index, inplace=True)
        print(len(df))
        # print(df.loc(df['avg training losses'] <= 10e-10))
        # df = df.loc(df['avg training losses'] <= 10e-10)
        # print(df.head())
    elif model_name == 'VanillaReLURNN':
        # df = df.loc(df['avg training losses'] <= 0.015)
        df.drop(df[df['avg training losses'] > 0.015].index, inplace=True)
        print(len(df))
    elif model_name=='VanillaGRU':
        # df = df.loc(df['avg training losses'] <= 0.015)
        df.drop(df[df['avg training losses'] > 0.001].index, inplace=True)
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
        try:
            max_depth = df['max depth for incorrect sequences (2000 tokens)'][i]
            print(type(max_depth))
            print(max_depth[:20])
            txt = df['first point of failure for each incorrect sequence'][i]
            txt = txt.replace('[', '')
            txt = txt.replace(']', '')
            all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
            # plt.subplots()
            bins = []

            fpf = avg_fpf[i]
            fpfs.append(fpf)
            max_depths = max_depth[i]
        # except:
        #     IndexError
        # for k in range(len(fpfs)):
        #     bins.append(k)
        # for j in range(min(max_depth[j]), max(max_depth[j])+1):
        #     if max_depth[j] not in bins:
        #         bins.append(max_depth[j])
        # plt.hist(x=max_depth[i], range=[0,1000])
        # plt.hist(fpfs, bins=hist_bins,range=[0,max(max_depth[i])])
            plt.subplots()
            # plt.hist(all_fpf, bins=range(0,2001,50))
            plt.hist(all_fpf, bins=range(0, 2001, 50))
            plt.xlabel('First point of failure for each incorrect sequence')
            plt.ylabel('Number of incorrect sequences')
            # plt.savefig(path+'histogram one model multiple sequences '+str(i)+'_TRAINED_MODELS_ONLY.png')
            plt.savefig(prefix + 'histogram one model multiple sequences ' + str(i) + '_TRAINED_MODELS_ONLY.png')
            plt.show()
            plt.close()
        except:
            IndexError

# create_histogram()

def get_timestep_depths(x):
    max_depth=0
    current_depth=0
    timestep_depths = []
    for i in range(len(x)):

        if x[i] == '(':
            current_depth += 1
            timestep_depths.append(current_depth)
            if current_depth > max_depth:
                max_depth = current_depth
        elif x[i] == ')':
            current_depth -= 1
            timestep_depths.append(current_depth)
    return max_depth, timestep_depths

def create_histogram_one_sequence_multiple_models():
    # choose a sequence (maybe 5, and create one histogram per sequence)
    # extract the max_depth of the sequence
    # extract the fpf of this particular sequence on the different models
    # plot histogram of the fpf
    # plot the timestep depths if possible


    if dataset_type=='nested':
        dataset = NextTokenPredictionDataset2000tokens_nested()
    elif dataset_type=='zigzag':
        dataset = NextTokenPredictionDataset2000tokens_zigzag()
    elif dataset_type=='concatenated':
        dataset = NextTokenPredictionDataset2000tokens()



    df = read_sheets()
    
    if model_name=='VanillaLSTM':
        # df = df.loc(df['avg training losses'] <= 10E-10)
        df.drop(df[df['avg training losses'] > 10e-10].index, inplace=True)
        print(df)
        print(len(df))
    elif model_name=='VanillaReLURNN':
        # df = df.loc(df['avg training losses'] <= 0.015)
        df.drop(df[df['avg training losses'] > 0.015].index, inplace=True)
    elif model_name=='VanillaGRU':
        # df = df.loc(df['avg training losses'] <= 0.015)
        df.drop(df[df['avg training losses'] > 0.001].index, inplace=True)
    # print(len(df))

    
    
    num_models = len(df) #number of rows in the dataframe = number of models
    print('num_models = ',num_models)

    #read relevant columns from the dataframe

    avg_fpf = df['average first point of failure (2000 tokens)']
    # all_fpfs = df['first point of failure for each incorrect sequence']

    bins = [*range(0, 2001, 50)]

    new_bins = []
    for i in range(len(bins)):
        if bins[i] == 0:
            new_bins.append(bins[i])
        elif bins[i] > 0:
            new_bins.append(bins[i] - 2)
            new_bins.append(bins[i] - 1)
            new_bins.append(bins[i])


    for i in range(10):
        fpfs = []
        seq, label, length = dataset[i]
        timestep_depths = []
        max_depth, timestep_depth = get_timestep_depths(dataset.x[i])
        timestep_depths.append(timestep_depth)
        for j in range(num_models):
            try:
                txt = df['first point of failure for each incorrect sequence'][j]
                txt = txt.replace('[','')
                txt = txt.replace(']','')
                all_fpf = [int(s) for s in txt.split(', ') if s.isdigit()]
                print('all_fpf = ',all_fpf)
                print('len(all_fpf) = ',len(all_fpf))
            # try:
                fpfs.append(all_fpf[i])
            except:
                IndexError
            # for k in range(len(all_fpf)):
            #     fpfs.append(all_fpf[k])
        plt.subplots()
        # plt.plot(timestep_depth, color='red')
        # bins = [0,48, 49, 98, 99, 198, 199, 200, 248, 249, ]
        # bins = []

        plt.hist(fpfs, bins=new_bins)
        # plt.hist(fpfs, bins=range(0, 2001, 50))
        plt.plot([i for i in range(len(timestep_depth))], timestep_depth, color='red')

        plt.xlabel('First point of failure for each incorrect sequence')
        plt.ylabel('Number of incorrect sequences')
        plt.savefig(prefix + 'histogram one sequence multiple models ' + str(i) + '_TRAINED_MODELS_ONLY.png')
        plt.show()
        plt.close()
        plt.subplots()
        plt.plot([i for i in range(len(timestep_depth))],timestep_depth, color='red')
        plt.xlabel('Timestep')
        plt.ylabel('Stack Depths')
        # plt.savefig(path + 'timestep depth one sequence multiple models ' + str(i) + '_TRAINED_MODELS_ONLY.png')
        plt.savefig(prefix + 'timestep depth one sequence multiple models ' + str(i) + '_TRAINED_MODELS_ONLY.png')
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

modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL_'


def select_model(model_name, input_size, hidden_size, num_layers,batch_size, num_classes, output_activation):
    if model_name=='VanillaLSTM':
        selected_model = VanillaLSTM(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaRNN':
        selected_model = VanillaRNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaGRU':
        selected_model = VanillaGRU(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name == 'VanillaReLURNN':
        selected_model = VanillaReLURNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)

    return selected_model.to(device)

def create_depth_charts():
    input_size = 2
    num_classes = 2
    output_activation='Sigmoid'
    for run in range(num_runs):
        mdl = modelname + 'run' + str(run) + '.pth'
        model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes,
                             output_activation)
        model.load_state_dict(torch.load(mdl))
        model.to(device)