import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models_batch import VanillaLSTM, VanillaRNN, VanillaGRU, VanillaReLURNN
from Dyck_Generator_Suzgun_Batch import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, \
    NextTokenPredictionTrainDataset, NextTokenPredictionDataset102to500tokens,NextTokenPredictionDataset502to1000tokens, \
    NextTokenPredictionDataset990to1000tokens, NextTokenPredictionDataset2000tokens, \
    NextTokenPredictionDataset2000tokens_nested, NextTokenPredictionDataset2000tokens_zigzag

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
dataset_type = args.dataset_type
shuffle_dataset = args.shuffle_dataset
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


def inspect_model_parameters():
    input_size = 2
    num_classes = 2
    output_activation='Sigmoid'
    for run in range(num_runs):
        mdl = modelname + 'run' + str(run) + '.pth'
        model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes,
                             output_activation)
        model.load_state_dict(torch.load(mdl))
        model.to(device)

        print('*************************************')
        print('RUN ',run)
        print('*************************************')
        if model_name=='VanillaReLURNN':
            print(model_name)
            # print(model.rnn.Variables.weights)
            # print(model.rnn.Variables.biases)
            weights = []
            biases = []

            print(model.rnn.named_parameters())
            for param in model.rnn.named_parameters():
                if 'weight' in param[0]:
                    weights.append(param[1])
                elif 'bias' in param[0]:
                    biases.append(param[1])

        # elif model_name=='VanillaLSTM':
        #     print(model_name)
        #     weights = []
        #     biases = []
        #     print(model.lstm.named_parameters())
        #     for param in model.lstm.named_parameters():
        #         if 'weight' in param[0]:
        #             weights.append(param[1])
        #         elif 'bias' in param[0]:
        #             biases.append(param[1])

            print(weights)
            print(biases)
            # print('RNN weight_ih = ',model.rnn.weight_ih_l)
            # print('RNN weight_hh = ',model.rnn.weight_hh_l)
            # print('RNN bias_ih = ',model.rnn.bias_ih_l)
            # print('RNN bias_hh = ',model.rnn.bias_hh_l)

        elif model_name=='VanillaLSTM':
            print(model_name)
            # weights_ih=[]
            weights_ii = []
            weights_if = []
            weights_ig = []
            weights_io = []

            biases_ii = []
            biases_if = []
            biases_ig = []
            biases_io = []

            weights_hi = []
            weights_hf = []
            weights_hg = []
            weights_ho = []

            biases_hi = []
            biases_hf = []
            biases_hg = []
            biases_ho = []

            # weights_hh = []
            # biases_ih = []
            # biases_hh = []
            print(list(model.lstm.named_parameters()))
            # for param in model.lstm.names_parameters():
            for param in model.lstm.named_parameters():
                if 'weight_hh' in param[0]:
                    weights_hh = param[1]
                    weights_hi.append(weights_hh[0].item())
                    weights_hf.append(weights_hh[1].item())
                    weights_hg.append(weights_hh[2].item())
                    weights_ho.append(weights_hh[3].item())



                elif 'weight_ih' in param[0]:
                    weights_ih = param[1]
                    weights_ii.append(weights_ih[0].item())
                    weights_if.append(weights_ih[1].item())
                    weights_ig.append(weights_ih[2].item())
                    weights_io.append(weights_ih[3].item())

            print('weight_hi = ',weights_hi)
            print('weight_ii = ',weights_ii)





inspect_model_parameters()