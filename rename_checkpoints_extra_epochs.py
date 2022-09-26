import os

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
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionTrainDataset, NextTokenPredictionValidationDataset, NextTokenPredictionLongTestDataset_SAMPLE, NextTokenPredictionShortTestDataset_SAMPLE, NextTokenPredictionTrainDataset_SAMPLE, NextTokenPredictionValidationDataset_SAMPLE
from torch.optim.lr_scheduler import StepLR
import math
import time

"""
the LSTM checkpoints when training extra epochs were misnamed to add num_epochs to the epoch in additional epochs
"""


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
parser.add_argument('--checkpoint_step', type=int, help='checkpoint step', default=0)
parser.add_argument('--shuffle_dataset',type=bool,default=False)
parser.add_argument('--extra_epochs',type=int, default=10)



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
lr_scheduler_step = args.lr_scheduler_step
lr_scheduler_gamma = args.lr_scheduler_gamma

checkpoint_step = int(num_epochs/4)
if args.checkpoint_step!=0:
    checkpoint_step = args.checkpoint_step

shuffle_dataset = args.shuffle_dataset
extra_epochs = args.extra_epochs

use_optimiser='Adam'

num_bracket_pairs = 25

length_bracket_pairs = 50

input_size=2
num_classes = 2
output_activation='Sigmoid'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"

print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('batch_size = ',batch_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)
# print('load_model = ',load_model)
print('extra epochs = ',extra_epochs)


checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'


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
    # return selected_model


for run in range(num_runs):
    for epoch in range(num_epochs,(num_epochs+extra_epochs),1):
        if epoch%checkpoint_step==0:
            # checkpoint_path = checkpoint + 'run' + str(run) + "_epoch" + str(epoch+num_epochs) + ".pth"
            checkpoint_path_old = checkpoint + 'run' + str(run) + "_epoch" + str(epoch + num_epochs) + ".pth"
            checkpoint_path = checkpoint + 'run' + str(run) + "_epoch" + str(epoch) + ".pth"
            if os.path.exists(checkpoint_path_old)==True and os.path.exists(checkpoint_path)==False:
                os.rename(checkpoint_path_old, checkpoint_path)
            checkpoint_model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes,
                                            output_activation)

            checkpt = torch.load(checkpoint_path)
            checkpoint_model.load_state_dict(checkpt['model_state_dict'])
            loss = checkpt['loss']
            checkpoint_model.to(device)
            checkpt = torch.load(checkpoint_path)


            criterion = nn.MSELoss()
            # learning_rate = args.learning_rate
            optimiser = optim.Adam(checkpoint_model.parameters(), lr=learning_rate)
            # optimiser.zero_grad()
            optimiser.load_state_dict(checkpt['optimiser_state_dict'])

            torch.save({'run': run,
                        'epoch': epoch,
                        'model_state_dict': checkpoint_model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'loss': loss}, checkpoint_path)

            # checkpoint_path_old = checkpoint+'run'+str(run)+"_epoch"+str(epoch+num_epochs)+".pth"
            # checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"
            # os.rename(checkpoint_path_old,checkpoint_path)