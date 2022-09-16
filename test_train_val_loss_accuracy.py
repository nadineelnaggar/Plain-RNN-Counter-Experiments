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



# seed = 10
# torch.manual_seed(seed)
# np.random.seed(seed)
#
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
# parser.add_argument('--checkpoint_step', type=int, help='checkpoint step', default=0)
# parser.add_argument('--shuffle_dataset',type=bool,default=False)
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
# lr_scheduler_step = args.lr_scheduler_step
# lr_scheduler_gamma = args.lr_scheduler_gamma
#
# checkpoint_step = int(num_epochs/4)
# if args.checkpoint_step!=0:
#     checkpoint_step = args.checkpoint_step
#
# shuffle_dataset = args.shuffle_dataset
#


# bins = [*range(0, 49,48), 49, *range(50, 99, 48), 99, *range(100, 149, 48), 149, *range(150, 199, 48)]

bins = [*range(0, 2001, 50)]

new_bins = []
for i in range(len(bins)):
    if bins[i]==0:
        new_bins.append(bins[i])
    elif bins[i]>0:
        new_bins.append(bins[i]-2)
        new_bins.append(bins[i]-1)
        new_bins.append(bins[i])


print(bins)

print(new_bins)
