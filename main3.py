import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models import VanillaRNN, VanillaLSTM, VanillaGRU
from Dyck_Generator_Suzgun import DyckLanguage
import random
# import vanilla_models
# from vanilla_models import VanillaLSTM

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = ',device)

NUM_PAR = 1
MIN_SIZE = 2
MAX_SIZE = 50
P_VAL = 0.5
Q_VAL = 0.25


Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)




parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification')
parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
parser.add_argument('--hidden_size', type=int, help='hidden size')
parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('--num_epochs', type=int, help='number of training epochs')
parser.add_argument('--num_runs', type=int, help='number of training runs')


args = parser.parse_args()

model_name = args.model_name
task = args.task
feedback = args.feedback
hidden_size = args.hidden_size
num_layers = args.num_layers
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_runs = args.num_runs

use_optimiser='Adam'


print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)


model = VanillaLSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=2).to(device)



X = ['()', '()()', '(())', '()()()', '(())()', '()(())', '((()))']
y = ['10', '1010', '1110', '101010', '111010', '101110', '111110']






