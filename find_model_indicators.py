import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from models_batch import VanillaLSTM, VanillaReLURNN
import os



parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, help='type of model, LSTM, ReLU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


relu_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
relu_excel_path_2 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
relu_excel_path_3 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs.xlsx'


def read_sheets(num_runs, excel_name):
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run" + str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(excel_name, sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs



args = parser.parse_args()

model_type = args.model_type

if model_type=='ReLU':
    input_size = 2
    hidden_size = 1
    num_classes = 2
    output_activation='Sigmoid'


    best_runs_10runs_50epochs = [4, 5, 8]
    best_runs_20runs_30epochs = [4, 5, 8, 10, 12]
    best_runs_5runs_30epochs = [0, 3]
    checkpoints = []

    relu_prefix_1 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    relu_prefix_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
    relu_prefix_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'

    relu_checkpoint_prefix_1 = relu_prefix_1+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_CHECKPOINT'
    relu_checkpoint_prefix_2 = relu_prefix_2+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_CHECKPOINT'
    relu_checkpoint_prefix_3 = relu_prefix_3+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs_CHECKPOINT_'



    for i in range(len(best_runs_10runs_50epochs)):
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i])+'_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch25.pth')
    
    for i in range(len(best_runs_20runs_30epochs)):
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i])+'_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch25.pth')
    
    for i in range(len(best_runs_5runs_30epochs)):
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i])+'_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch25.pth')

    ab_ratios = []
    u_values = []


    for i in range(len(checkpoints)):
        checkpt = torch.load(checkpoints[i])
        model = VanillaReLURNN(input_size, hidden_size, num_layers=1, batch_size=1, output_size=num_classes,
                               output_activation=output_activation)

        model.load_state_dict(checkpt['model_state_dict'])
        model.to(device)

        print('relu input weight = ',model.rnn.weight_ih)
        print('relu input bias = ',model.bias_ih)
        print('relu hidden weight = ',model.rnn.weight_hh)
        print('relu_hidden_bias = ',model.rnn.bias_hh)



