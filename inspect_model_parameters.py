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
import math


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
checkpoint_step = args.checkpoint_step


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

read_excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '.xlsx'

checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'


excel_name = path+''+model_name+'_METRICS.xlsx'

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


def read_sheets():
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run"+str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(read_excel_name,sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs


def inspect_lstm(model):
    print(model)

    for param in model.lstm.named_parameters():
        if 'weight_hh' in param[0]:
            weights_hh = param[1]
            weights_hi = weights_hh[0]
            weights_hf=weights_hh[1]
            weights_hg = weights_hh[2]
            weights_ho = weights_hh[3]



        elif 'weight_ih' in param[0]:
            weights_ih = param[1]
            weights_ii = weights_ih[0]
            weights_if = weights_ih[1]
            weights_ig = weights_ih[2]
            weights_io = weights_ih[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]

        elif 'bias_hh' in param[0]:
            biases_hh = param[1]
            biases_hi = biases_hh[0]
            biases_hf = biases_hh[1]
            biases_hg = biases_hh[2]
            biases_ho = biases_hh[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]

    print('\n')
    print('TO CALCULATE IT')
    print('weight_ii = ', weights_ii)
    print('bias_ii = ', biases_ii)
    print('weight_hi = ', weights_hi)
    print('bias_hi = ', biases_hi)

    metric_it_1 = min(weights_ii[0].item(), weights_ii[1].item()) + biases_ii.item() + biases_hi.item() - torch.abs(weights_hi[run]).item()

    # metrics_it_1.append(metric_it_1)
    print('metric_it_1 = ', metric_it_1)
    print('sigmoid(metric_it_1_min) = ', torch.sigmoid(torch.tensor(metric_it_1, dtype=torch.float32)))
    metric_it_1_best_case = max(weights_ii[0].item(), weights_ii[1].item()) + biases_ii.item() + \
                            biases_hi.item() + torch.abs(weights_hi).item()
    print('metric_it_1_best_case = ', metric_it_1_best_case)
    print('sigmoid(metric_it_1_best_case) = ', torch.sigmoid(torch.tensor(metric_it_1_best_case, dtype=torch.float32)))

    print('\n')
    print('TO CALCULATE FT')
    print('weight_if = ', weights_if)
    print('bias_if = ', biases_if)
    print('weight_hf = ', weights_hf)
    print('bias_hf = ', biases_hf)

    metric_ft_1 = min(weights_if[0].item(), weights_if[1].item()) + biases_if.item() + biases_hf.item() - torch.abs(weights_hf[run]).item()
    # metrics_ft.append(metric_ft_1)
    print('metric_ft_1 = ', metric_ft_1)
    print('sigmoid(metric_ft_1) = ', torch.sigmoid(torch.tensor(metric_ft_1, dtype=torch.float32)))
    metric_ft_1_best_case = max(weights_if[0].item(), weights_if[1].item()) + biases_if.item() + \
                            biases_hf.item() + torch.abs(weights_hf).item()
    print('metric_ft_1_best_case = ', metric_ft_1_best_case)
    print('sigmoid(metric_ft_1_best_case) = ', torch.sigmoid(torch.tensor(metric_ft_1_best_case, dtype=torch.float32)))

    print('\n')
    print('TO CALCULATE GT (C TILDE IN THE PAPER)')
    print('weight_ig = ', weights_ig)
    print('bias_ig = ', biases_ig)
    print('weight_hg = ', weights_hg)
    print('bias_hg = ', biases_hg)

    metric_ctilde_open = weights_ig[0].item() + biases_ig.item() + biases_hg.item() + torch.abs(
        weights_hg).item()
    # metrics_ctilde_open.append(metric_ctilde_open)
    metric_ctilde_open_worst_case = weights_ig[0].item() + biases_ig.item() + biases_hg.item() - torch.abs(
        weights_hg).item()
    # metrics_ctilde_open_worst_case.append(metric_ctilde_open_worst_case)

    metric_ctilde_close = weights_ig[1].item() + biases_ig.item() + biases_hg.item() - torch.abs(
        weights_hg).item()
    # metrics_ctilde_close.append(metric_ctilde_close)

    metric_ctilde_close_worst_case = weights_ig[1].item() + biases_ig.item() + biases_hg.item() + torch.abs(
        weights_hg).item()
    # metrics_ctilde_close_worst_case.append(metric_ctilde_close_worst_case)

    print('metric_ctilde_open = ', metric_ctilde_open)
    print('metric_ctilde_close = ', metric_ctilde_close)

    print('tanh(metric_ctilde_open) = ', torch.tanh(torch.tensor(metric_ctilde_open, dtype=torch.float32)))
    print('tanh(metric_ctilde_close) = ', torch.tanh(torch.tensor(metric_ctilde_close, dtype=torch.float32)))

    print('metric_ctilde_open worst case = ', metric_ctilde_open_worst_case)
    print('metric_ctilde_close worst case = ', metric_ctilde_close_worst_case)

    print('tanh(metric_ctilde_open worst case) = ',
          torch.tanh(torch.tensor(metric_ctilde_open_worst_case, dtype=torch.float32)))
    print('tanh(metric_ctilde_close worst case) = ',
          torch.tanh(torch.tensor(metric_ctilde_close_worst_case, dtype=torch.float32)))

    print('\n')
    print('TO CALCULATE OT')
    print('weight_io = ', weights_io)
    print('bias_io = ', biases_io)
    print('weight_ho = ', weights_ho)
    print('bias_ho = ', biases_ho)

    metric_ot = biases_io.item() + biases_ho.item() - max(weights_io[0].item(),
                                                                    weights_io[1].item()) - weights_ho.item()
    # metrics_ot.append(metric_ot)
    print('metric_ot = ', metric_ot)
    print('sigmoid(metric_ot) = ', torch.sigmoid(torch.tensor(metric_ot, dtype=torch.float32)))
    print('\n')

    return weights_if, weights_ii, weights_ig, weights_io, biases_if, biases_ii, biases_ig, biases_io, \
           weights_hf, weights_hi, weights_hg, weights_ho, biases_hf, biases_hi, biases_hg, biases_ho, \
           metric_ft_1, metric_ft_1_best_case, metric_it_1, metric_it_1_best_case, metric_ctilde_open, metric_ctilde_open_worst_case, \
           metric_ctilde_close, metric_ctilde_close_worst_case, metric_ot


def inspect_model_parameters():
    input_size = 2
    num_classes = 2
    output_activation='Sigmoid'

    dfs_read = read_sheets()
    runs = []

    avg_train_losses = []
    avg_val_losses = []
    avg_long_val_losses = []
    epochs = []
    inverse_avg_train_losses = []
    inverse_avg_val_losses = []
    inverse_avg_long_val_losses = []

    if model_name=='VanillaLSTM':
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

        metrics_ft = []
        metrics_it_1 = []
        metrics_it_constant = []
        metrics_ot = []
        metrics_ctilde_open = []
        metrics_ctilde_close = []
        metrics_ctilde_open_worst_case = []
        metrics_ctilde_close_worst_case = []

    elif model_name=='VanillaReLURNN':
        weights_ih = []
        weights_hh = []
        biases_ih = []
        biases_hh = []

        metric_wi = []
        metric_bi = []
        metric_wh = []
        metric_bh = []


    for run in range(num_runs):

        df = dfs_read[run]
        losses_train = df['Average training losses']
        losses_train = losses_train.tolist()
        losses_val = df['Average validation losses']
        losses_val = losses_val.tolist()
        losses_long_val = df['Average long validation losses']
        losses_long_val = losses_long_val.tolist()
        checkpoint_count = 0

        print('*************************************')
        print('RUN ',run)
        print('*************************************')

        for epoch in range(num_epochs):
            if epoch%checkpoint_step==0 and checkpoint_count<=num_checkpoints:
                checkpoint_count+=1
                runs.append(run)
                avg_train_losses.append(losses_train[epoch])
                inverse_avg_train_losses.append(1/losses_train[epoch])
                avg_val_losses.append(losses_val[epoch])
                inverse_avg_val_losses.append(1 / losses_val[epoch])
                avg_long_val_losses.append(losses_long_val[epoch])
                inverse_avg_long_val_losses.append(1 / losses_long_val[epoch])
                epochs.append(epoch)
                checkpoint_model = select_model(model_name,input_size,hidden_size,num_layers,batch_size,num_classes,output_activation)
                # checkpoint_model.to(device)
                checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"

                checkpt = torch.load(checkpoint_path)
                checkpoint_model.load_state_dict(checkpt['model_state_dict'])
                checkpoint_model.to(device)
                if model_name == 'VanillaLSTM':
                    print(model_name)
                    weight_if, weight_ii, weight_ig, weight_io, bias_if, bias_ii, bias_ig, bias_io, \
                    weight_hf, weight_hi, weight_hg, weight_ho, bias_hf, bias_hi, bias_hg, bias_ho, \
                    metric_ft_1, metric_ft_1_best_case, metric_it_1, metric_it_1_best_case, metric_ctilde_open, metric_ctilde_open_worst_case, \
                    metric_ctilde_close, metric_ctilde_close_worst_case, metric_ot = inspect_lstm(checkpoint_model)


                    weights_if.append(weight_if)
                    weights_ii.append(weight_ig)
                    weights_ig.append(weight_ig)
                    weights_io.append(weight_io)
                    biases_if.append(bias_if)
                    biases_ii.append(bias_ii)
                    biases_ig.append(bias_ig)
                    # finish appending the rest to the arrays, do the same for non checkpoints, and for RNNs

        mdl = modelname + 'run' + str(run) + '.pth'
        model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes,
                             output_activation)
        model.load_state_dict(torch.load(mdl))
        model.to(device)

        # print('*************************************')
        # print('RUN ',run)
        # print('*************************************')
        if model_name=='VanillaReLURNN':
            # print(model_name)
            # print(model.rnn.Variables.weights)
            # print(model.rnn.Variables.biases)
            # weights = []
            # biases = []

            # print(model.rnn.named_parameters())
            for param in model.rnn.named_parameters():
                # if 'weight' in param[0]:
                #     weights.append(param[1])
                # elif 'bias' in param[0]:
                #     biases.append(param[1])

                if 'weight_ih' in param[0]:
                    weights_ih.append(param[1])
                elif 'bias_ih' in param[0]:
                    biases_ih.append(param[1])
                elif 'weight_hh' in param[0]:
                    weights_hh.append(param[1])
                elif 'bias_hh' in param[0]:
                    biases_hh.append(param[1])


            print('weight_ih = ',weights_ih[run])
            print('bias_ih = ',biases_ih[run])
            print('weight_hh = ',weights_hh[run])
            print('bias_hh = ',biases_hh[run])
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

            # print(weights)
            # print(biases)
            # print('RNN weight_ih = ',model.rnn.weight_ih_l)
            # print('RNN weight_hh = ',model.rnn.weight_hh_l)
            # print('RNN bias_ih = ',model.rnn.bias_ih_l)
            # print('RNN bias_hh = ',model.rnn.bias_hh_l)

        elif model_name=='VanillaLSTM':
            print(model_name)



            # weights_ii = []
            # weights_if = []
            # weights_ig = []
            # weights_io = []
            #
            # biases_ii = []
            # biases_if = []
            # biases_ig = []
            # biases_io = []
            #
            # weights_hi = []
            # weights_hf = []
            # weights_hg = []
            # weights_ho = []
            #
            # biases_hi = []
            # biases_hf = []
            # biases_hg = []
            # biases_ho = []



            # for param in model.lstm.named_parameters():
            #     if 'weight_hh' in param[0]:
            #         weights_hh = param[1]
            #         weights_hi.append(weights_hh[0])
            #         weights_hf.append(weights_hh[1])
            #         weights_hg.append(weights_hh[2])
            #         weights_ho.append(weights_hh[3])
            #
            #
            #
            #     elif 'weight_ih' in param[0]:
            #         weights_ih = param[1]
            #         weights_ii.append(weights_ih[0])
            #         weights_if.append(weights_ih[1])
            #         weights_ig.append(weights_ih[2])
            #         weights_io.append(weights_ih[3])
            #
            #     elif 'bias_ih' in param[0]:
            #         biases_ih = param[1]
            #         biases_ii.append(biases_ih[0])
            #         biases_if.append(biases_ih[1])
            #         biases_ig.append(biases_ih[2])
            #         biases_io.append(biases_ih[3])
            #
            #     elif 'bias_hh' in param[0]:
            #         biases_hh=param[1]
            #         biases_hi.append(biases_hh[0])
            #         biases_hf.append(biases_hh[1])
            #         biases_hg.append(biases_hh[2])
            #         biases_ho.append(biases_hh[3])
            #
            #     elif 'bias_ih' in param[0]:
            #         biases_ih=param[1]
            #         biases_ii.append(biases_ih[0])
            #         biases_if.append(biases_ih[1])
            #         biases_ig.append(biases_ih[2])
            #         biases_io.append(biases_ih[3])



            # print('\n')
            # print('TO CALCULATE IT')
            # print('weight_ii = ', weights_ii[run])
            # print('bias_ii = ', biases_ii[run])
            # print('weight_hi = ',weights_hi[run])
            # print('bias_hi = ',biases_hi[run])
            #
            # metric_it_1 = min(weights_ii[run][0].item(),weights_ii[run][1].item())+biases_ii[run].item()+biases_hi[run].item()-torch.abs(weights_hi[run]).item()
            # # metric_it_constant = torch.max()
            # metrics_it_1.append(metric_it_1)
            # print('metric_it_1 = ',metric_it_1)
            # print('sigmoid(metric_it_1_min) = ',torch.sigmoid(torch.tensor(metric_it_1,dtype=torch.float32)))
            # metric_it_1_best_case = max(weights_ii[run][0].item(), weights_ii[run][1].item()) + biases_ii[run].item() + biases_hi[
            #     run].item() + torch.abs(weights_hi[run]).item()
            # print('metric_it_1_best_case = ',metric_it_1_best_case)
            # print('sigmoid(metric_it_1_best_case) = ',torch.sigmoid(torch.tensor(metric_it_1_best_case,dtype=torch.float32)))
            #
            # print('\n')
            # print('TO CALCULATE FT')
            # print('weight_if = ',weights_if[run])
            # print('bias_if = ',biases_if[run])
            # print('weight_hf = ',weights_hf[run])
            # print('bias_hf = ',biases_hf[run])
            #
            # metric_ft_1 = min(weights_if[run][0].item(), weights_if[run][1].item())+biases_if[run].item()+biases_hf[run].item()-torch.abs(weights_hf[run]).item()
            # metrics_ft.append(metric_ft_1)
            # print('metric_ft_1 = ',metric_ft_1)
            # print('sigmoid(metric_ft_1) = ',torch.sigmoid(torch.tensor(metric_ft_1,dtype=torch.float32)))
            # metric_ft_1_best_case = max(weights_if[run][0].item(), weights_if[run][1].item()) + biases_if[run].item() + biases_hf[
            #     run].item() + torch.abs(weights_hf[run]).item()
            # print('metric_ft_1_best_case = ',metric_ft_1_best_case)
            # print('sigmoid(metric_ft_1_best_case) = ', torch.sigmoid(torch.tensor(metric_ft_1_best_case, dtype=torch.float32)))
            #
            # print('\n')
            # print('TO CALCULATE GT (C TILDE IN THE PAPER)')
            # print('weight_ig = ',weights_ig[run])
            # print('bias_ig = ',biases_ig[run])
            # print('weight_hg = ',weights_hg[run])
            # print('bias_hg = ',biases_hg[run])
            #
            # metric_ctilde_open = weights_ig[run][0].item()+biases_ig[run].item()+biases_hg[run].item()+torch.abs(weights_hg[run]).item()
            # metrics_ctilde_open.append(metric_ctilde_open)
            # metric_ctilde_open_worst_case = weights_ig[run][0].item() + biases_ig[run].item() + biases_hg[run].item() - torch.abs(
            #     weights_hg[run]).item()
            # metrics_ctilde_open_worst_case.append(metric_ctilde_open_worst_case)
            #
            # metric_ctilde_close = weights_ig[run][1].item() + biases_ig[run].item() + biases_hg[run].item() - torch.abs(
            #     weights_hg[run]).item()
            # metrics_ctilde_close.append(metric_ctilde_close)
            #
            # metric_ctilde_close_worst_case = weights_ig[run][1].item() + biases_ig[run].item() + biases_hg[run].item() + torch.abs(
            #     weights_hg[run]).item()
            # metrics_ctilde_close_worst_case.append(metric_ctilde_close_worst_case)
            #
            # print('metric_ctilde_open = ',metric_ctilde_open)
            # print('metric_ctilde_close = ',metric_ctilde_close)
            #
            # print('tanh(metric_ctilde_open) = ', torch.tanh(torch.tensor(metric_ctilde_open, dtype=torch.float32)))
            # print('tanh(metric_ctilde_close) = ', torch.tanh(torch.tensor(metric_ctilde_close, dtype=torch.float32)))
            #
            # print('metric_ctilde_open worst case = ', metric_ctilde_open_worst_case)
            # print('metric_ctilde_close worst case = ', metric_ctilde_close_worst_case)
            #
            # print('tanh(metric_ctilde_open worst case) = ', torch.tanh(torch.tensor(metric_ctilde_open_worst_case, dtype=torch.float32)))
            # print('tanh(metric_ctilde_close worst case) = ', torch.tanh(torch.tensor(metric_ctilde_close_worst_case, dtype=torch.float32)))
            #
            # print('\n')
            # print('TO CALCULATE OT')
            # print('weight_io = ',weights_io[run])
            # print('bias_io = ',biases_io[run])
            # print('weight_ho = ',weights_ho[run])
            # print('bias_ho = ',biases_ho[run])
            #
            # metric_ot = biases_io[run].item() + biases_ho[run].item()-max(weights_io[run][0].item(),weights_io[run][1].item())-weights_ho[run].item()
            # metrics_ot.append(metric_ot)
            # print('metric_ot = ',metric_ot)
            # print('sigmoid(metric_ot) = ',torch.sigmoid(torch.tensor(metric_ot,dtype=torch.float32)))
            # print('\n')





inspect_model_parameters()