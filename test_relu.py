import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Dyck_Generator_Suzgun_Batch import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from Dyck1_Datasets import NextTokenPredictionValidationDataset, NextTokenPredictionDataset1000tokens
from torch.optim.lr_scheduler import StepLR
import math
import time
import matplotlib.pyplot as plt
import os

from models_batch import VanillaReLURNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if os.path.exists('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'):
    prefix='/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
elif os.path.exists('/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'):
    prefix='/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'


#
# excel_name_a_dev_models = prefix+'INDICATORS_handmade_models_a_dev_test.xlsx'
# excel_name_u_dev_models = prefix+'INDICATORS_handmade_models_u_dev_test.xlsx'
# prefix='/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'

excel_name = prefix+'INDICATORS_handmade_models_test.xlsx'

class VanillaReLURNN__(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid', rnn_input_weight=[1,-1], rnn_hidden_weight=[1]):
        super(VanillaReLURNN__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}r
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', bias=False)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.rnn.weight_ih_l0=nn.Parameter(torch.tensor([rnn_input_weight], dtype=torch.float32))
        self.rnn.weight_hh_l0=nn.Parameter(torch.tensor([rnn_hidden_weight], dtype=torch.float32))
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight=nn.Parameter(torch.tensor([[1],[1]],dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor([1,-0.5], dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        # print(x)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)
        # print(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)
        # print(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):

            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        return Y_hat_out.to(device)
        # return torch.tensor([Y_hat2])


# model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid', batch_size=1)
#
# # print(param[1][0][0] for param in model.rnn.named_parameters())
# for param in model.rnn.named_parameters():
#     # print(param[1][0])
#     # print(param[0][0])
#     print(param)
#
# for param in model.fc2.named_parameters():
#     print(param)
#
# input1 = torch.tensor([[[1., 0.],
#          [0., 1.]]], dtype=torch.float32).to(device)
#
# print(len(input1[0]))






# for i in range(len(input1[0])):
#     print(model(input1[0][i].unsqueeze(dim=0).unsqueeze(dim=0), length=torch.tensor(2,dtype=torch.float32).unsqueeze(dim=0)))

# print(model(input1,torch.tensor(len(input1),dtype=torch.float32).unsqueeze(dim=0)))

# output_seq = model(input1, torch.tensor(len(input1),dtype=torch.float32).unsqueeze(dim=0))
# output_seq = model(input1, torch.tensor([len(input1)],dtype=torch.float32))
# print(output_seq)
# print('************')

epsilon=0.5

vocab=['(', ')']
n_letters = len(vocab)

NUM_PAR = 1
MIN_SIZE = 2
MAX_SIZE = 50
P_VAL = 0.5
Q_VAL = 0.25

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)

batch_size=1
shuffle_dataset=False

def collate_fn(batch):

    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    # print('labels in collate function  = ',labels)
    lengths = [len(sentence) for sentence in sentences]

    sentences.sort(key=len, reverse=True)
    labels.sort(key=len,reverse=True)
    lengths.sort(reverse=True)


    seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=batch_size)

    return sentences, labels, seq_tensor.to(device), labels_tensor.to(device), lengths_tensor


validation_dataset = NextTokenPredictionValidationDataset()


validation_loader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)

long_dataset = NextTokenPredictionDataset1000tokens()
long_loader = DataLoader(long_dataset,batch_size=1, shuffle=shuffle_dataset,collate_fn=collate_fn)

# devs_a = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# devs_b = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# # devs_abratio = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# devs_uvalue = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# 
# 
# weights_input = []
# weights_u = []
# 
# for i in range(len(devs_a)):
#     for j in range(len(devs_b)):
#         for k in range(len(devs_uvalue)):
#             weight_a = 1-devs_a[i]
#             weight_b = -1-devs_b[j]
#             # weight_u = torch.tensor([[1-devs_uvalue[k]]],dtype=torch.float32)
#             weight_u = 1 - devs_uvalue[k]
#             weights_u.append(weight_u)
#             # weight_input=torch.tensor([[weight_a, weight_b]], dtype=torch.float32)
#             weight_input = [weight_a, weight_b]
#             weights_input.append(weight_input)
#             # weights_input.append([1,-1])
# 
# print(weights_input)
# print(len(weights_input))
# print(weights_u)
# print(len(weights_u))

# devs =[-1e-6, -1.5e-5,-1e-5, -1.5e-4, -1e-4, -1.5e-3, -1e-3, -1.5e-2, -1e-2, -1.5e-1, -1e-1, -0.5e-1, 0.5e-1,1e-1, 1.5e-1, 1e-2, 1.5e-2, 1e-3, 1.5e-3, 1e-4, 1.5e-4, 1e-5, 1.5e-5, 1e-6]
# print('devs = ',devs)
# print(len(devs))
# 
# models = []
# for i in range(len(weights_input)):
#     model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid', batch_size=1, rnn_input_weight=weights_input[i], rnn_hidden_weight=weights_u[i])
#     models.append(model)
# print(len(models))
# # print(models[0])
# 
# 
# models_dev_u = []
# for i in range(len(devs)):
#     model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid',
#                              batch_size=1, rnn_input_weight=[1,-1], rnn_hidden_weight=1+devs[i])
#     models_dev_u.append(model)
# print(len(models_dev_u))
# 
# models_dev_a = []
# for i in range(len(devs)):
#     model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid',
#                              batch_size=1, rnn_input_weight=[1+devs[i],-1], rnn_hidden_weight=1)
#     models_dev_a.append(model)
# print(len(models_dev_a))

def encode_batch(sentences, labels, lengths, batch_size):

    max_length = max(lengths)
    # print(max_length)
    sentence_tensor = torch.zeros(batch_size,max_length,len(vocab))

    labels_tensor = torch.tensor([])
    for i in range(batch_size):

        sentence = sentences[i]
        labels_tensor = torch.cat((labels_tensor, Dyck.lineToTensorSigmoid(labels[i],max_len=max_length)))
        # labels_tensor = torch.cat((labels_tensor,Dyck.batchToTensorSigmoid(labels,lengths,batch_size,max_length)))
        if len(sentence)<max_length:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos]=1
    sentence_tensor.requires_grad_(True)
    # lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64).cpu()
    # print('labels tensor = ',labels_tensor)
    return sentence_tensor, labels_tensor, lengths_tensor


def collate_fn(batch):

    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    # print('labels in collate function  = ',labels)
    lengths = [len(sentence) for sentence in sentences]

    sentences.sort(key=len, reverse=True)
    labels.sort(key=len,reverse=True)
    lengths.sort(reverse=True)


    seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=batch_size)

    return sentences, labels, seq_tensor.to(device), labels_tensor.to(device), lengths_tensor


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

def validate_model(model, loader):
    # model.eval()
    num_correct = 0
    log_file=''


    if loader==validation_loader:
        # log_file = validation_log
        # dataset='Validation Set'
        ds = validation_dataset
    # elif loader==train_loader:
        # log_file = train_validation_log
        # dataset = 'Train Set'
        # ds = train_dataset
    criterion = nn.MSELoss()

    total_loss = 0
    # losses = []

    # with open(log_file,'a') as f:
    #     f.write('////////////////////////////////////////\n')
    #     f.write('TEST '+dataset+'\n')


    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)

        output_seq = model.mask(output_seq, target_seq, length)
        loss = criterion(output_seq,target_seq)
        total_loss+=loss.item()


        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)


        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()


        for j in range(batch_size):

            if torch.equal(out_seq[j], target_seq[j]):

                num_correct += 1




    accuracy = num_correct / len(ds) * 100
    # with open(log_file, 'a') as f:
    #     if loader==validation_loader:
    # 
    #         f.write('val accuracy for run'+str(run)+' epoch '+str(epoch)+' = ' + str(accuracy)+'%, val loss = '+str(loss.item()/len(ds)) + '\n')


    return accuracy, loss.item()/len(ds)


def test_model(model, loader, dataset):
    """
    add a function here to calculate the average point where the model fails.
    if the model gets everything correct then it wont be counted in the values which fail at any point
    scatter plots in the main function after all models have been evaluated
    one scatter plot for long sequences, one for very long sequences

    """

    if dataset=='validation':
        # log_file=test_log
        ds = validation_dataset
    elif dataset=='long':
        ds = long_dataset

    correct_guesses = []
    incorrect_guesses = []
    correct_guesses_length = []
    incorrect_guesses_length = []
    incorrect_guesses_first_fail = []
    sum_first_fail_points = 0

    criterion = nn.MSELoss()
    total_loss = 0


    max_depths_correct_guesses = []
    timestep_depths_correct_guesses = []
    max_depths_incorrect_guesses = []
    timestep_depths_incorrect_guesses = []


    model.eval()
    num_correct = 0

    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):

        output_seq = model(input_seq.to(device), length)

        output_seq = model.mask(output_seq, target_seq, length)
        loss = criterion(output_seq, target_seq)
        total_loss += loss.item()



        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)


        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()



        for j in range(batch_size):

            # if out_np[j].all() == target_np[j].all():
            max_depth, timestep_depth = get_timestep_depths(sentences[j])
            if torch.equal(out_seq[j], target_seq[j]):
                num_correct += 1
                correct_guesses.append(sentences[j])
                correct_guesses_length.append(length[j].item())
                # incorrect_guesses_first_fail.append(length[j].item())
                sum_first_fail_points+=length[j].item()
                max_depths_correct_guesses.append(max_depth)
                timestep_depths_correct_guesses.append(timestep_depth)

            else:
                incorrect_guesses.append(sentences[j])
                for k in range(length[j]):
                    if torch.equal(out_seq[j][k], target_seq[j][k]) != True:
                        incorrect_guesses_first_fail.append(k)
                        sum_first_fail_points+=k
                        incorrect_guesses_length.append(length[j].item())
                        max_depths_incorrect_guesses.append(max_depth)
                        timestep_depths_incorrect_guesses.append(timestep_depth)
                        break




    accuracy = num_correct / len(ds) * 100
    # with open(log_file, 'a') as f:
    #     f.write('accuracy = ' + str(accuracy)+'%' + '\n')
    print(''+dataset+' test accuracy = '+ str(accuracy)+'%')
    # avg_first_fail_point = sum_first_fail_points/len(incorrect_guesses)
    avg_first_fail_point = sum_first_fail_points / (len(incorrect_guesses)+num_correct)

    # return accuracy, correct_guesses,correct_guesses_length, incorrect_guesses, incorrect_guesses_length, incorrect_guesses_first_fail,avg_first_fail_point, max_depth, timestep_depth
    return accuracy, total_loss/len(ds), correct_guesses,correct_guesses_length, incorrect_guesses, incorrect_guesses_length, incorrect_guesses_first_fail,avg_first_fail_point, max_depths_correct_guesses, timestep_depths_correct_guesses, max_depths_incorrect_guesses, timestep_depths_incorrect_guesses


# def main2():
#
#     val_accuracies_ideal = []
#     val_losses_ideal = []
#     long_accuracies_ideal=[]
#     num_correct_ideal=[]
#     val_fpfs_ideal = []
#     long_fpfs_ideal=[]
#
#     val_accuracies_u_dev = []
#     val_losses_u_dev = []
#     long_accuracies_u_dev = []
#     long_fpfs_u_dev = []
#     num_correct_u_dev = []
#     val_fpfs_u_dev = []
#
#
#
#     val_accuracies_a_dev = []
#     val_losses_a_dev = []
#     long_accuracies_a_dev = []
#     long_fpfs_a_dev = []
#     num_correct_a_dev = []
#     val_fpfs_a_dev = []
#
#     models_dev_u = []
#     for i in range(len(devs)):
#         model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid',
#                                  batch_size=1, rnn_input_weight=[1, -1], rnn_hidden_weight=1 + devs[i])
#         models_dev_u.append(model)
#     print(len(models_dev_u))
#
#     models_dev_a = []
#     for i in range(len(devs)):
#         model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid',
#                                  batch_size=1, rnn_input_weight=[1 + devs[i], -1], rnn_hidden_weight=1)
#         models_dev_a.append(model)
#
#
#     # criterion = nn.MSELoss()
#     df_u_dev = pd.DataFrame()
#     df_a_dev = pd.DataFrame()
#
#     for i in range(len(models_dev_u)):
#         # for i, (sentences, labels, input_seq, target_seq, length) in enumerate(validation_loader):
#         #     model.zero_grad()
#         #     output_seq = model(input_seq.to(device), length)
#         #     output_seq = model.mask(output_seq, target_seq, length)
#         #     loss = criterion(output_seq, target_seq)
#         # val_accuracy_u_dev, val_loss_u_dev = validate_model(models_dev_u[i],validation_loader)
#         val_accuracy, val_correct_guesses, val_correct_guesses_length, val_incorrect_guesses, val_incorrect_guesses_length, val_incorrect_guesses_first_fail, val_avg_first_fail_point, max_depth_correct, timestep_depth_correct, max_depth_incorrect, timestep_depth_incorrect = test_model(
#             model, validation_loader, 'short')
#
#         # val_accuracies_u_dev.append(val_accuracy_u_dev)
#         # val_losses_u_dev.append(val_loss_u_dev)


def main():


    input_size = n_letters

    model_a_weights = []
    model_b_weights = []
    model_u_weights = []
    model_a_devs = []
    model_u_devs = []
    # model_ab_ratios = []
    model_ab_ratio_devs_absolute = []
    model_ab_ratio_devs_signed = []
    model_euclidean_norms = []


    test_accuracies = []
    test_losses = []
    correct_guesses = []
    correct_guesses_lengths = []
    incorrect_guesses = []
    incorrect_guesses_lengths = []
    incorrect_guesses_first_fail = []
    avg_point_of_failure_short = []
    max_depths_correct_guesses = []
    timestep_depths_correct_guesses = []
    max_depths_incorrect_guesses = []
    timestep_depths_incorrect_guesses = []



    long_test_accuracies = []
    long_losses = []
    long_correct = []
    long_correct_lengths = []
    long_incorrect = []
    long_incorrect_lengths = []
    long_incorrect_first_fail = []
    long_avg_point_of_failure_short = []
    long_max_depths_correct_guesses = []
    long_timestep_depths_correct_guesses = []
    long_max_depths_incorrect_guesses = []
    long_timestep_depths_incorrect_guesses = []

    # val_accuracies = []
    # val_losses = []
    # val_correct_guesses_all=[]
    # val_correct_guesses_lengths = []
    # val_incorrect_guesses_all=[]
    # val_incorrect_guesses_lengths = []
    # val_incorrect_guesses_fpfs=[]
    # val_avg_fpfs=[]
    # val_max_depths_correct= []
    # val_timestep_depths_correct = []
    # val_max_depths_incorrect = []
    # val_timestep_depths_incorrect = []
    #
    #
    # long_accuracies = []
    # long_losses = []
    # long_correct_guesses_all=[]
    # long_correct_guesses_lengths = []
    # long_incorrect_guesses_all=[]
    # long_incorrect_guesses_lengths = []
    # long_incorrect_guesses_fpfs=[]
    # long_avg_fpfs=[]
    # long_max_depths_correct= []
    # long_timestep_depths_correct = []
    # long_max_depths_incorrect = []
    # long_timestep_depths_incorrect = []
    

    devs_a = [0.004, -0.004, 0.008, -0.008, 0.012, -0.012, 0.016, -0.016, 0.02, -0.02, 0]
    # devs_a = []
    devs_u = [0.0001, -0.0001, 0.0002, -0.0002, 0.0003, -0.0003, 0.0004, -0.0004, 0.0005, -0.0005, 0]
    models = []

    for i in range(len(devs_a)):
        for j in range(len(devs_u)):
            model_a_weight = 1+devs_a[i]
            model_b_weight = -1
            model_u_weight = 1+devs_u[j]
            model_ = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, batch_size=1, output_activation='Sigmoid', rnn_input_weight=[model_a_weight,model_b_weight], rnn_hidden_weight=[model_u_weight])
            models.append(model_)
            model_a_weights.append(model_a_weight)
            model_b_weights.append(model_b_weight)
            model_u_weights.append(model_u_weight)
            model_a_devs.append(devs_a[i])
            model_u_devs.append(devs_u[j])
            model_ab_ratio = model_a_weight/model_b_weight
            model_ab_ratio_dev = model_ab_ratio--1
            model_ab_ratio_devs_signed.append(model_ab_ratio_dev)
            model_ab_ratio_devs_absolute.append(abs(model_ab_ratio_dev))
            model_euclidean_norm = math.sqrt((model_ab_ratio_dev**2)+(devs_u[j]**2))
            model_euclidean_norms.append(model_euclidean_norm)

    


    # model = VanillaReLURNN__(input_size=2, hidden_size=1, output_size=2, num_layers=1, output_activation='Sigmoid',
    #                          batch_size=1, rnn_input_weight=[1.,-1],rnn_hidden_weight=[0.9995])

    # model.to(device)
    # test_accuracy, test_correct_guesses,test_correct_guesses_length, test_incorrect_guesses, test_incorrect_guesses_length, test_incorrect_guesses_first_fail,test_avg_first_fail_point, test_max_depth, test_timestep_depth = test_model(model, test_loader, 'short')
    
    for i in range(len(models)):
        model=models[i].to(device)
        print('*******************************************************************')
        print('model ',i)
        test_accuracy, test_loss, test_correct_guesses, test_correct_guesses_length, test_incorrect_guesses, test_incorrect_guesses_length, test_incorrect_guesses_first_fail, test_avg_first_fail_point, max_depth_correct, timestep_depth_correct, max_depth_incorrect, timestep_depth_incorrect = test_model(model, validation_loader, 'validation')

        print('validation test accuracy = ',test_accuracy)
        # print('avg fpf = ',test_avg_first_fail_point)
    
    
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        correct_guesses.append(test_correct_guesses)
        correct_guesses_lengths.append(test_correct_guesses_length)
        incorrect_guesses.append(test_incorrect_guesses)
        incorrect_guesses_lengths.append(test_incorrect_guesses_length)
        incorrect_guesses_first_fail.append(test_incorrect_guesses_first_fail)
        avg_point_of_failure_short.append(test_avg_first_fail_point)
        max_depths_correct_guesses.append(max_depth_correct)
        max_depths_incorrect_guesses.append(max_depth_incorrect)
        timestep_depths_correct_guesses.append(timestep_depth_correct)
        timestep_depths_incorrect_guesses.append(timestep_depth_incorrect)
    
        long_accuracy, long_loss, long_correct_guesses, long_correct_guesses_length, long_incorrect_guesses, long_incorrect_guesses_length, long_incorrect_guesses_first_fail, long_avg_first_fail_point, long_max_depth_correct, long_timestep_depth_correct, long_max_depth_incorrect, long_timestep_depth_incorrect = test_model(model, long_loader, 'long')
        print('long accuracy = ',long_accuracy)
        # print('long correct guesses = ',long_correct_guesses)
        print('long fpf = ',long_avg_first_fail_point)

        long_test_accuracies.append(long_accuracy)
        long_losses.append(long_loss)
        long_correct.append(long_correct_guesses)
        long_correct_lengths.append(long_correct_guesses_length)
        long_incorrect.append(long_incorrect_guesses)
        long_incorrect_lengths.append(long_incorrect_guesses_length)
        long_incorrect_first_fail.append(long_incorrect_guesses_first_fail)
        long_avg_point_of_failure_short.append(long_avg_first_fail_point)
        long_max_depths_correct_guesses.append(long_max_depth_correct)
        long_max_depths_incorrect_guesses.append(long_max_depth_incorrect)
        long_timestep_depths_correct_guesses.append(long_timestep_depth_correct)
        long_timestep_depths_incorrect_guesses.append(long_timestep_depth_incorrect)



    df = pd.DataFrame()
    df['model_a_weights'] = model_a_weights
    df['model_b_weights']=model_b_weights
    df['model_u_weights']=model_u_weights
    df['model_u_devs']=model_u_devs
    df['model_a_devs'] = model_a_devs
    df['model_ab_ratios_devs_absolute']=model_ab_ratio_devs_absolute
    df['model_ab_ratios_devs_signed']=model_ab_ratio_devs_signed
    df['model_euclidean_norms']=model_euclidean_norms
    df['val_accuracies'] = test_accuracies
    df['val_losses'] = test_losses
    df['log_val_losses']=np.log(test_losses)
    df['neg_log_val_losses']= -1*np.log(test_losses)
    df['val_correct_guesses']=correct_guesses
    df['val_correct_guesses_lengths']=correct_guesses_lengths
    df['val_incorrect_guesses']=incorrect_guesses
    df['val_incorrect_guesses_lengths']=incorrect_guesses_lengths
    df['val_incorrect_guesses_first_fail']=incorrect_guesses_first_fail
    df['val_avg_point_of_failure_short']=avg_point_of_failure_short
    df['val_max_depths_correct_guesses']=max_depths_correct_guesses
    df['val_timestep_depths_correct_guesses']=timestep_depths_correct_guesses
    df['val_max_depths_incorrect_guesses']=max_depths_incorrect_guesses
    df['val_timestep_depths_incorrect_guesses']=timestep_depths_incorrect_guesses
    df['long_test_accuracies']=long_test_accuracies
    df['long_losses']=long_losses
    df['log_long_losses']=np.log(long_losses)
    df['neg_log_long_losses'] = -1*np.log(long_losses)
    df['long_correct_guesses']=long_correct
    df['long_correct_guesses_lengths']=long_correct_lengths
    df['long_incorrect_guesses']=long_incorrect
    df['long_incorrect_guesses_lengths']=long_incorrect_lengths
    df['long_incorrect_guesses_first_fail']=long_incorrect_first_fail
    df['long_avg_point_of_failure']=long_avg_point_of_failure_short
    df['long_max_depths_correct_guesses']=long_max_depths_correct_guesses
    df['long_timestep_depths_correct_guesses']=long_timestep_depths_correct_guesses
    df['long_max_depths_incorrect_guesses']=long_max_depths_incorrect_guesses
    df['long_timestep_depths_incorrect_guesses']=long_timestep_depths_incorrect_guesses

    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    df.to_excel(writer, index=False)
    writer.save()











if __name__=='__main__':
    main()

