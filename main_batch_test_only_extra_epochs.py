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
    NextTokenPredictionDataset2000tokens_nested, NextTokenPredictionDataset2000tokens_zigzag, NextTokenPredictionDataset1000tokens

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)


"""
Steps:

- Read the excel sheets and save them into a list of dataframes
- Create arrays of train losses, validation losses, long validation losses (from the values in dataframes)
- loop based on runs
- Import the model based on the arguments from arg parser
- Test on the long and very long test sets
- loop based on checkpoint step
- Import the saved models from every checkpoint
- 


"""

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
# parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or appended')
parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or concatenated')
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
lr_scheduler_gamma = args.lr_scheduler_gamma
lr_scheduler_step = args.lr_scheduler_step
num_checkpoints = args.num_checkpoints
dataset_type = args.dataset_type
extra_epochs = args.extra_epochs


# best_run = args.best_run
#
# if best_run==-1:
#     best_run = num_runs-1

checkpoint_step = int(num_epochs/4)
if args.checkpoint_step!=0:
    checkpoint_step = args.checkpoint_step

shuffle_dataset = args.shuffle_dataset

# model_name = 'VanillaLSTM'
# task = 'NextTokenPrediction'
# feedback='EveryTimeStep'
# hidden_size = 3
# num_layers = 1
# learning_rate = 0.01
# num_epochs = 5
# num_runs = 10
# batch_size = 100


use_optimiser='Adam'

num_bracket_pairs = 25

length_bracket_pairs = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = ['(', ')']
# vocab = {'PAD':0, '(':1,')':2}
tags = {'':0, '0':1, '1':2}
n_letters= len(vocab)
n_tags = len(tags)-1
num_bracket_pairs = 25
length_bracket_pairs = 50

# batch_size = 4

pad_token=0

# NUM_PAR = 1
# MIN_SIZE = 102
# MAX_SIZE = 500
# P_VAL = 0.5
# Q_VAL = 0.25
#
#
# epsilon=0.5
#
# # train_size = 10000
# test_size = 10000
# long_size = 10000
#
# Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)


NUM_PAR = 1
MIN_SIZE = 950
MAX_SIZE = 1000
P_VAL = 0.5
Q_VAL = 0.25


epsilon=0.5

# train_size = 10000
# test_size = 10000
# long_size = 10000

# len(NextTokenPredictionDataset950to1000tokens())

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)


# path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"

# path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
#        +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"

# path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
#        +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
#        +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"

path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"\
       +str(batch_size)+"_batch_size/"+str(learning_rate)+"_learning_rate/"+str(num_epochs)+"_epochs/"\
       +str(lr_scheduler_step)+"_lr_scheduler_step/"+str(lr_scheduler_gamma)+"_lr_scheduler_gamma/"\
       +str(hidden_size)+"_hidden_units/"+str(num_runs)+"_runs/shuffle_"+str(shuffle_dataset)+"/"


# print('model_name = ',model_name)
# print('task = ',task)
# print('feedback = ',feedback)
# print('hidden_size = ',hidden_size)
# print('num_layers = ',num_layers)
# print('learning_rate = ',learning_rate)
# print('num_epochs = ',num_epochs)
# print('num_runs = ',num_runs)
# # print('load_model = ',load_model)

print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('batch_size = ',batch_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)
print('shuffle = ',shuffle_dataset)
print('dataset_type = ',dataset_type)




# file_name = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.txt'
# excel_name = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.xlsx'
# modelname = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL.pth'
# optimname = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_OPTIMISER.pth'
# train_log = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TRAIN_LOG.txt'
# test_log = path+'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TEST_LOG.txt'
# long_test_log = path+'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_LONG_TEST_LOG.txt'
# plot_name = path+'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_PLOT.png'

file_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_INFERENCE' + '.txt'



excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_' +str(extra_epochs)+'extra_epochs'+'.xlsx'


modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL_'
#
# optimname = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_OPTIMISER.pth'
#
# test_log = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + '_TEST_LOG_INFERENCE.txt'
# long_test_log = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + '_LONG_TEST_LOG_INFERENCE.txt'
# plot_name = path+'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+ str(num_runs)+'runs_'+str(checkpoint_step)+"checkpoint_step_"+str(num_checkpoints)+"checkpoints" + '_PLOT.png'

checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'

# scatter_name_train = path+ 'Dyck1_' + task + '_' + str(
#         num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
#         hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
#         num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_102to500tokens_train_loss_SCATTER_PLOT.png'


prefix = path+'INFERENCE_'+dataset_type+'_'+str(checkpoint_step)+'checkpoint_step_upto'+str(num_checkpoints)+'checkpoints_'



excel_name_inference=prefix+'EXCEL INFERENCE_EXTRA_EPOCHS.xlsx'



def encode_batch(sentences, labels, lengths, batch_size):

    max_length = max(lengths)
    # print(max_length)
    sentence_tensor = torch.zeros(batch_size,max_length,len(vocab))

    labels_tensor = torch.tensor([])
    for i in range(len(sentences)):
    # for i in range(batch_size):

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
    if len(lengths_tensor)<batch_size:
        for j in range(batch_size - len(sentences)):
            lengths_tensor=torch.cat((lengths_tensor,torch.tensor(0,dtype=torch.int64)))


    # print('labels tensor = ',labels_tensor)
    return sentence_tensor, labels_tensor, lengths_tensor


def collate_fn(batch):

    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    # max_depth = [batch[i]['max_depth'] for i in range(len(batch))]
    # print('labels in collate function  = ',labels)
    lengths = [len(sentence) for sentence in sentences]
    # max_depths = [batch[i]['max_depth'] for i in range(len(batch))]
    # timestep_depths = [batch_size[i]['timestep_depths'] for i in range(len(batch))]


    sentences.sort(key=len, reverse=True)
    labels.sort(key=len,reverse=True)
    lengths.sort(reverse=True)
    # max_depth.sort(reverse=True)




    # seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels,lengths, batch_size=len(sentences))
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



test_dataset = NextTokenPredictionDataset2000tokens_nested()
test_size = len(test_dataset)

if dataset_type=='nested':
    test_dataset=NextTokenPredictionDataset2000tokens_nested()
elif dataset_type=='zigzag':
    test_dataset=NextTokenPredictionDataset2000tokens_zigzag()
# elif dataset_type=='appended':
elif dataset_type == 'concatenated':
    test_dataset=NextTokenPredictionDataset2000tokens()
elif dataset_type == '1000token':
    test_dataset=NextTokenPredictionDataset1000tokens()

test_size=len(test_dataset)

# train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# long_loader = DataLoader(long_dataset, batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)


# train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False)


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

def read_sheets():
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run"+str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(excel_name,sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs





def main():




    output_activation = 'Sigmoid'

    if task == 'TernaryClassification':
        num_classes = 3
        output_activation = 'Softmax'
    elif task == 'BinaryClassification' or task == 'NextTokenPrediction':
        num_classes = 2
        output_activation = 'Sigmoid'




    input_size = n_letters







    # with open(file_name, 'a') as f:
    #     # f.write('Output activation = ' + output_activation + '\n')
    #     # f.write('Optimiser used = ' + use_optimiser + '\n')
    #     # f.write('Learning rate = ' + str(learning_rate) + '\n')
    #     # f.write('Number of runs = ' + str(num_runs) + '\n')
    #     # f.write('Number of epochs in each run = ' + str(num_epochs) + '\n')
    #     f.write('Saved model name = ' + modelname + '\n')
    #     # f.write('Saved optimiser name = ' + optimname + '\n')
    #     f.write('Excel name = ' + excel_name + '\n')
    #     # f.write('Train log name = ' + train_log + '\n')
    #     f.write('Test log name = ' + test_log + '\n')
    #     f.write('Long test log name = ' + long_test_log + '\n')
    #     f.write('///////////////////////////////////////////////////////////////\n')
    #     f.write('\n')

    dfs_read = read_sheets()
    # train_accuracies = []
    test_accuracies = []
    # long_test_accuracies = []
    # train_dataframes = []
    runs = []
    correct_guesses = []
    correct_guesses_lengths = []
    correct_guesses_max_depth = []
    # correct_guesses_long = []
    # correct_guesses_long_lengths = []


    incorrect_guesses = []
    incorrect_guesses_lengths = []
    # incorrect_guesses_long = []
    # incorrect_guesses_long_lengths = []
    incorrect_guesses_first_fail = []
    incorrect_guesses_max_depth = []
    # incorrect_guesses_long_first_fail = []

    avg_point_of_failure_short = []
    # incorrect_lengths = []
    # avg_point_of_failure_long = []
    # incorrect_lengths_long = []
    avg_train_losses = []
    avg_val_losses = []
    avg_long_val_losses = []
    epochs = []
    inverse_avg_train_losses = []
    inverse_avg_val_losses = []
    inverse_avg_long_val_losses = []

    log_avg_train_losses = []
    log_avg_val_losses = []
    log_avg_long_val_losses = []

    max_depths_correct_guesses = []
    timestep_depths_correct_guesses = []
    max_depths_incorrect_guesses = []
    timestep_depths_incorrect_guesses = []




    for run in range(num_runs):
        df = dfs_read[run]
        losses_train = df['Average training losses']
        losses_train = losses_train.tolist()
        losses_val = df['Average validation losses']
        losses_val=losses_val.tolist()
        losses_long_val = df['Average long validation losses']
        losses_long_val = losses_long_val.tolist()
        # runs.append(run)
        checkpoint_count = 0
        for epoch in range(num_epochs, (num_epochs+extra_epochs), 1):
            if epoch%checkpoint_step==0 and checkpoint_count<=num_checkpoints:
                checkpoint_count+=1
                runs.append(run)
                avg_train_losses.append(losses_train[epoch-num_epochs])
                inverse_avg_train_losses.append(1/losses_train[epoch-num_epochs])
                avg_val_losses.append(losses_val[epoch-num_epochs])
                inverse_avg_val_losses.append(1 / losses_val[epoch-num_epochs])
                avg_long_val_losses.append(losses_long_val[epoch-num_epochs])
                inverse_avg_long_val_losses.append(1 / losses_long_val[epoch-num_epochs])
                epochs.append(epoch)
                checkpoint_model = select_model(model_name,input_size,hidden_size,num_layers,batch_size,num_classes,output_activation)
                # checkpoint_model.to(device)
                checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"

                checkpt = torch.load(checkpoint_path)
                checkpoint_model.load_state_dict(checkpt['model_state_dict'])
                checkpoint_model.to(device)
                # checkpoint_test_accuracy, checkpoint_correct_guesses,checkpoint_correct_guesses_length, checkpoint_incorrect_guesses, checkpoint_incorrect_guesses_length, checkpoint_incorrect_guesses_first_fail,checkpoint_avg_first_fail_point, checkpoint_max_depth, checkpoint_timestep_depth = test_model(checkpoint_model,test_loader,'short')
                checkpoint_test_accuracy, checkpoint_correct_guesses, checkpoint_correct_guesses_length, checkpoint_incorrect_guesses, checkpoint_incorrect_guesses_length, checkpoint_incorrect_guesses_first_fail, checkpoint_avg_first_fail_point, checkpoint_max_depth_correct, checkpoint_timestep_depth_correct, checkpoint_max_depth_incorrect, checkpoint_timestep_depth_incorrect = test_model(checkpoint_model, test_loader, 'short')

                test_accuracies.append(checkpoint_test_accuracy)
                correct_guesses.append(checkpoint_correct_guesses)
                correct_guesses_lengths.append(checkpoint_correct_guesses_length)
                incorrect_guesses.append(checkpoint_incorrect_guesses)
                incorrect_guesses_lengths.append(checkpoint_incorrect_guesses_length)
                incorrect_guesses_first_fail.append(checkpoint_incorrect_guesses_first_fail)
                avg_point_of_failure_short.append(checkpoint_avg_first_fail_point)
                # max_depths_correct_guesses.append(checkpoint_max_depth_correct)
                # max_depths_incorrect_guesses.append(checkpoint_max_depth_incorrect)
                # max_depths.append(checkpoint_max_depth)
                # timestep_depths.append(checkpoint_timestep_depth)
                max_depths_correct_guesses.append(checkpoint_max_depth_correct)
                max_depths_incorrect_guesses.append(checkpoint_max_depth_incorrect)
                timestep_depths_correct_guesses.append(checkpoint_timestep_depth_correct)
                timestep_depths_incorrect_guesses.append(checkpoint_timestep_depth_incorrect)



                # checkpoint_long_accuracy, checkpoint_long_correct_guesses,checkpoint_long_correct_guesses_length, checkpoint_long_incorrect_guesses, checkpoint_long_incorrect_guesses_length, checkpoint_long_incorrect_guesses_first_fail,checkpoint_long_avg_first_fail_point = test_model(checkpoint_model,long_loader,'long')
                # long_test_accuracies.append(checkpoint_long_accuracy)
                # correct_guesses_long.append(checkpoint_long_correct_guesses)
                # correct_guesses_long_lengths.append(checkpoint_long_correct_guesses_length)
                # incorrect_guesses_long.append(checkpoint_long_incorrect_guesses)
                # incorrect_guesses_long_lengths.append(checkpoint_long_incorrect_guesses_length)
                # incorrect_guesses_long_first_fail.append(checkpoint_long_incorrect_guesses_first_fail)
                # avg_point_of_failure_long.append(checkpoint_long_avg_first_fail_point)








        #
        # runs.append(run)
        # epochs.append(num_epochs-1)
        # avg_train_losses.append(losses_train[num_epochs-1])
        # inverse_avg_train_losses.append(1 / losses_train[epoch])
        # avg_val_losses.append(losses_val[num_epochs-1])
        # inverse_avg_val_losses.append(1 / losses_val[epoch])
        # avg_long_val_losses.append(losses_long_val[num_epochs-1])
        # inverse_avg_long_val_losses.append(1 / losses_long_val[epoch])
        # mdl = modelname + 'run' + str(run) + '.pth'
        # model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes,output_activation)
        # model.load_state_dict(torch.load(mdl))
        # model.to(device)
        # # test_accuracy, test_correct_guesses,test_correct_guesses_length, test_incorrect_guesses, test_incorrect_guesses_length, test_incorrect_guesses_first_fail,test_avg_first_fail_point, test_max_depth, test_timestep_depth = test_model(model, test_loader, 'short')
        # test_accuracy, test_correct_guesses, test_correct_guesses_length, test_incorrect_guesses, test_incorrect_guesses_length, test_incorrect_guesses_first_fail, test_avg_first_fail_point, max_depth_correct, timestep_depth_correct, max_depth_incorrect, timestep_depth_incorrect = test_model(model, test_loader, 'short')
        #
        # test_accuracies.append(test_accuracy)
        # correct_guesses.append(test_correct_guesses)
        # correct_guesses_lengths.append(test_correct_guesses_length)
        # incorrect_guesses.append(test_incorrect_guesses)
        # incorrect_guesses_lengths.append(test_incorrect_guesses_length)
        # incorrect_guesses_first_fail.append(test_incorrect_guesses_first_fail)
        # avg_point_of_failure_short.append(test_avg_first_fail_point)
        # max_depths_correct_guesses.append(max_depth_correct)
        # max_depths_incorrect_guesses.append(max_depth_incorrect)
        # timestep_depths_correct_guesses.append(timestep_depth_correct)
        # timestep_depths_incorrect_guesses.append(timestep_depth_incorrect)

        # max_depths.append(test_max_depth)
        # timestep_depths.append(test_timestep_depth)



        # long_test_accuracy, long_correct_guesses,long_correct_guesses_length, long_incorrect_guesses, long_incorrect_guesses_length, long_incorrect_guesses_first_fail,long_avg_first_fail_point = test_model(model, long_loader, 'long')
        # long_test_accuracies.append(long_test_accuracy)
        # correct_guesses_long.append(long_correct_guesses)
        # correct_guesses_long_lengths.append(long_correct_guesses_length)
        # incorrect_guesses_long.append(long_incorrect_guesses)
        # incorrect_guesses_long_lengths.append(long_incorrect_guesses_length)
        # incorrect_guesses_long_first_fail.append(long_incorrect_guesses_first_fail)
        # avg_point_of_failure_long.append(long_avg_first_fail_point)



        # with open(file_name, "a") as f:
        #     # f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
        #     # f.write('test accuracy for 102 to 500 tokens for run '+str(run)+' = ' + str(test_accuracy) + '%\n')
        #     # f.write('long test accuracy for 502 to 1000 tokens for run '+str(run)+' = ' + str(long_test_accuracy) + '%\n')
        #     # f.write('test accuracy for 950 to 1000 tokens for run ' + str(run) + ' = ' + str(test_accuracy) + '%\n')
        #     f.write('test accuracy for 2000 tokens for run ' + str(run) + ' = ' + str(test_accuracy) + '%\n')

    log_avg_train_losses=np.log(avg_train_losses)
    log_avg_val_losses = np.log(avg_val_losses)
    log_avg_long_val_losses = np.log(avg_long_val_losses)

    log_inverse_avg_train_losses = np.log(inverse_avg_train_losses)
    log_inverse_avg_val_losses = np.log(inverse_avg_val_losses)
    log_inverse_avg_long_val_losses = np.log(inverse_avg_long_val_losses)

    df1 = pd.DataFrame()
    df1['run'] = runs
    df1['epoch'] = epochs
    df1['avg training losses'] = avg_train_losses
    df1['avg validation losses'] = avg_val_losses
    df1['avg long validation losses'] = avg_long_val_losses
    df1['log of avg train losses'] = log_avg_train_losses
    df1['log of avg validation losses'] = log_avg_val_losses
    df1['log of avg long validation losses'] = log_avg_long_val_losses
    df1['log of inverse avg train losses'] = log_inverse_avg_train_losses
    df1['log of inverse avg validation losses'] = log_inverse_avg_val_losses
    df1['log of inverse avg long validation losses'] = log_inverse_avg_long_val_losses
    # df1['correct guesses (990 to 1000 tokens)'] = correct_guesses
    # df1['correct guesses seq lengths (990 to 1000 tokens)'] = correct_guesses_lengths
    # df1['average first point of failure (990 to 1000 tokens)'] = avg_point_of_failure_short
    df1['correct guesses (2000 tokens)'] = correct_guesses
    df1['correct guesses seq lengths (2000 tokens)'] = correct_guesses_lengths
    df1['incorrect guesses (2000 tokens)'] = incorrect_guesses
    df1['incorrect guesses seq lengths (2000 tokens)'] = incorrect_guesses_lengths
    df1['average first point of failure (2000 tokens)'] = avg_point_of_failure_short
    df1['first point of failure for each incorrect sequence'] = incorrect_guesses_first_fail
    df1['max depth for correct sequences (2000 tokens)'] = max_depths_correct_guesses
    df1['timestep depths for correct sequences'] = timestep_depths_correct_guesses
    df1['max depth for incorrect sequences (2000 tokens)'] = max_depths_incorrect_guesses
    df1['timestep depths for incorrect sequences'] = timestep_depths_incorrect_guesses

    # df1['max depths']=max_depths
    # df1['timestep depths'] = timestep_depths
    # df1['correct guesses long (502 to 1000 tokens)']=correct_guesses_long
    # df1['correct guesses long seq lenghts (502 to 1000 tokens)']=correct_guesses_long_lengths
    # df1['avg point of failure long (502 to 1000 tokens)']=avg_point_of_failure_long

    writer = pd.ExcelWriter(excel_name_inference, engine='xlsxwriter')

    df1.to_excel(writer, index=False)
    writer.save()



def test_model(model, loader, dataset):
    """
    add a function here to calculate the average point where the model fails.
    if the model gets everything correct then it wont be counted in the values which fail at any point
    scatter plots in the main function after all models have been evaluated
    one scatter plot for long sequences, one for very long sequences

    """

    correct_guesses = []
    incorrect_guesses = []
    correct_guesses_length = []
    incorrect_guesses_length = []
    incorrect_guesses_first_fail = []
    sum_first_fail_points = 0


    max_depths_correct_guesses = []
    timestep_depths_correct_guesses = []
    max_depths_incorrect_guesses = []
    timestep_depths_incorrect_guesses = []

    model.eval()
    num_correct = 0
    # dataset = ''
    log_file=''
    # if len(X[0])>num_bracket_pairs*2:
    #     dataset = 'long'
    #     log_file =long_test_log
    # else:
    #     dataset='short'
    #     log_file = test_log
    if dataset=='short':
        # log_file=test_log
        ds = test_dataset
    # elif dataset=='long':
    #     log_file=long_test_log
    #     ds = long_dataset


    # with open(log_file,'a') as f:
    #     f.write('////////////////////////////////////////\n')
    #     f.write('TEST '+dataset+'\n')

    # for i in range(len(X)):
    #     input_seq = Dyck.lineToTensor(X[i])
    #     target_seq = Dyck.lineToTensorSigmoid(y[i])
    #     len_seq = len(input_seq)
    #     output_seq = torch.zeros(target_seq.shape)
    #
    #     input_seq.to(device)
    #     target_seq.to(device)
    #     output_seq.to(device)
    #
    #     # if model.model_name == 'VanillaLSTM':
    #     #     hidden = (torch.zeros(1, 1, model.hidden_size).to(device), torch.zeros(1, 1, model.hidden_size).to(device))
    #     # elif model.model_name == 'VanillaRNN' or model.model_name == 'VanillaGRU':
    #     #     hidden = torch.zeros(1, 1, model.hidden_size).to(device)
    #
    #     hidden = model.init_hidden()
    #
    #     for j in range(len_seq):
    #         # out, hidden = model(input_seq[j].to(device), hidden)
    #         out, hidden = model(Dyck.lineToTensor(X[i][j]).to(device), hidden)
    #         output_seq[j] = out
    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):

    # for i, (sentences, labels, input_seq, target_seq, length, max_depth, timestep_depth) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)
        # output_seq[i] = out

        # with open(log_file, 'a') as f:
        #     f.write('////////////////////////////////////////\n')
        #     # f.write('input batch = ' + str(ds[i * batch_size:i * batch_size + batch_size]['x']) + '\n')
        #     # f.write('encoded batch = ' + str(input_seq) + '\n')
        #     f.write('input batch = ' + str(sentences) + '\n')
        #     f.write('encoded batch = ' + str(input_seq) + '\n')

        output_seq = model.mask(output_seq, target_seq, length)

        # with open(log_file, 'a') as f:
        #     f.write('////////////////////////////////////////\n')
        #     f.write('input sentence = ' + ds[i]['x'] + '\n')
        #     f.write('encoded sentence = ' + str(input_seq) + '\n')

        # with open(log_file, 'a') as f:
        #     f.write('actual output in test function = ' + str(output_seq) + '\n')

        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)

        # out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
        # target_np = np.int_(target_seq.detach().cpu().numpy())

        out_seq = output_seq.clone().detach() >= epsilon
        out_seq = out_seq.float()


        # with open(log_file, 'a') as f:
        #     # f.write('rounded output in test function = ' + str(out_np) + '\n')
        #     # f.write('target in test function = ' + str(target_np) + '\n')
        #
        #     f.write('rounded output in test function = ' + str(out_seq) + '\n')
        #     f.write('target in test function = ' + str(target_seq) + '\n')

        for j in range(batch_size):

            # if out_np[j].all() == target_np[j].all():
            max_depth, timestep_depth = get_timestep_depths(sentences[j])
            if torch.equal(out_seq[j], target_seq[j]):
            # if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
                num_correct += 1
                correct_guesses.append(sentences[j])
                correct_guesses_length.append(length[j].item())
                # incorrect_guesses_first_fail.append(length[j].item())
                sum_first_fail_points+=length[j].item()
                max_depths_correct_guesses.append(max_depth)
                timestep_depths_correct_guesses.append(timestep_depth)

                # with open(log_file, 'a') as f:
                #     f.write('CORRECT' + '\n')
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

                # with open(log_file, 'a') as f:
                #     f.write('INCORRECT' + '\n')

        # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
        #     num_correct += 1
        #     with open(log_file, 'a') as f:
        #         f.write('CORRECT' + '\n')
        # else:
        #     with open(log_file, 'a') as f:
        #         f.write('INCORRECT' + '\n')


    accuracy = num_correct / len(ds) * 100
    # with open(log_file, 'a') as f:
    #     f.write('accuracy = ' + str(accuracy)+'%' + '\n')
    print(''+dataset+' test accuracy = '+ str(accuracy)+'%')
    # avg_first_fail_point = sum_first_fail_points/len(incorrect_guesses)
    avg_first_fail_point = sum_first_fail_points / (len(incorrect_guesses)+num_correct)

    # return accuracy, correct_guesses,correct_guesses_length, incorrect_guesses, incorrect_guesses_length, incorrect_guesses_first_fail,avg_first_fail_point, max_depth, timestep_depth
    return accuracy, correct_guesses,correct_guesses_length, incorrect_guesses, incorrect_guesses_length, incorrect_guesses_first_fail,avg_first_fail_point, max_depths_correct_guesses, timestep_depths_correct_guesses, max_depths_incorrect_guesses, timestep_depths_incorrect_guesses






if __name__=='__main__':
    main()



