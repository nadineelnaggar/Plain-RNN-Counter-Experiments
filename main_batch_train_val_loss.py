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
# parser.add_argument('--num_checkpoints', type=int,default=100, help='number of checkpoints we want to include if we dont need all of them (e.g., first 5 checkpoints only), stop after n checkpoints')
# parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or appended')
# parser.add_argument('--dataset_type',type=str, default='nested',help='nested, zigzag or concatenated')


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
# num_checkpoints = args.num_checkpoints
# dataset_type = args.dataset_type


# best_run = args.best_run
#
# if best_run==-1:
#     best_run = num_runs-1

checkpoint_step = int(num_epochs/4)
if args.checkpoint_step!=0:
    checkpoint_step = args.checkpoint_step

num_checkpoints= num_epochs*num_runs


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



NUM_PAR = 1
MIN_SIZE = 2
MAX_SIZE = 50
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



file_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs_INFERENCE' + '.txt'



excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '.xlsx'

modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL_'

optimname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_OPTIMISER.pth'

checkpoint = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_CHECKPOINT_'


train_validation_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_TRAINING_SET_VALIDATION_LOG.txt'

excel_name_new = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+str(lr_scheduler_step)+"lr_scheduler_step_"+str(lr_scheduler_gamma)+"lr_scheduler_gamma_"+ str(num_runs)+'runs' + '_NEW.xlsx'

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

    # num_sequences = len(lengths)
    # if len(lengths)<batch_size:
    #     for j in range(batch_size-num_sequences):
    #         lengths.append(0)

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

    # max_depths = []
    # timestep_depths = []
    # for i in range(len(batch)):
    #     max_depth, timestep_depth = get_timestep_depths(sentences[i])
    #     max_depths.append(max_depth)
    #     timestep_depths.append(timestep_depth)

    # max_depths_tensor = torch.tensor(max_depths,dtype=torch.float32)
    # timestep_depths_tensor = torch.tensor(timestep_depths,dtype=torch.float32)


    # return seq_tensor.to(device), labels_tensor.to(device), lengths_tensor.to(device)
    # return sentences, labels, seq_tensor.to(device), labels_tensor.to(device), lengths_tensor, max_depths, timestep_depths
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


# train_dataset = NextTokenPredictionTrainDataset()
# test_dataset = NextTokenPredictionDataset102to500tokens()
# long_dataset = NextTokenPredictionDataset502to1000tokens()

# # test_dataset = NextTokenPredictionDataset990to1000tokens()
# # test_dataset = NextTokenPredictionDataset2000tokens()
# test_dataset = NextTokenPredictionDataset2000tokens_nested()
# test_size = len(test_dataset)

train_dataset=NextTokenPredictionTrainDataset()
train_size = len(train_dataset)


train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



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




def validate_model(model, loader, dataset, run, epoch):
    # model.eval()
    num_correct = 0
    # dataset = ''
    log_file=''


    # if loader==validation_loader:
    #     # log_file = validation_log
    #     dataset='Validation Set'
    #     # ds = validation_dataset
    # elif loader==train_loader:
    #     # log_file = train_validation_log
    #     dataset = 'Train Set'
    #     ds = train_dataset

    dataset = 'Train Set'
    ds = train_dataset
    log_file = train_validation_log
    criterion = nn.MSELoss()

    total_loss = 0

    for i, (sentences, labels, input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)
        # output_seq[i] = out

        # with open(log_file, 'a') as f:
        #     f.write('////////////////////////////////////////\n')
        #     f.write('input batch = ' + str(ds[i * batch_size:i * batch_size + batch_size]['x']) + '\n')
        #     f.write('encoded batch = ' + str(input_seq) + '\n')

        output_seq = model.mask(output_seq, target_seq, length)
        loss = criterion(output_seq,target_seq)
        total_loss+=loss.item()

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
        #     f.write('rounded output in test function = ' + str(out_np) + '\n')
        #     f.write('target in test function = ' + str(target_np) + '\n')

        for j in range(batch_size):

            # if np.array_equal(out_np[j],target_np[j]):
            if torch.equal(out_seq[j], target_seq[j]):
            # if out_np[j].all() == target_np[j].all():
            # if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
                num_correct += 1


            #     with open(log_file, 'a') as f:
            #         f.write('CORRECT' + '\n')
            # else:
            #
            #     with open(log_file, 'a') as f:
            #         f.write('INCORRECT' + '\n')

        # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
        #     num_correct += 1
        #     with open(log_file, 'a') as f:
        #         f.write('CORRECT' + '\n')
        # else:
        #     with open(log_file, 'a') as f:
        #         f.write('INCORRECT' + '\n')


    accuracy = num_correct / len(ds) * 100
    with open(log_file, 'a') as f:
    #     if loader==validation_loader:
    #
    #         f.write('val accuracy for run'+str(run)+' epoch '+str(epoch)+' = ' + str(accuracy)+'%, val loss = '+str(loss.item()/len(ds)) + '\n')
    # # print(''+dataset+' accuracy = '+ str(accuracy)+'% '+ 'avg loss = '+str(loss.item()/len(ds)))
    #     elif loader==train_loader:
    #         f.write('train val accuracy for run' + str(run) + ' epoch ' + str(epoch) + ' = ' + str(
    #             accuracy) + '%, train val loss = ' + str(loss.item() / len(ds)) + '\n')
        f.write('train val accuracy for run' + str(run) + ' epoch ' + str(epoch) + ' = ' + str(
            accuracy) + '%, train val loss = ' + str(loss.item() / len(ds)) + '\n')


    return accuracy, loss.item()/len(ds)


def main():




    output_activation = 'Sigmoid'

    if task == 'TernaryClassification':
        num_classes = 3
        output_activation = 'Softmax'
    elif task == 'BinaryClassification' or task == 'NextTokenPrediction':
        num_classes = 2
        output_activation = 'Sigmoid'




    input_size = n_letters








    dfs_read = read_sheets()
    # train_accuracies = []
    test_accuracies = []
    # long_test_accuracies = []
    # train_dataframes = []
    run = []




    # avg_train_losses = []
    # avg_val_losses = []
    # avg_long_val_losses = []
    epochs = []


    train_dataframes = []
    runs = []

    dfs = []





    for run in range(num_runs):
        df = dfs_read[run]
        losses_train = df['Average training losses']
        losses_train = losses_train.tolist()
        losses_val = df['Average validation losses']
        losses_val=losses_val.tolist()
        losses_long_val = df['Average long validation losses']
        losses_long_val = losses_long_val.tolist()

        train_val_accuracies = []
        train_val_losses = []



        # runs.append(run)
        checkpoint_count = 0
        runs.append('run'+str(run))
        for epoch in range(num_epochs):
            if epoch%checkpoint_step==0 and checkpoint_count<=num_checkpoints:
                checkpoint_count+=1
                # runs.append(run)
                # avg_train_losses.append(losses_train[epoch])
                # avg_val_losses.append(losses_val[epoch])
                # avg_long_val_losses.append(losses_long_val[epoch])
                # epochs.append(epoch)
                checkpoint_model = select_model(model_name,input_size,hidden_size,num_layers,batch_size,num_classes,output_activation)
                # checkpoint_model.to(device)
                checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"

                checkpt = torch.load(checkpoint_path)
                checkpoint_model.load_state_dict(checkpt['model_state_dict'])
                checkpoint_model.to(device)
                # checkpoint_test_accuracy, checkpoint_correct_guesses,checkpoint_correct_guesses_length, checkpoint_incorrect_guesses, checkpoint_incorrect_guesses_length, checkpoint_incorrect_guesses_first_fail,checkpoint_avg_first_fail_point, checkpoint_max_depth, checkpoint_timestep_depth = test_model(checkpoint_model,test_loader,'short')
                # checkpoint_test_accuracy, checkpoint_correct_guesses, checkpoint_correct_guesses_length, checkpoint_incorrect_guesses, checkpoint_incorrect_guesses_length, checkpoint_incorrect_guesses_first_fail, checkpoint_avg_first_fail_point, checkpoint_max_depth_correct, checkpoint_timestep_depth_correct, checkpoint_max_depth_incorrect, checkpoint_timestep_depth_incorrect = test_model(checkpoint_model, test_loader, 'short')
                train_val_acc_checkpoint, train_val_loss_checkpoint = validate_model(checkpoint_model, train_loader, train_dataset, run, epoch)
                # test_accuracies.append(checkpoint_test_accuracy)
                train_val_accuracies.append(train_val_acc_checkpoint)
                train_val_losses.append(train_val_loss_checkpoint)
                print('run ',run, ' epoch ',epoch,' train val loss = ',train_val_loss_checkpoint, 'train val acc = ', train_val_acc_checkpoint)








                # checkpoint_long_accuracy, checkpoint_long_correct_guesses,checkpoint_long_correct_guesses_length, checkpoint_long_incorrect_guesses, checkpoint_long_incorrect_guesses_length, checkpoint_long_incorrect_guesses_first_fail,checkpoint_long_avg_first_fail_point = test_model(checkpoint_model,long_loader,'long')
                # long_test_accuracies.append(checkpoint_long_accuracy)
                # correct_guesses_long.append(checkpoint_long_correct_guesses)
                # correct_guesses_long_lengths.append(checkpoint_long_correct_guesses_length)
                # incorrect_guesses_long.append(checkpoint_long_incorrect_guesses)
                # incorrect_guesses_long_lengths.append(checkpoint_long_incorrect_guesses_length)
                # incorrect_guesses_long_first_fail.append(checkpoint_long_incorrect_guesses_first_fail)
                # avg_point_of_failure_long.append(checkpoint_long_avg_first_fail_point)

        df1 = pd.DataFrame()
        df1['epoch'] = df['epoch']
        df1['Training accuracies'] = df['Training accuracies']
        df1['Average training losses'] = df['Average training losses']
        df1['Train validation accuracies'] = train_val_accuracies
        df1['Average train validation losses'] = train_val_losses
        df1['Average validation losses'] = df['Average validation losses']
        df1['Validation accuracies'] = df['Validation accuracies']
        df1['Average long validation losses'] = df['Average long validation losses']
        df1['Long validation accuracies'] = df['Long validation accuracies']
        df1['learning rates'] = df['learning rates']
        df1['epoch correct guesses'] = df['epoch correct guesses']
        df1['epoch incorrect guesses'] = df['epoch incorrect guesses']
        df1['epoch error indices'] = df['epoch error indices']
        df1['epoch error seq lengths'] = df['epoch error seq lengths']

        dfs.append(df1)

    dfs_write = dict(zip(runs, dfs))
    writer = pd.ExcelWriter(excel_name_new, engine='xlsxwriter')

    for sheet_name in dfs_write.keys():
        dfs_write[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()














if __name__=='__main__':
    main()



