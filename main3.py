import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models import VanillaRNN, VanillaLSTM, VanillaGRU
from Dyck_Generator_Suzgun import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionTrainDataset

import tensorflow as tf
from tensorflow import summary

import xlsxwriter

# print(torch.__version__)


# # log_dir="logs"
# log_dir = "/content/drive/MyDrive/PhD/EXPT_LOGS"
# sum_writer = SummaryWriter(log_dir)

# SUZUGUN EXPERIMENT RUN HERE

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('device = ',device)




NUM_PAR = 1
MIN_SIZE = 2
MAX_SIZE = 50
P_VAL = 0.5
Q_VAL = 0.25


epsilon=0.5

train_size = 10000
test_size = 5000
long_size = 5000

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification')
parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
parser.add_argument('--hidden_size', type=int, help='hidden size')
parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--num_epochs', type=int, help='number of training epochs')
parser.add_argument('--num_runs', type=int, help='number of training runs')
parser.add_argument('--batch_size', type=int, help='batch size, if 0 then shuffle between batch sizes and learning rates, if 1 then online training')
# parser.add_argument('load_model', type=int, help='load previous model (1), train model from scratch (0)', default=0)


args = parser.parse_args()

model_name = args.model_name
task = args.task
feedback = args.feedback
hidden_size = args.hidden_size
num_layers = args.num_layers
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_runs = args.num_runs
batch_size=args.batch_size
# load_model = args.load_model

use_optimiser='Adam'

num_bracket_pairs = 25

length_bracket_pairs = 50


# # log_dir="logs"
# log_dir = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/"+model_name+"/logs"
# sum_writer = SummaryWriter(log_dir)

path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/"+model_name+"/"

print('model_name = ',model_name)
print('task = ',task)
print('feedback = ',feedback)
print('hidden_size = ',hidden_size)
print('num_layers = ',num_layers)
print('learning_rate = ',learning_rate)
print('num_epochs = ',num_epochs)
print('num_runs = ',num_runs)
# print('load_model = ',load_model)

file_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.txt'
excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.xlsx'
modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL.pth'
optimname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_OPTIMISER.pth'
train_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TRAIN_LOG.txt'
test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TEST_LOG.txt'
long_test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_LONG_TEST_LOG.txt'
plot_name = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_PLOT.png'


def read_dataset(task):

    X = []
    y = []
    data = []

    X_long = []
    y_long = []
    data_long = []

    if task == 'NextTokenPrediction':
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                X.append(sentence)
                y.append(label)
        with open('Dyck1_Dataset_Suzgun_test_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                X_long.append(sentence)
                y_long.append(label)

    # print('len X long == ', len(X_long))
    # print('len y long == ', len(y_long))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:train_size+test_size]
    y_test = y[train_size:train_size+test_size]
    X_long = X_long[:long_size]
    y_long=y_long[:long_size]

    return X_train, y_train, X_test, y_test, X_long, y_long



def select_model(model_name, input_size, hidden_size, num_layers, num_classes, output_activation):
    if model_name=='VanillaLSTM':
        model = VanillaLSTM(input_size,hidden_size, num_layers, num_classes, output_activation=output_activation)
    elif model_name=='VanillaRNN':
        model = VanillaRNN(input_size, hidden_size, num_layers, num_classes, output_activation=output_activation)
    elif model_name=='VanillaGRU':
        model = VanillaGRU(input_size,hidden_size, num_layers, num_classes, output_activation=output_activation)
    return model.to(device)


def main():


    output_activation = 'Sigmoid'

    if task == 'TernaryClassification':
        num_classes = 3
        output_activation = 'Softmax'
    elif task == 'BinaryClassification' or task == 'NextTokenPrediction':
        num_classes = 2
        output_activation = 'Sigmoid'



    vocab = ['(', ')']
    n_letters = len(vocab)
    input_size = n_letters

    # X_train, y_train, X_test, y_test, X_long, y_long = read_dataset('NextTokenPrediction')
    X_train, y_train = NextTokenPredictionTrainDataset()
    X_test, y_test = NextTokenPredictionShortTestDataset()
    X_long, y_long = NextTokenPredictionLongTestDataset()





    with open(file_name, 'a') as f:
        f.write('Output activation = ' + output_activation + '\n')
        f.write('Optimiser used = ' + use_optimiser + '\n')
        f.write('Learning rate = ' + str(learning_rate) + '\n')
        f.write('Number of runs = ' + str(num_runs) + '\n')
        f.write('Number of epochs in each run = ' + str(num_epochs) + '\n')
        f.write('Saved model name = ' + modelname + '\n')
        f.write('Saved optimiser name = ' + optimname + '\n')
        f.write('Excel name = ' + excel_name + '\n')
        f.write('Train log name = ' + train_log + '\n')
        f.write('Test log name = ' + test_log + '\n')
        f.write('Long test log name = ' + long_test_log + '\n')
        f.write('///////////////////////////////////////////////////////////////\n')
        f.write('\n')

    train_accuracies = []
    test_accuracies = []
    long_test_accuracies = []
    train_dataframes = []
    runs = []
    for i in range(num_runs):
        model = select_model(model_name, input_size, hidden_size, num_layers, num_classes, output_activation='Sigmoid')
        # print(model.model_name)
        model.to(device)

        # log_dir="logs"
        log_dir = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_" + str(task) + "/" + model_name + "/logs/run"+str(i)
        sum_writer = SummaryWriter(log_dir)


        runs.append('run'+str(i))
        print('****************************************************************************\n')
        train_accuracy, df = train(model, X_train, y_train, sum_writer)
        train_accuracies.append(train_accuracy)
        train_dataframes.append(df)
        test_accuracy = test_model(model, X_test, y_test)
        test_accuracies.append(test_accuracy)
        long_test_accuracy = test_model(model, X_long, y_long)
        long_test_accuracies.append(long_test_accuracy)

        with open(file_name, "a") as f:
            f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
            f.write('test accuracy for run ' + str(i) + ' = ' + str(test_accuracy) + '%\n')
            f.write('long test accuracy for run '+str(i)+' = '+str(long_test_accuracy)+'%\n')

    dfs = dict(zip(runs, train_dataframes))
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    for sheet_name in dfs.keys():
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()

    max_train_accuracy = max(train_accuracies)
    min_train_accuracy = min(train_accuracies)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    std_train_accuracy = np.std(train_accuracies)

    max_test_accuracy = max(test_accuracies)
    min_test_accuracy = min(test_accuracies)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)

    max_long_test_accuracy = max(long_test_accuracies)
    min_long_test_accuracy = min(long_test_accuracies)
    avg_long_test_accuracy = sum(long_test_accuracies) / len(test_accuracies)
    std_long_test_accuracy = np.std(long_test_accuracies)

    with open(file_name, "a") as f:
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum train accuracy = ' + str(max_train_accuracy) + '%\n')
        f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
        f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
        f.write('Standard Deviation for train accuracy = ' + str(std_train_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test accuracy = ' + str(max_test_accuracy) + '%\n')
        f.write('Minimum test accuracy = ' + str(min_test_accuracy) + '%\n')
        f.write('Average test accuracy = ' + str(avg_test_accuracy) + '%\n')
        f.write('Standard Deviation for test accuracy = ' + str(std_test_accuracy) + '\n')

        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum long test accuracy = ' + str(max_long_test_accuracy) + '%\n')
        f.write('Minimum long test accuracy = ' + str(min_long_test_accuracy) + '%\n')
        f.write('Average long test accuracy = ' + str(avg_long_test_accuracy) + '%\n')
        f.write('Standard Deviation for long test accuracy = ' + str(std_long_test_accuracy) + '\n')








def train(model, X, y, sum_writer):


    criterion = nn.MSELoss()
    # learning_rate = args.learning_rate
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    optimiser.zero_grad()
    losses = []
    correct_arr = []
    accuracies = []
    epochs = []
    all_epoch_incorrect_guesses = []
    df1 = pd.DataFrame()
    print_flag = False


    print(model)
    num_timesteps = 0

    for elem in X:
        num_timesteps+=len(elem)
    print('num_timesteps = ',num_timesteps)



    for epoch in range(num_epochs):
        num_correct = 0
        num_correct_timesteps = 0
        total_loss = 0
        epoch_incorrect_guesses = []
        epoch_correct_guesses = []
        epochs.append(epoch)
        if epoch==num_epochs-1:
            print_flag=True
        if print_flag == True:
            with open(train_log, 'a') as f:
                f.write('\nEPOCH ' + str(epoch) + '\n')
        for i in range(len(X)):
            model.zero_grad()
            input_seq = Dyck.lineToTensor(X[i])
            target_seq = Dyck.lineToTensorSigmoid(y[i])
            len_seq = len(input_seq)
            # len_seq = len(X[i])
            # print(Dyck.lineToTensor(X[i][0]).shape)
            output_seq = torch.zeros(target_seq.shape)

            input_seq.to(device)
            target_seq.to(device)
            output_seq.to(device)



            hidden = model.init_hidden()


            for j in range(len_seq):
                # sum_writer.add_graph(model, (Dyck.lineToTensor(X[i][j]),hidden))
                # sum_writer.close()
                # out, hidden = model(input_seq[j].to(device), hidden)
                # out, hidden = model(Dyck.lineToTensor(X[i][j]).to(device), hidden)

                # inp = Dyck.lineToTensor(X[i][0][j])
                # print(inp.shape)
                # out, hidden = model(inp.to(device), hidden)
                out, hidden = model(Dyck.lineToTensor(X[i][j]).to(device), hidden)

                output_seq[j]=out

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('////////////////////////////////////////\n')
                    f.write('input sentence = ' + str(X[i]) + '\n')
                    f.write('encoded sentence = '+str(input_seq)+'\n')

            loss = criterion(output_seq, target_seq)
            total_loss += loss.item()
            loss.backward()
            optimiser.step()

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('actual output in train function = ' + str(output_seq) + '\n')

            out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
            target_np = np.int_(target_seq.detach().cpu().numpy())

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('rounded output in train function = ' + str(out_np) + '\n')
                    f.write('target in train function = ' + str(target_np) + '\n')



            if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
                num_correct += 1
                # correct_arr.append(X[i])
                epoch_correct_guesses.append(X[i])
                if print_flag == True:
                    with open(train_log, 'a') as f:
                        f.write('CORRECT' + '\n')
            else:
                epoch_incorrect_guesses.append(X[i])
                if print_flag == True:
                    with open(train_log, 'a') as f:
                        f.write('INCORRECT' + '\n')



        accuracy = num_correct/len(X)*100
        print('Accuracy for epoch ', epoch, '=', accuracy, '%')
        accuracies.append(accuracy)
        losses.append(total_loss/len(X))
        sum_writer.add_scalar('epoch_losses', total_loss/len(X),global_step=epoch)
        sum_writer.add_scalar('accuracy', accuracy, global_step=epoch)
        sum_writer.close()
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
        correct_arr.append(epoch_correct_guesses)
        if epoch == num_epochs - 1:
            # print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            print('num_correct = ',num_correct)
            print('Final training accuracy = ', num_correct / len(X) * 100, '%')
            # print('**************************************************************************\n')
    df1['epoch'] = epochs
    df1['accuracies'] = accuracies
    df1['Total epoch losses'] = losses
    df1['epoch correct guesses'] = correct_arr
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses

    sum_writer.add_hparams({'model_name':model.model_name,'dataset_size': len(X), 'num_epochs': num_epochs, 'learning_rate': learning_rate,
                            'optimiser': use_optimiser}, {'accuracy': accuracy, 'loss': total_loss/len(X)})
    sum_writer.add_graph(model, (Dyck.lineToTensor(X[0][0]), model.init_hidden()))
    sum_writer.close()

    torch.save(model.state_dict(), modelname)
    torch.save(optimiser.state_dict(), optimname)

        # print(accuracies)
        # print(accuracy)
    return accuracy, df1

def test_model(model, X, y):
    model.eval()
    num_correct = 0
    dataset = ''
    log_file=''
    if len(X[0])>num_bracket_pairs*2:
        dataset = 'long'
        log_file =long_test_log
    else:
        dataset='short'
        log_file = test_log

    with open(log_file,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')

    for i in range(len(X)):
        input_seq = Dyck.lineToTensor(X[i])
        target_seq = Dyck.lineToTensorSigmoid(y[i])
        len_seq = len(input_seq)
        output_seq = torch.zeros(target_seq.shape)

        input_seq.to(device)
        target_seq.to(device)
        output_seq.to(device)



        hidden = model.init_hidden()

        for j in range(len_seq):
            # out, hidden = model(input_seq[j].to(device), hidden)
            out, hidden = model(Dyck.lineToTensor(X[i][j]).to(device), hidden)
            output_seq[j] = out

        with open(log_file, 'a') as f:
            f.write('////////////////////////////////////////\n')
            f.write('input sentence = ' + X[i] + '\n')
            f.write('encoded sentence = ' + str(input_seq) + '\n')

        with open(log_file, 'a') as f:
            f.write('actual output in test function = ' + str(output_seq) + '\n')

        out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
        target_np = np.int_(target_seq.detach().cpu().numpy())

        with open(log_file, 'a') as f:
            f.write('rounded output in test function = ' + str(out_np) + '\n')
            f.write('target in test function = ' + str(target_np) + '\n')

        if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
            num_correct += 1
            with open(log_file, 'a') as f:
                f.write('CORRECT' + '\n')
        else:
            with open(log_file, 'a') as f:
                f.write('INCORRECT' + '\n')


    accuracy = num_correct / len(X) * 100
    with open(log_file, 'a') as f:
        f.write('accuracy = ' + str(accuracy)+'%' + '\n')
    print(''+dataset+' test accuracy = '+ str(accuracy)+'%')


    return accuracy







if __name__=='__main__':
    main()
