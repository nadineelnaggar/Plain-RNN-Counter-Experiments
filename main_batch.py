import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt
from models_batch import VanillaLSTM, VanillaRNN, VanillaGRU
from Dyck_Generator_Suzgun_Batch import DyckLanguage
import random
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from Dyck1_Datasets import NextTokenPredictionLongTestDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionTrainDataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='input model name (VanillaLSTM, VanillaRNN, VanillaGRU)')
parser.add_argument('--task', type=str, help='NextTokenPrediction, BinaryClassification, TernaryClassification')
parser.add_argument('--feedback', type=str, help='EveryTimeStep, EndofSequence')
parser.add_argument('--hidden_size', type=int, help='hidden size')
parser.add_argument('--num_layers', type=int, help='number of layers', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--learning_rate', type=float, help='learning rate')
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
batch_size = args.batch_size
# load_model = args.load_model


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

train_size = 10000
test_size = 5000
long_size = 5000

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)



path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_"+str(task)+"/Minibatch_Training/"+model_name+"/"

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
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.txt'
excel_name = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '.xlsx'
modelname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_MODEL.pth'
optimname = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_OPTIMISER.pth'
train_log = path+ 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TRAIN_LOG.txt'
test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_TEST_LOG.txt'
long_test_log = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_LONG_TEST_LOG.txt'
plot_name = path+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_Feedback_' + feedback + '_' +str(batch_size) +'_batch_size_'+'_' + str(
        hidden_size) + 'hidden_units_' + use_optimiser + '_lr=' + str(learning_rate) + '_' + str(
        num_epochs) + 'epochs_'+ str(num_runs)+'runs' + '_PLOT.png'

with open(file_name, 'w') as f:
    f.write('\n')

with open(train_log, 'w') as f:
    f.write('\n')

with open(test_log, 'w') as f:
    f.write('\n')
with open(long_test_log, 'w') as f:
    f.write('\n')

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
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
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


    # seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels,lengths, batch_size=len(sentences))
    seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=batch_size)


    return seq_tensor.to(device), labels_tensor.to(device), lengths_tensor.to(device)



train_dataset = NextTokenPredictionTrainDataset()
test_dataset = NextTokenPredictionShortTestDataset()
long_dataset = NextTokenPredictionLongTestDataset()

train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
long_loader = DataLoader(long_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False)


def select_model(model_name, input_size, hidden_size, num_layers,batch_size, num_classes, output_activation):
    if model_name=='VanillaLSTM':
        model = VanillaLSTM(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaRNN':
        model = VanillaRNN(input_size, hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    elif model_name=='VanillaGRU':
        model = VanillaGRU(input_size,hidden_size, num_layers, batch_size, num_classes, output_activation=output_activation)
    return model.to(device)


# print(Dyck.lineToTensorSigmoid('1110'))
# model = select_model(model_name, input_size=n_letters, hidden_size=hidden_size, num_layers=num_layers,
#                      batch_size=batch_size, num_classes=n_letters, output_activation='Sigmoid')
#
# for i, (input_seq, target_seq, length) in enumerate(train_loader):
#     print('input_seq = ', input_seq)
#     print('target seq = ',target_seq)
#     print('lengths = ', length)
#     print('input seq shape = ', input_seq.shape)
#     print('target seq shape = ', target_seq.shape)
#     print('length shape = ', length.shape)
#     out = model(input_seq.to(device), length)
#     print(out)
#     print('out.shape = ',out.shape)
#     out = model.mask(out, target_seq, length)
#     print(out)
#     break






def main():
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

    output_activation = 'Sigmoid'

    if task == 'TernaryClassification':
        num_classes = 3
        output_activation = 'Softmax'
    elif task == 'BinaryClassification' or task == 'NextTokenPrediction':
        num_classes = 2
        output_activation = 'Sigmoid'




    input_size = n_letters







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
        seed = num_runs+i
        # seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)
        with open(train_log, 'a') as f:
            f.write('random seed for run '+str(i)+' = '+str(seed)+'\n')
        model = select_model(model_name, input_size, hidden_size, num_layers, batch_size, num_classes, output_activation='Sigmoid')
        # print(model.model_name)
        model.to(device)

        # log_dir="logs"
        log_dir = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_" + str(task) + "/Minibatch_Training/" + model_name + "/logs/run"+str(i)
        sum_writer = SummaryWriter(log_dir)


        runs.append('run'+str(i))
        print('****************************************************************************\n')
        train_accuracy, df = train(model, train_loader, sum_writer)
        train_accuracies.append(train_accuracy)
        train_dataframes.append(df)
        test_accuracy = test_model(model, test_loader, 'short')
        test_accuracies.append(test_accuracy)
        long_test_accuracy = test_model(model, long_loader, 'long')
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











def train(model, loader, sum_writer):





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

    # global_step=0

    print(model)
    print('num_train_samples = ',len(loader.dataset))

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


        for i, (input_seq, target_seq, length) in enumerate(loader):
            model.zero_grad()
            # output_seq = torch.zeros(target_seq.shape)
            output_seq = model(input_seq.to(device), length)
            # output_seq[i] = out
            # print('output seq = ',output_seq)
            # print('output seq shape = ',output_seq.shape)
            # print('target seq = ',target_seq)
            # print('target seq shape = ',target_seq.shape)
            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('////////////////////////////////////////\n')
                    f.write('input batch = ' + str(train_dataset[i*batch_size:i*batch_size+batch_size]['x']) + '\n')
                    f.write('encoded batch = '+str(input_seq)+'\n')

            # print(output_seq.shape)
            output_seq=model.mask(output_seq, target_seq, length)
            loss = criterion(output_seq, target_seq)
            # print('loss = ',loss)
            total_loss += loss.item()
            loss.backward()
            optimiser.step()

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('actual output in train function = ' + str(output_seq) + '\n')

            # print('output_seq.shape before reshape = ', output_seq.shape)
            # print('target_seq.shape before reshape = ', target_seq.shape)
            output_seq = output_seq.view(batch_size, length[0], n_letters)
            target_seq = target_seq.view(batch_size, length[0], n_letters)
            # print('output_seq.shape = ',output_seq.shape)
            # print('target_seq.shape = ',target_seq.shape)

            out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
            target_np = np.int_(target_seq.detach().cpu().numpy())
            # print('out_np.shape = ', out_np.shape)
            # print('target_np.shape = ', target_np.shape)

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('rounded output in train function = ' + str(out_np) + '\n')
                    f.write('target in train function = ' + str(target_np) + '\n')



            # print('out_np = ',out_np)
            # print('target_np = ',target_np)
            # print('flattened output np = ',out_np.flatten())
            # print('flattened target np = ', target_np.flatten())

            # count = 0
            for j in range(batch_size):
                # print('out_np[j] = ',out_np[j])
                # print('out_np[j].shape = ',out_np[j].shape)
                # print('target_np[j] = ',target_np[j])
                # print('target_np[j].shape = ',target_np[j].shape)

                if np.array_equal(out_np[j],target_np[j]):
                # if np.equal(out_np[j].all(), target_np[j].all()).all():
                # if out_np[j].all()==target_np[j].all():
                    # print('output_np[j] = target_np[j]')
                #     count+=1
                #     print('count correct = ',count)
                # print('np.all(np.equal(out_np[j], target_np[j])) = ',np.all(np.equal(out_np[j],target_np[j])))
                # if np.all(np.equal(out_np[j], target_np[j])):
                # if np.all(np.equal(out_np[j], target_np[j])):
                # if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
                    num_correct += 1
                    # epoch_correct_guesses.append(X[i])
                    epoch_correct_guesses.append(train_dataset[(i*batch_size)+j]['x'])
                    if print_flag == True:
                        with open(train_log, 'a') as f:
                            f.write('CORRECT' + '\n')
                else:
                    epoch_incorrect_guesses.append(train_dataset[(i*batch_size)+j]['x'])
                    if print_flag == True:
                        with open(train_log, 'a') as f:
                            f.write('INCORRECT' + '\n')
            # print('num_correct = ',num_correct)
            # break
        # break




        accuracy = num_correct/len(train_dataset)*100
        # print('\n')
        print('Accuracy for epoch ', epoch, '=', accuracy, '%, total loss for epoch ', epoch,' = ',total_loss,' num_correct = ',num_correct)
        # break
        accuracies.append(accuracy)
        losses.append(total_loss/len(train_dataset))
        sum_writer.add_scalar('epoch_losses', total_loss/len(train_dataset),global_step=epoch)
        sum_writer.add_scalar('accuracy', accuracy, global_step=epoch)
        # global_step+=1
        sum_writer.close()
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
        correct_arr.append(epoch_correct_guesses)
        if epoch == num_epochs - 1:
            # print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            print('num_correct = ',num_correct)
            print('Final training accuracy = ', num_correct / len(train_dataset) * 100, '%')
            # print('**************************************************************************\n')
    df1['epoch'] = epochs
    df1['accuracies'] = accuracies
    df1['Total epoch losses'] = losses
    df1['epoch correct guesses'] = correct_arr
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses

    sum_writer.add_hparams({'model_name':model.model_name,'dataset_size': len(train_dataset), 'num_epochs': num_epochs,
                            'learning_rate': learning_rate, 'batch_size':batch_size,
                            'optimiser': use_optimiser}, {'accuracy': accuracy, 'loss': total_loss/len(train_dataset)})
    # sum_writer.add_graph(model, (Dyck.lineToTensor(X[0][0]), model.init_hidden()))
    # sum_writer.add_graph(model, loader[0])
    # sum_writer.add_graph(model, input_seq, length)
    sum_writer.close()

    torch.save(model.state_dict(), modelname)
    torch.save(optimiser.state_dict(), optimname)

        # print(accuracies)
        # print(accuracy)
    return accuracy, df1

def test_model(model, loader, dataset):
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
        log_file=test_log
        ds = test_dataset
    elif dataset=='long':
        log_file=long_test_log
        ds = long_dataset


    with open(log_file,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')

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
    for i, (input_seq, target_seq, length) in enumerate(loader):
        output_seq = model(input_seq.to(device), length)
        # output_seq[i] = out

        with open(log_file, 'a') as f:
            f.write('////////////////////////////////////////\n')
            f.write('input batch = ' + str(ds[i * batch_size:i * batch_size + batch_size]['x']) + '\n')
            f.write('encoded batch = ' + str(input_seq) + '\n')

        output_seq = model.mask(output_seq, target_seq, length)

        with open(log_file, 'a') as f:
            f.write('////////////////////////////////////////\n')
            f.write('input sentence = ' + ds[i]['x'] + '\n')
            f.write('encoded sentence = ' + str(input_seq) + '\n')

        with open(log_file, 'a') as f:
            f.write('actual output in test function = ' + str(output_seq) + '\n')

        output_seq = output_seq.view(batch_size, length[0], n_letters)
        target_seq = target_seq.view(batch_size, length[0], n_letters)

        out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
        target_np = np.int_(target_seq.detach().cpu().numpy())



        with open(log_file, 'a') as f:
            f.write('rounded output in test function = ' + str(out_np) + '\n')
            f.write('target in test function = ' + str(target_np) + '\n')

        for j in range(batch_size):

            if np.array_equal(out_np[j],target_np[j]):
            # if out_np[j].all() == target_np[j].all():
            # if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
                num_correct += 1


                with open(log_file, 'a') as f:
                    f.write('CORRECT' + '\n')
            else:

                with open(log_file, 'a') as f:
                    f.write('INCORRECT' + '\n')

        # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
        #     num_correct += 1
        #     with open(log_file, 'a') as f:
        #         f.write('CORRECT' + '\n')
        # else:
        #     with open(log_file, 'a') as f:
        #         f.write('INCORRECT' + '\n')


    accuracy = num_correct / len(ds) * 100
    with open(log_file, 'a') as f:
        f.write('accuracy = ' + str(accuracy)+'%' + '\n')
    print(''+dataset+' test accuracy = '+ str(accuracy)+'%')


    return accuracy







if __name__=='__main__':
    main()



