import pandas as pd
import re

prefix = '/Users/nadineelnaggar/Desktop/SemiDyck1Logs/'

path_semi_dyck_1_BRI = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_MRI = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG'
path_semi_dyck_1_BRI_colab = ''
path_semi_dyck_1_MRI_colab = ''
path_semi_dyck_1_BCIB = ''
path_semi_dyck_1_BCIN = ''
path_semi_dyck_1_MCIB = ''
path_semi_dyck_1_MCIN = ''


epochs = []
train_accs = []
train_losses = []
train_val_accs = []
train_val_losses = []
long_val_accs = []
long_val_losses = []


lines = []

with open(path_semi_dyck_1_BRI, 'r') as f:
    for line in f:
        if line.startswith('Accuracy for epoch'):
            lines.append(line)

# print(lines[0])


for i in range(len(lines)):
    line = lines[i].split(',')
    print(line)
    print(line[0])
    epoch = line[0].split('epoch ')[1]
    avg_train_acc = epoch.split('=')[1]
    epoch = epoch.split('=')[0]
    avg_train_acc=avg_train_acc.split('%')[0]
    print(epoch)
    print(avg_train_acc)
    avg_train_loss = line[1].split('= ')[1].split(' ')[0]

    print(avg_train_loss)
    break


