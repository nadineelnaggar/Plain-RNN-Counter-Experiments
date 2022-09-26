import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def read_sheets(num_runs, excel_name):
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run"+str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(excel_name,sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs



lstm_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
gru_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaGRU/1_batch_size/0.001_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaGRU_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.001_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs.xlsx'

lstm_excel_path_extra_epochs = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_10extra_epochs.xlsx'
gru_excel_path_extra_epochs = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaGRU/1_batch_size/0.001_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaGRU_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.001_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_10extra_epochs.xlsx'

relu_excel_path = ''



lstm_dfs = read_sheets(10,lstm_excel_path)
lstm_dfs_extra_epochs = read_sheets(10, lstm_excel_path_extra_epochs)
print(len(lstm_dfs[0]))

print('*************')
print('LSTM')

lstm_best_val_loss_per_run = []
lstm_best_corresponding_val_acc_per_run = []
lstm_best_epoch_per_run = []
lstm_run = []

lstm_best_model_train_val_accuracies = []
lstm_best_model_long_val_accuracies = []
lstm_best_model_val_accuracies = []

lstm_val_losses_10_epochs = []
lstm_val_losses_15_epochs = []
lstm_val_losses_20_epochs = []
lstm_val_losses_25_epochs = []
lstm_val_losses_30_epochs = []



for i in range(len(lstm_dfs)):
    lstm_val_losses = []
    lstm_train_val_losses = []
    lstm_long_val_losses = []

    lstm_run.append(i)
    val_losses = lstm_dfs[i]['Average validation losses']
    lstm_val_losses_10_epochs.append(float(val_losses[9]))
    lstm_val_losses_15_epochs.append(float(val_losses[14]))
    lstm_val_losses_20_epochs.append(float(val_losses[19]))
    val_losses_extra = lstm_dfs_extra_epochs[i]['Average validation losses']
    lstm_val_losses_25_epochs.append(float(val_losses_extra[4]))
    lstm_val_losses_30_epochs.append(float(val_losses_extra[9]))


    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = lstm_dfs[i][lstm_dfs[i]['Average validation losses'] ==min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    lstm_best_epoch_per_run.append(int(row['epoch']))
    lstm_best_val_loss_per_run.append(float(row['Average validation losses']))
    lstm_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')
    plt.subplots()
    plt.plot(lstm_dfs[i]['epoch'][2:],np.log(lstm_dfs[i]['Average validation losses'][2:]), label='Validation loss')
    plt.plot(lstm_dfs[i]['epoch'][2:],np.log(lstm_dfs[i]['Average train validation losses'][2:]), label='Train loss')
    plt.plot(lstm_dfs[i]['epoch'][2:], np.log(lstm_dfs[i]['Average long validation losses'][2:]), label='Long validation loss')
    plt.legend()
    plt.savefig('lstm_loss_plots_20_epochs_run'+str(i)+'.png')
    plt.close()

    lstm_best_model_val_accuracies.append(float(row['Validation accuracies']))
    lstm_best_model_train_val_accuracies.append(float(row['Train validation accuracies']))
    lstm_best_model_long_val_accuracies.append(float(row['Long validation accuracies']))





average_lstm_best_epoch = 0
std_lstm_best_epoch = 0
epoch_starting_1 = []
for epoch in lstm_best_epoch_per_run:
    # print(epoch)
    # epoch+=1
    # print(epoch)
    # print("***************")
    epoch_starting_1.append(epoch+1)
    average_lstm_best_epoch+=(epoch+1)
print('lstm best epoch per run (starting from 0) = ',lstm_best_epoch_per_run)
print('lstm best epoch per run (starting from 1) = ',epoch_starting_1)
average_lstm_best_epoch=average_lstm_best_epoch/len(lstm_best_epoch_per_run)
print('lstm average best epoch across 10 runs of 20 epochs = ',average_lstm_best_epoch,' (assuming epoch starts at 1)')
std_lstm_best_epoch=np.std(epoch_starting_1)
print('lstm standard deviation of best epoch across 10 runs of 20 epochs = ',std_lstm_best_epoch, '(assuming epoch starts at 1)')
average_lstm_best_val_loss = np.mean(lstm_best_val_loss_per_run)
print('lstm average lowest val loss across 10 runs of 20 epochs = ',average_lstm_best_val_loss)
std_lstm_best_val_loss = np.std(lstm_best_val_loss_per_run)
print('lstm standard deviation of lowest val loss across 10 runs of 20 epochs = ',std_lstm_best_val_loss)
average_lstm_best_val_accuracy = np.mean(lstm_best_corresponding_val_acc_per_run)
print('lstm average corresponding val accuracy across 10 runs of 20 epochs = ',average_lstm_best_val_accuracy)
std_lstm_best_val_accuracy = np.std(lstm_best_corresponding_val_acc_per_run)
print('lstm standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ',std_lstm_best_val_accuracy)

print('lstm average val loss at epoch 10 (assuming starting from 1) = ',np.mean(lstm_val_losses_10_epochs))
print('lstm average val loss at epoch 15 (assuming starting from 1) = ',np.mean(lstm_val_losses_15_epochs))
print('lstm average val loss at epoch 20 (assuming starting from 1) = ',np.mean(lstm_val_losses_20_epochs))
print('lstm average val loss at epoch 25 (assuming starting from 1) = ',np.mean(lstm_val_losses_25_epochs))
print('lstm average val loss at epoch 30 (assuming starting from 1) = ',np.mean(lstm_val_losses_30_epochs))

print('Train Accuracy = (avg, min, max)', np.mean(lstm_best_model_train_val_accuracies), np.min(lstm_best_model_train_val_accuracies), np.max(lstm_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(lstm_best_model_val_accuracies), np.min(lstm_best_model_val_accuracies), np.max(lstm_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(lstm_best_model_long_val_accuracies), np.min(lstm_best_model_long_val_accuracies), np.max(lstm_best_model_long_val_accuracies))


print('************************************')

print('GRU')

gru_dfs = read_sheets(10,gru_excel_path)
print(len(gru_dfs[0]))

print('*************')
print('gru')

gru_best_val_loss_per_run = []
gru_best_corresponding_val_acc_per_run = []
gru_best_epoch_per_run = []
gru_run = []


gru_val_losses_10_epochs = []
gru_val_losses_15_epochs = []
gru_val_losses_20_epochs = []


gru_best_model_train_val_accuracies = []
gru_best_model_long_val_accuracies = []
gru_best_model_val_accuracies = []

for i in range(len(gru_dfs)):
    gru_run.append(i)
    val_losses = gru_dfs[i]['Average validation losses']
    gru_val_losses_10_epochs.append(float(val_losses[9]))
    gru_val_losses_15_epochs.append(float(val_losses[14]))
    gru_val_losses_20_epochs.append(float(val_losses[19]))
    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = gru_dfs[i][gru_dfs[i]['Average validation losses'] ==min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    gru_best_epoch_per_run.append(int(row['epoch']))
    gru_best_val_loss_per_run.append(float(row['Average validation losses']))
    gru_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')

    gru_val_losses = []
    gru_train_val_losses = []
    gru_long_val_losses = []
    plt.subplots()
    plt.plot(gru_dfs[i]['epoch'], gru_dfs[i]['Average validation losses'], label='Validation loss')
    plt.plot(gru_dfs[i]['epoch'], gru_dfs[i]['Average train validation losses'], label='Train loss')
    plt.plot(gru_dfs[i]['epoch'], gru_dfs[i]['Average long validation losses'], label='Long validation loss')
    plt.legend()
    plt.savefig('gru_loss_plots_20_epochs_run' + str(i) + '.png')
    plt.close()

    gru_best_model_val_accuracies.append(float(row['Validation accuracies']))
    gru_best_model_train_val_accuracies.append(float(row['Train validation accuracies']))
    gru_best_model_long_val_accuracies.append(float(row['Long validation accuracies']))

average_gru_best_epoch = 0
std_gru_best_epoch = 0
epoch_starting_1 = []
for epoch in gru_best_epoch_per_run:
    # print(epoch)
    # epoch+=1
    # print(epoch)
    # print("***************")
    epoch_starting_1.append(epoch+1)
    average_gru_best_epoch+=(epoch+1)
print('gru best epoch per run (starting from 0) = ',gru_best_epoch_per_run)
print('gru best epoch per run (starting from 1) = ',epoch_starting_1)
average_gru_best_epoch=average_gru_best_epoch/len(gru_best_epoch_per_run)
print('gru average best epoch across 10 runs of 20 epochs = ',average_gru_best_epoch,' (assuming epoch starts at 1)')
std_gru_best_epoch=np.std(epoch_starting_1)
print('gru standard deviation of best epoch across 10 runs of 20 epochs = ',std_gru_best_epoch, '(assuming epoch starts at 1)')
average_gru_best_val_loss = np.mean(gru_best_val_loss_per_run)
print('gru average lowest val loss across 10 runs of 20 epochs = ',average_gru_best_val_loss)
std_gru_best_val_loss = np.std(gru_best_val_loss_per_run)
print('gru standard deviation of lowest val loss across 10 runs of 20 epochs = ',std_gru_best_val_loss)
average_gru_best_val_accuracy = np.mean(gru_best_corresponding_val_acc_per_run)
print('gru average corresponding val accuracy across 10 runs of 20 epochs = ',average_gru_best_val_accuracy)
std_gru_best_val_accuracy = np.std(gru_best_corresponding_val_acc_per_run)
print('gru standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ',std_gru_best_val_accuracy)
print('gru average val loss at epoch 10 (assuming starting from 1) = ',np.mean(gru_val_losses_10_epochs))
print('gru average val loss at epoch 15 (assuming starting from 1) = ',np.mean(gru_val_losses_15_epochs))
print('gru average val loss at epoch 20 (assuming starting from 1) = ',np.mean(gru_val_losses_20_epochs))
print('Train Accuracy = (avg, min, max)', np.mean(gru_best_model_train_val_accuracies), np.min(gru_best_model_train_val_accuracies), np.max(gru_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(gru_best_model_val_accuracies), np.min(gru_best_model_val_accuracies), np.max(gru_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(gru_best_model_long_val_accuracies), np.min(gru_best_model_long_val_accuracies), np.max(gru_best_model_long_val_accuracies))



