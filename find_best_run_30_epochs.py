import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


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


lstm_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
gru_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaGRU/1_batch_size/0.001_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaGRU_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.001_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs.xlsx'

lstm_excel_path_extra_epochs = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_10extra_epochs.xlsx'
gru_excel_path_extra_epochs = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaGRU/1_batch_size/0.001_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaGRU_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.001_20epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_10extra_epochs.xlsx'

relu_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
relu_excel_path_2 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
relu_excel_path_3 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs.xlsx'

lstm_dfs = read_sheets(10, lstm_excel_path)
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

    all_lstm_val_losses = []
    all_lstm_val_losses = list(np.log(lstm_dfs[i]['Average validation losses'][2:])) + list(
        np.log(lstm_dfs_extra_epochs[i]['Average validation losses']))
    all_lstm_epochs = list(lstm_dfs[i]['epoch'][2:]) + list(lstm_dfs_extra_epochs[i]['epoch'])
    all_lstm_train_losses = []
    all_lstm_train_losses = list(np.log(lstm_dfs[i]['Average train validation losses'][2:])) + list(
        np.log(lstm_dfs_extra_epochs[i]['Average train validation losses']))
    all_lstm_long_losses = []
    all_lstm_long_losses = list(np.log(lstm_dfs[i]['Average long validation losses'][2:])) + list(
        np.log(lstm_dfs_extra_epochs[i]['Average long validation losses']))

    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = lstm_dfs[i][lstm_dfs[i]['Average validation losses'] == min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    lstm_best_epoch_per_run.append(int(row['epoch']))
    lstm_best_val_loss_per_run.append(float(row['Average validation losses']))
    lstm_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')
    plt.subplots()
    plt.plot(lstm_dfs[i]['epoch'][2:], np.log(lstm_dfs[i]['Average validation losses'][2:]), label='Validation loss')
    plt.plot(lstm_dfs[i]['epoch'][2:], np.log(lstm_dfs[i]['Average train validation losses'][2:]), label='Train loss')
    plt.plot(lstm_dfs[i]['epoch'][2:], np.log(lstm_dfs[i]['Average long validation losses'][2:]),
             label='Long validation loss')
    plt.legend()
    plt.savefig('lstm_loss_plots_20_epochs_run' + str(i) + '.png')
    plt.close()

    plt.subplots()
    plt.plot(all_lstm_epochs, all_lstm_val_losses, label='Validation loss')
    plt.plot(all_lstm_epochs, all_lstm_train_losses, label='Train loss')
    plt.plot(all_lstm_epochs, all_lstm_long_losses, label='Long validation loss')
    plt.legend()
    plt.savefig('lstm_loss_plots_30_epochs_run' + str(i) + '.png')
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
    epoch_starting_1.append(epoch + 1)
    average_lstm_best_epoch += (epoch + 1)
print('lstm best epoch per run (starting from 0) = ', lstm_best_epoch_per_run)
print('lstm best epoch per run (starting from 1) = ', epoch_starting_1)
average_lstm_best_epoch = average_lstm_best_epoch / len(lstm_best_epoch_per_run)
print('lstm average best epoch across 10 runs of 20 epochs = ', average_lstm_best_epoch,
      ' (assuming epoch starts at 1)')
std_lstm_best_epoch = np.std(epoch_starting_1)
print('lstm standard deviation of best epoch across 10 runs of 20 epochs = ', std_lstm_best_epoch,
      '(assuming epoch starts at 1)')
average_lstm_best_val_loss = np.mean(lstm_best_val_loss_per_run)
print('lstm average lowest val loss across 10 runs of 20 epochs = ', average_lstm_best_val_loss)
std_lstm_best_val_loss = np.std(lstm_best_val_loss_per_run)
print('lstm standard deviation of lowest val loss across 10 runs of 20 epochs = ', std_lstm_best_val_loss)
average_lstm_best_val_accuracy = np.mean(lstm_best_corresponding_val_acc_per_run)
print('lstm average corresponding val accuracy across 10 runs of 20 epochs = ', average_lstm_best_val_accuracy)
std_lstm_best_val_accuracy = np.std(lstm_best_corresponding_val_acc_per_run)
print('lstm standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ',
      std_lstm_best_val_accuracy)

print('lstm average val loss at epoch 10 (assuming starting from 1) = ', np.mean(lstm_val_losses_10_epochs))
print('lstm average val loss at epoch 15 (assuming starting from 1) = ', np.mean(lstm_val_losses_15_epochs))
print('lstm average val loss at epoch 20 (assuming starting from 1) = ', np.mean(lstm_val_losses_20_epochs))
print('lstm average val loss at epoch 25 (assuming starting from 1) = ', np.mean(lstm_val_losses_25_epochs))
print('lstm average val loss at epoch 30 (assuming starting from 1) = ', np.mean(lstm_val_losses_30_epochs))

print('lstm average val loss change from epoch 10 to epoch 15 = ',
      (np.mean(lstm_val_losses_10_epochs) - np.mean(lstm_val_losses_15_epochs)) / np.mean(
          lstm_val_losses_10_epochs) * 100, '%')
print('lstm average val loss change from epoch 15 to epoch 20 = ',
      (np.mean(lstm_val_losses_15_epochs) - np.mean(lstm_val_losses_20_epochs)) / np.mean(
          lstm_val_losses_15_epochs) * 100, '%')
print('lstm average val loss change from epoch 20 to epoch 25 = ',
      (np.mean(lstm_val_losses_20_epochs) - np.mean(lstm_val_losses_25_epochs)) / np.mean(
          lstm_val_losses_20_epochs) * 100, '%')
print('lstm average val loss change from epoch 25 to epoch 30 = ',
      (np.mean(lstm_val_losses_25_epochs) - np.mean(lstm_val_losses_30_epochs)) / np.mean(
          lstm_val_losses_25_epochs) * 100, '%')

min_validation_losses = []
max_validation_losses = []
avg_validation_losses = []
std_validation_losses = []

epochs = [i for i in range(5, 30, 1)]

for j in range(5, 20, 1):
    losses = []
    for i in range(len(lstm_dfs)):
        losses.append(lstm_dfs[i]['Average validation losses'][j])

    min_validation_losses.append(min(losses))
    max_validation_losses.append(max(losses))
    avg_validation_losses.append(np.mean(losses))
    std_validation_losses.append(np.std(losses))

epochs_2 = [i for i in range(10)]

for j in range(10):
    losses = []
    for i in range(10):
        losses.append(lstm_dfs_extra_epochs[i]['Average validation losses'][j])

    min_validation_losses.append(min(losses))
    max_validation_losses.append(max(losses))
    avg_validation_losses.append(np.mean(losses))
    std_validation_losses.append(np.std(losses))

epochs_all = []
for epoch in epochs:
    epochs_all.append(epoch)

for epoch in epochs_2:
    epochs_all.append(epoch + 20)
plt.subplots()
plt.plot(epochs_all[10:], min_validation_losses[10:], label='Minimum validation loss')
plt.plot(epochs_all[10:], max_validation_losses[10:], label='Maximum validation loss')
plt.plot(epochs_all[10:], avg_validation_losses[10:], label='Avergage validation loss')
plt.legend()
plt.savefig('lstm_loss_plots_30_epochs.png')
plt.close()

print('Train Accuracy = (avg, min, max)', np.mean(lstm_best_model_train_val_accuracies),
      np.min(lstm_best_model_train_val_accuracies), np.max(lstm_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(lstm_best_model_val_accuracies),
      np.min(lstm_best_model_val_accuracies), np.max(lstm_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(lstm_best_model_long_val_accuracies),
      np.min(lstm_best_model_long_val_accuracies), np.max(lstm_best_model_long_val_accuracies))

print('************************************')

gru_dfs = read_sheets(10, gru_excel_path)
gru_dfs_extra_epochs = read_sheets(10, gru_excel_path_extra_epochs)
print(len(gru_dfs[0]))

print('*************')
print('GRU')

gru_best_val_loss_per_run = []
gru_best_corresponding_val_acc_per_run = []
gru_best_epoch_per_run = []
gru_run = []

gru_best_model_train_val_accuracies = []
gru_best_model_long_val_accuracies = []
gru_best_model_val_accuracies = []

gru_val_losses_10_epochs = []
gru_val_losses_15_epochs = []
gru_val_losses_20_epochs = []
gru_val_losses_25_epochs = []
gru_val_losses_30_epochs = []

for i in range(len(gru_dfs)):
    gru_val_losses = []
    gru_train_val_losses = []
    gru_long_val_losses = []

    gru_run.append(i)
    val_losses = gru_dfs[i]['Average validation losses']
    gru_val_losses_10_epochs.append(float(val_losses[9]))
    gru_val_losses_15_epochs.append(float(val_losses[14]))
    gru_val_losses_20_epochs.append(float(val_losses[19]))
    val_losses_extra = gru_dfs_extra_epochs[i]['Average validation losses']
    gru_val_losses_25_epochs.append(float(val_losses_extra[4]))
    gru_val_losses_30_epochs.append(float(val_losses_extra[9]))

    all_gru_val_losses = []
    all_gru_val_losses = list(np.log(gru_dfs[i]['Average validation losses'][2:])) + list(
        np.log(gru_dfs_extra_epochs[i]['Average validation losses']))
    all_gru_epochs = list(gru_dfs[i]['epoch'][2:]) + list(gru_dfs_extra_epochs[i]['epoch'])
    all_gru_train_losses = []
    all_gru_train_losses = list(np.log(gru_dfs[i]['Average train validation losses'][2:])) + list(
        np.log(gru_dfs_extra_epochs[i]['Average train validation losses']))
    all_gru_long_losses = []
    all_gru_long_losses = list(np.log(gru_dfs[i]['Average long validation losses'][2:])) + list(
        np.log(gru_dfs_extra_epochs[i]['Average long validation losses']))

    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = gru_dfs[i][gru_dfs[i]['Average validation losses'] == min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    gru_best_epoch_per_run.append(int(row['epoch']))
    gru_best_val_loss_per_run.append(float(row['Average validation losses']))
    gru_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')
    plt.subplots()
    plt.plot(gru_dfs[i]['epoch'][2:], np.log(gru_dfs[i]['Average validation losses'][2:]), label='Validation loss')
    plt.plot(gru_dfs[i]['epoch'][2:], np.log(gru_dfs[i]['Average train validation losses'][2:]), label='Train loss')
    plt.plot(gru_dfs[i]['epoch'][2:], np.log(gru_dfs[i]['Average long validation losses'][2:]),
             label='Long validation loss')
    plt.legend()
    plt.savefig('gru_loss_plots_20_epochs_run' + str(i) + '.png')
    plt.close()

    plt.subplots()
    plt.plot(all_gru_epochs, all_gru_val_losses, label='Validation loss')
    plt.plot(all_gru_epochs, all_gru_train_losses, label='Train loss')
    plt.plot(all_gru_epochs, all_gru_long_losses, label='Long validation loss')
    plt.legend()
    plt.savefig('gru_loss_plots_30_epochs_run' + str(i) + '.png')
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
    epoch_starting_1.append(epoch + 1)
    average_gru_best_epoch += (epoch + 1)
print('gru best epoch per run (starting from 0) = ', gru_best_epoch_per_run)
print('gru best epoch per run (starting from 1) = ', epoch_starting_1)
average_gru_best_epoch = average_gru_best_epoch / len(gru_best_epoch_per_run)
print('gru average best epoch across 10 runs of 20 epochs = ', average_gru_best_epoch, ' (assuming epoch starts at 1)')
std_gru_best_epoch = np.std(epoch_starting_1)
print('gru standard deviation of best epoch across 10 runs of 20 epochs = ', std_gru_best_epoch,
      '(assuming epoch starts at 1)')
average_gru_best_val_loss = np.mean(gru_best_val_loss_per_run)
print('gru average lowest val loss across 10 runs of 20 epochs = ', average_gru_best_val_loss)
std_gru_best_val_loss = np.std(gru_best_val_loss_per_run)
print('gru standard deviation of lowest val loss across 10 runs of 20 epochs = ', std_gru_best_val_loss)
average_gru_best_val_accuracy = np.mean(gru_best_corresponding_val_acc_per_run)
print('gru average corresponding val accuracy across 10 runs of 20 epochs = ', average_gru_best_val_accuracy)
std_gru_best_val_accuracy = np.std(gru_best_corresponding_val_acc_per_run)
print('gru standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ', std_gru_best_val_accuracy)

print('gru average val loss at epoch 10 (assuming starting from 1) = ', np.mean(gru_val_losses_10_epochs))
print('gru average val loss at epoch 15 (assuming starting from 1) = ', np.mean(gru_val_losses_15_epochs))
print('gru average val loss at epoch 20 (assuming starting from 1) = ', np.mean(gru_val_losses_20_epochs))
print('gru average val loss at epoch 25 (assuming starting from 1) = ', np.mean(gru_val_losses_25_epochs))
print('gru average val loss at epoch 30 (assuming starting from 1) = ', np.mean(gru_val_losses_30_epochs))
print('gru average val loss change from epoch 10 to epoch 15 = ',
      (np.mean(gru_val_losses_10_epochs) - np.mean(gru_val_losses_15_epochs)) / np.mean(gru_val_losses_10_epochs) * 100,
      '%')
print('gru average val loss change from epoch 15 to epoch 20 = ',
      (np.mean(gru_val_losses_15_epochs) - np.mean(gru_val_losses_20_epochs)) / np.mean(gru_val_losses_15_epochs) * 100,
      '%')
print('gru average val loss change from epoch 20 to epoch 25 = ',
      (np.mean(gru_val_losses_20_epochs) - np.mean(gru_val_losses_25_epochs)) / np.mean(gru_val_losses_20_epochs) * 100,
      '%')
print('gru average val loss change from epoch 25 to epoch 30 = ',
      (np.mean(gru_val_losses_25_epochs) - np.mean(gru_val_losses_30_epochs)) / np.mean(gru_val_losses_25_epochs) * 100,
      '%')

print('Train Accuracy = (avg, min, max)', np.mean(gru_best_model_train_val_accuracies),
      np.min(gru_best_model_train_val_accuracies), np.max(gru_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(gru_best_model_val_accuracies),
      np.min(gru_best_model_val_accuracies), np.max(gru_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(gru_best_model_long_val_accuracies),
      np.min(gru_best_model_long_val_accuracies), np.max(gru_best_model_long_val_accuracies))

relu_dfs_1 = read_sheets(10, relu_excel_path)  # 10 runs, 50 epochs
relu_dfs_2 = read_sheets(20, relu_excel_path_2)  # 20 runs, 30 epochs
relu_dfs_3 = read_sheets(5, relu_excel_path_3)  # 5 runs, 30 epochs

relu_dfs = []
# for df in relu_dfs_1:
#     relu_dfs.append(df)
#
# for df in relu_dfs_2:
#     relu_dfs.append(df)
#
# for df in relu_dfs_3:
#     relu_dfs.append(df)

relu_dfs.append(relu_dfs_1[4])
relu_dfs.append(relu_dfs_1[5])
relu_dfs.append(relu_dfs_1[8])
relu_dfs.append(relu_dfs_2[4])
relu_dfs.append(relu_dfs_2[5])
relu_dfs.append(relu_dfs_2[8])
relu_dfs.append(relu_dfs_2[10])
relu_dfs.append(relu_dfs_2[12])
relu_dfs.append(relu_dfs_3[0])
relu_dfs.append(relu_dfs_3[3])

print(len(relu_dfs))

relu_dfs_extra_epochs = relu_dfs
for df in relu_dfs_extra_epochs:
    df = df[20:]

relu_best_val_loss_per_run = []
relu_best_corresponding_val_acc_per_run = []
relu_best_epoch_per_run = []
relu_run = []

relu_best_model_train_val_accuracies = []
relu_best_model_long_val_accuracies = []
relu_best_model_val_accuracies = []

relu_val_losses_10_epochs = []
relu_val_losses_15_epochs = []
relu_val_losses_20_epochs = []
relu_val_losses_25_epochs = []
relu_val_losses_30_epochs = []

relu_val_losses_1_epochs = []
relu_val_losses_2_epochs = []
relu_val_losses_3_epochs = []
relu_val_losses_4_epochs = []
relu_val_losses_5_epochs = []
relu_val_losses_6_epochs = []
relu_val_losses_7_epochs = []
relu_val_losses_8_epochs = []
relu_val_losses_9_epochs = []
relu_val_losses_11_epochs = []
relu_val_losses_12_epochs = []
relu_val_losses_13_epochs = []
relu_val_losses_14_epochs = []
relu_val_losses_16_epochs = []
relu_val_losses_17_epochs = []
relu_val_losses_18_epochs = []
relu_val_losses_19_epochs = []
relu_val_losses_21_epochs = []
relu_val_losses_22_epochs = []
relu_val_losses_23_epochs = []
relu_val_losses_24_epochs = []
relu_val_losses_26_epochs = []
relu_val_losses_27_epochs = []
relu_val_losses_28_epochs = []
relu_val_losses_29_epochs = []

print('*************')
print('ReLU')

relu_final_best_val_loss_per_run = []
relu_final_best_corresponding_val_acc_per_run = []
relu_final_best_epoch_per_run = []
relu_final_run = []

relu_final_best_model_train_val_accuracies = []
relu_final_best_model_long_val_accuracies = []
relu_final_best_model_val_accuracies = []

relu_final_val_losses_10_epochs = []
relu_final_val_losses_15_epochs = []
relu_final_val_losses_20_epochs = []
relu_final_val_losses_25_epochs = []
relu_final_val_losses_30_epochs = []

df_indices = [i for i in range(len(relu_dfs))]
print(df_indices)

for i in range(len(relu_dfs)):
    relu_val_losses = []
    relu_train_val_losses = []
    relu_long_val_losses = []

    relu_run.append(i)
    val_losses = relu_dfs[i]['Average validation losses'][:20]
    relu_val_losses_1_epochs.append(float(val_losses[0]))
    relu_val_losses_2_epochs.append(float(val_losses[1]))
    relu_val_losses_3_epochs.append(float(val_losses[2]))
    relu_val_losses_4_epochs.append(float(val_losses[3]))
    relu_val_losses_5_epochs.append(float(val_losses[4]))
    relu_val_losses_6_epochs.append(float(val_losses[5]))
    relu_val_losses_7_epochs.append(float(val_losses[6]))
    relu_val_losses_8_epochs.append(float(val_losses[7]))
    relu_val_losses_9_epochs.append(float(val_losses[8]))
    relu_val_losses_10_epochs.append(float(val_losses[9]))
    relu_val_losses_11_epochs.append(float(val_losses[10]))
    relu_val_losses_12_epochs.append(float(val_losses[11]))
    relu_val_losses_13_epochs.append(float(val_losses[12]))
    relu_val_losses_14_epochs.append(float(val_losses[13]))
    relu_val_losses_15_epochs.append(float(val_losses[14]))
    relu_val_losses_16_epochs.append(float(val_losses[15]))
    relu_val_losses_17_epochs.append(float(val_losses[16]))
    relu_val_losses_18_epochs.append(float(val_losses[17]))
    relu_val_losses_19_epochs.append(float(val_losses[18]))
    relu_val_losses_20_epochs.append(float(val_losses[19]))
    val_losses_extra = relu_dfs_extra_epochs[i]['Average validation losses'][:30]
    relu_val_losses_21_epochs.append(float(val_losses_extra[0]))
    relu_val_losses_22_epochs.append(float(val_losses_extra[1]))
    relu_val_losses_23_epochs.append(float(val_losses_extra[2]))
    relu_val_losses_24_epochs.append(float(val_losses_extra[3]))
    relu_val_losses_25_epochs.append(float(val_losses_extra[4]))
    relu_val_losses_26_epochs.append(float(val_losses_extra[5]))
    relu_val_losses_27_epochs.append(float(val_losses_extra[6]))
    relu_val_losses_28_epochs.append(float(val_losses_extra[7]))
    relu_val_losses_29_epochs.append(float(val_losses_extra[8]))
    relu_val_losses_30_epochs.append(float(val_losses_extra[9]))

    # val_losses_extra = relu_dfs[i]['Average validation losses'][:20]

    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = relu_dfs[i][relu_dfs[i]['Average validation losses'] == min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    relu_best_epoch_per_run.append(int(row['epoch']))
    relu_best_val_loss_per_run.append(float(row['Average validation losses']))
    relu_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')
    plt.subplots()
    plt.plot(relu_dfs[i]['epoch'][2:20], np.log(relu_dfs[i]['Average validation losses'][2:20]),
             label='Validation loss')
    plt.plot(relu_dfs[i]['epoch'][2:20], np.log(relu_dfs[i]['Average train validation losses'][2:20]),
             label='Train loss')
    plt.plot(relu_dfs[i]['epoch'][2:20], np.log(relu_dfs[i]['Average long validation losses'][2:20]),
             label='Long validation loss')
    plt.legend()
    plt.savefig('relu_loss_plots_20_epochs_run' + str(i) + '****.png')
    plt.close()

    plt.subplots()
    plt.plot(relu_dfs[i]['epoch'][2:30], np.log(relu_dfs[i]['Average validation losses'][2:30]),
             label='Validation loss')
    plt.plot(relu_dfs[i]['epoch'][2:30], np.log(relu_dfs[i]['Average train validation losses'][2:30]),
             label='Train loss')
    plt.plot(relu_dfs[i]['epoch'][2:30], np.log(relu_dfs[i]['Average long validation losses'][2:30]),
             label='Long validation loss')
    plt.legend()
    plt.savefig('relu_loss_plots_30_epochs_run' + str(i) + '****.png')
    plt.close()

    relu_best_model_val_accuracies.append(float(row['Validation accuracies']))
    relu_best_model_train_val_accuracies.append(float(row['Train validation accuracies']))
    relu_best_model_long_val_accuracies.append(float(row['Long validation accuracies']))

average_relu_best_epoch = 0
std_relu_best_epoch = 0
epoch_starting_1 = []
for epoch in relu_best_epoch_per_run:
    # print(epoch)
    # epoch+=1
    # print(epoch)
    # print("***************")
    epoch_starting_1.append(epoch + 1)
    average_relu_best_epoch += (epoch + 1)
print('relu best epoch per run (starting from 0) = ', relu_best_epoch_per_run)
print('relu best epoch per run (starting from 1) = ', epoch_starting_1)
print('relu best val loss per run = ', relu_best_val_loss_per_run)
# print(len(relu_best_val_loss_per_run))
average_relu_best_epoch = average_relu_best_epoch / len(relu_best_epoch_per_run)
print('relu average best epoch across 10 runs of 20 epochs = ', average_relu_best_epoch,
      ' (assuming epoch starts at 1)')
std_relu_best_epoch = np.std(epoch_starting_1)
print('relu standard deviation of best epoch across 10 runs of 20 epochs = ', std_relu_best_epoch,
      '(assuming epoch starts at 1)')
average_relu_best_val_loss = np.mean(relu_best_val_loss_per_run)
print('relu average lowest val loss across 10 runs of 20 epochs = ', average_relu_best_val_loss)
std_relu_best_val_loss = np.std(relu_best_val_loss_per_run)
print('relu standard deviation of lowest val loss across 10 runs of 20 epochs = ', std_relu_best_val_loss)
average_relu_best_val_accuracy = np.mean(relu_best_corresponding_val_acc_per_run)
print('relu average corresponding val accuracy across 10 runs of 20 epochs = ', average_relu_best_val_accuracy)
std_relu_best_val_accuracy = np.std(relu_best_corresponding_val_acc_per_run)
print('relu standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ',
      std_relu_best_val_accuracy)

print('relu average val loss at epoch 10 (assuming starting from 1) = ', np.mean(relu_val_losses_10_epochs))
print('relu average val loss at epoch 15 (assuming starting from 1) = ', np.mean(relu_val_losses_15_epochs))
print('relu average val loss at epoch 20 (assuming starting from 1) = ', np.mean(relu_val_losses_20_epochs))
print('relu average val loss at epoch 25 (assuming starting from 1) = ', np.mean(relu_val_losses_25_epochs))
print('relu average val loss at epoch 30 (assuming starting from 1) = ', np.mean(relu_val_losses_30_epochs))
print('relu std val loss at epoch 10 (assuming starting from 1) = ', np.std(relu_val_losses_10_epochs))
print('relu std val loss at epoch 15 (assuming starting from 1) = ', np.std(relu_val_losses_15_epochs))
print('relu std val loss at epoch 20 (assuming starting from 1) = ', np.std(relu_val_losses_20_epochs))
print('relu std val loss at epoch 25 (assuming starting from 1) = ', np.std(relu_val_losses_25_epochs))
print('relu std val loss at epoch 30 (assuming starting from 1) = ', np.std(relu_val_losses_30_epochs))
print('relu average val loss change from epoch 10 to epoch 15 = ',
      (np.mean(relu_val_losses_10_epochs) - np.mean(relu_val_losses_15_epochs)) / np.mean(
          relu_val_losses_10_epochs) * 100, '%')
print('relu average val loss change from epoch 15 to epoch 20 = ',
      (np.mean(relu_val_losses_15_epochs) - np.mean(relu_val_losses_20_epochs)) / np.mean(
          relu_val_losses_15_epochs) * 100, '%')
print('relu average val loss change from epoch 20 to epoch 25 = ',
      (np.mean(relu_val_losses_20_epochs) - np.mean(relu_val_losses_25_epochs)) / np.mean(
          relu_val_losses_20_epochs) * 100, '%')
print('relu average val loss change from epoch 25 to epoch 30 = ',
      (np.mean(relu_val_losses_25_epochs) - np.mean(relu_val_losses_30_epochs)) / np.mean(
          relu_val_losses_25_epochs) * 100, '%')

print('**********')

print('relu average val loss at epoch 1 (assuming starting from 1) = ', np.mean(relu_val_losses_1_epochs), ', std =',
      np.std(relu_val_losses_1_epochs))
print('relu average val loss at epoch 2 (assuming starting from 1) = ', np.mean(relu_val_losses_2_epochs), ', std =',
      np.std(relu_val_losses_2_epochs))
print('relu average val loss at epoch 3 (assuming starting from 3) = ', np.mean(relu_val_losses_3_epochs), ', std =',
      np.std(relu_val_losses_3_epochs))
print('relu average val loss at epoch 4 (assuming starting from 4) = ', np.mean(relu_val_losses_4_epochs), ', std =',
      np.std(relu_val_losses_4_epochs))
print('relu average val loss at epoch 5 (assuming starting from 5) = ', np.mean(relu_val_losses_5_epochs), ', std =',
      np.std(relu_val_losses_5_epochs))
print('relu average val loss at epoch 6 (assuming starting from 6) = ', np.mean(relu_val_losses_6_epochs), ', std =',
      np.std(relu_val_losses_6_epochs))
print('relu average val loss at epoch 7 (assuming starting from 7) = ', np.mean(relu_val_losses_7_epochs), ', std =',
      np.std(relu_val_losses_7_epochs))
print('relu average val loss at epoch 8 (assuming starting from 8) = ', np.mean(relu_val_losses_8_epochs), ', std =',
      np.std(relu_val_losses_8_epochs))
print('relu average val loss at epoch 9 (assuming starting from 9) = ', np.mean(relu_val_losses_9_epochs), ', std =',
      np.std(relu_val_losses_9_epochs))
print('relu average val loss at epoch 10 (assuming starting from 10) = ', np.mean(relu_val_losses_10_epochs), ', std =',
      np.std(relu_val_losses_10_epochs))
print('relu average val loss at epoch 11 (assuming starting from 11) = ', np.mean(relu_val_losses_11_epochs), ', std =',
      np.std(relu_val_losses_11_epochs))
print('relu average val loss at epoch 12 (assuming starting from 12) = ', np.mean(relu_val_losses_12_epochs), ', std =',
      np.std(relu_val_losses_12_epochs))
print('relu average val loss at epoch 13 (assuming starting from 13) = ', np.mean(relu_val_losses_13_epochs), ', std =',
      np.std(relu_val_losses_13_epochs))
print('relu average val loss at epoch 14 (assuming starting from 14) = ', np.mean(relu_val_losses_14_epochs), ', std =',
      np.std(relu_val_losses_14_epochs))
print('relu average val loss at epoch 15 (assuming starting from 15) = ', np.mean(relu_val_losses_15_epochs), ', std =',
      np.std(relu_val_losses_15_epochs))
print('relu average val loss at epoch 16 (assuming starting from 16) = ', np.mean(relu_val_losses_16_epochs), ', std =',
      np.std(relu_val_losses_16_epochs))
print('relu average val loss at epoch 17 (assuming starting from 17) = ', np.mean(relu_val_losses_17_epochs), ', std =',
      np.std(relu_val_losses_17_epochs))
print('relu average val loss at epoch 18 (assuming starting from 18) = ', np.mean(relu_val_losses_18_epochs), ', std =',
      np.std(relu_val_losses_18_epochs))
print('relu average val loss at epoch 19 (assuming starting from 19) = ', np.mean(relu_val_losses_19_epochs), ', std =',
      np.std(relu_val_losses_19_epochs))
print('relu average val loss at epoch 20 (assuming starting from 20) = ', np.mean(relu_val_losses_20_epochs), ', std =',
      np.std(relu_val_losses_20_epochs))
print('relu average val loss at epoch 21 (assuming starting from 21) = ', np.mean(relu_val_losses_21_epochs), ', std =',
      np.std(relu_val_losses_21_epochs))
print('relu average val loss at epoch 22 (assuming starting from 22) = ', np.mean(relu_val_losses_22_epochs), ', std =',
      np.std(relu_val_losses_22_epochs))
print('relu average val loss at epoch 23 (assuming starting from 23) = ', np.mean(relu_val_losses_23_epochs), ', std =',
      np.std(relu_val_losses_23_epochs))
print('relu average val loss at epoch 24 (assuming starting from 24) = ', np.mean(relu_val_losses_24_epochs), ', std =',
      np.std(relu_val_losses_24_epochs))
print('relu average val loss at epoch 25 (assuming starting from 25) = ', np.mean(relu_val_losses_25_epochs), ', std =',
      np.std(relu_val_losses_25_epochs))
print('relu average val loss at epoch 26 (assuming starting from 26) = ', np.mean(relu_val_losses_26_epochs), ', std =',
      np.std(relu_val_losses_26_epochs))
print('relu average val loss at epoch 27 (assuming starting from 27) = ', np.mean(relu_val_losses_27_epochs), ', std =',
      np.std(relu_val_losses_27_epochs))
print('relu average val loss at epoch 28 (assuming starting from 28) = ', np.mean(relu_val_losses_28_epochs), ', std =',
      np.std(relu_val_losses_28_epochs))
print('relu average val loss at epoch 29 (assuming starting from 29) = ', np.mean(relu_val_losses_29_epochs), ', std =',
      np.std(relu_val_losses_29_epochs))
print('relu average val loss at epoch 30 (assuming starting from 30) = ', np.mean(relu_val_losses_30_epochs), ', std =',
      np.std(relu_val_losses_30_epochs))

min_validation_losses = []
max_validation_losses = []
avg_validation_losses = []
std_validation_losses = []

epochs = [i for i in range(1, 31, 1)]

for j in range(30):
    losses = []
    for i in range(len(relu_dfs)):
        losses.append(relu_dfs[i]['Average validation losses'][j])

    min_validation_losses.append(min(losses))
    max_validation_losses.append(max(losses))
    avg_validation_losses.append(np.mean(losses))
    std_validation_losses.append(np.std(losses))

plt.subplots()
plt.plot(epochs, min_validation_losses, label='Minimum validation loss')
plt.plot(epochs, max_validation_losses, label='Maximum validation loss')
plt.plot(epochs, avg_validation_losses, label='Avergage validation loss')
plt.legend()
plt.savefig('relu_loss_plots_30_epochs.png')
plt.close()

plt.subplots()
plt.rcParams['font.size'] = '12'
plt.plot(epochs, np.log10(min_validation_losses), label='Minimum validation loss', color='red', linestyle='dashed')
plt.plot(epochs, np.log10(max_validation_losses), label='Maximum validation loss', color='green', linestyle='dotted')
plt.plot(epochs, np.log10(avg_validation_losses), label='Average validation loss', color='blue')
# plt.ylabel('Decimal log of the validation loss', fontsize=14)
plt.ylabel('Validation loss (log)', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
# plt.title('Progression of the validation loss during training\n for ReLU models', fontsize=16)
plt.legend(prop={'size': 12})
plt.savefig('relu_loss_plots_30_epochs_LOGARITHMIC.png')
plt.close()

print('Train Accuracy = (avg, min, max)', np.mean(relu_best_model_train_val_accuracies),
      np.min(relu_best_model_train_val_accuracies), np.max(relu_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(relu_best_model_val_accuracies),
      np.min(relu_best_model_val_accuracies), np.max(relu_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(relu_best_model_long_val_accuracies),
      np.min(relu_best_model_long_val_accuracies), np.max(relu_best_model_long_val_accuracies))

print('************************************')

lstm2_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs.xlsx'

lstm2_dfs = read_sheets(10, lstm2_excel_path)

print(len(lstm2_dfs[0]))

print('*************')
print('lstm2')

lstm2_best_val_loss_per_run = []
lstm2_best_corresponding_val_acc_per_run = []
lstm2_best_epoch_per_run = []
lstm2_run = []

lstm2_best_model_train_val_accuracies = []
lstm2_best_model_long_val_accuracies = []
lstm2_best_model_val_accuracies = []

lstm2_val_losses_10_epochs = []
lstm2_val_losses_15_epochs = []
lstm2_val_losses_20_epochs = []
lstm2_val_losses_25_epochs = []
lstm2_val_losses_30_epochs = []

for i in range(len(lstm2_dfs)):
    lstm2_val_losses = []
    lstm2_train_val_losses = []
    lstm2_long_val_losses = []

    lstm2_run.append(i)
    val_losses = lstm2_dfs[i]['Average validation losses']
    lstm2_val_losses_10_epochs.append(float(val_losses[9]))
    lstm2_val_losses_15_epochs.append(float(val_losses[14]))
    lstm2_val_losses_20_epochs.append(float(val_losses[19]))
    lstm2_val_losses_25_epochs.append(float(val_losses[24]))
    lstm2_val_losses_30_epochs.append(float(val_losses[29]))

    all_lstm2_val_losses = []
    all_lstm2_val_losses = list(np.log(lstm2_dfs[i]['Average validation losses'][2:]))
    all_lstm2_epochs = list(lstm2_dfs[i]['epoch'][2:])
    all_lstm2_train_losses = []
    all_lstm2_train_losses = list(np.log(lstm2_dfs[i]['Average train validation losses'][2:]))
    all_lstm2_long_losses = []
    all_lstm2_long_losses = list(np.log(lstm2_dfs[i]['Average long validation losses'][2:]))

    min_val_loss = min(val_losses)
    # print(min_val_loss)
    row = lstm2_dfs[i][lstm2_dfs[i]['Average validation losses'] == min_val_loss]
    # print(int(row['epoch']), row['Average validation losses'].item, row['Validation accuracies'].item)
    # print(int(row['epoch']))
    # print(float(row['Average validation losses']))
    # print(float(row['Validation accuracies']))
    lstm2_best_epoch_per_run.append(int(row['epoch']))
    lstm2_best_val_loss_per_run.append(float(row['Average validation losses']))
    lstm2_best_corresponding_val_acc_per_run.append(float(row['Validation accuracies']))
    # print('*********')
    plt.subplots()
    plt.plot(lstm2_dfs[i]['epoch'][2:], np.log(lstm2_dfs[i]['Average validation losses'][2:]), label='Validation loss')
    plt.plot(lstm2_dfs[i]['epoch'][2:], np.log(lstm2_dfs[i]['Average train validation losses'][2:]), label='Train loss')
    plt.plot(lstm2_dfs[i]['epoch'][2:], np.log(lstm2_dfs[i]['Average long validation losses'][2:]),
             label='Long validation loss')
    plt.legend()
    plt.savefig('lstm2_loss_plots_30_epochs_run' + str(i) + '.png')
    plt.close()

    plt.subplots()
    plt.plot(all_lstm2_epochs, all_lstm2_val_losses, label='Validation loss')
    plt.plot(all_lstm2_epochs, all_lstm2_train_losses, label='Train loss')
    plt.plot(all_lstm2_epochs, all_lstm2_long_losses, label='Long validation loss')
    plt.legend()
    plt.savefig('lstm2_loss_plots_30_epochs_run' + str(i) + '.png')
    plt.close()

    lstm2_best_model_val_accuracies.append(float(row['Validation accuracies']))
    lstm2_best_model_train_val_accuracies.append(float(row['Train validation accuracies']))
    lstm2_best_model_long_val_accuracies.append(float(row['Long validation accuracies']))

average_lstm2_best_epoch = 0
std_lstm2_best_epoch = 0
epoch_starting_1 = []
for epoch in lstm2_best_epoch_per_run:
    # print(epoch)
    # epoch+=1
    # print(epoch)
    # print("***************")
    epoch_starting_1.append(epoch + 1)
    average_lstm2_best_epoch += (epoch + 1)
print('lstm2 best epoch per run (starting from 0) = ', lstm2_best_epoch_per_run)
print('lstm2 best epoch per run (starting from 1) = ', epoch_starting_1)
average_lstm2_best_epoch = average_lstm2_best_epoch / len(lstm2_best_epoch_per_run)
print('lstm2 average best epoch across 10 runs of 20 epochs = ', average_lstm2_best_epoch,
      ' (assuming epoch starts at 1)')
std_lstm2_best_epoch = np.std(epoch_starting_1)
print('lstm2 standard deviation of best epoch across 10 runs of 20 epochs = ', std_lstm2_best_epoch,
      '(assuming epoch starts at 1)')
average_lstm2_best_val_loss = np.mean(lstm2_best_val_loss_per_run)
print('lstm2 average lowest val loss across 10 runs of 20 epochs = ', average_lstm2_best_val_loss)
std_lstm2_best_val_loss = np.std(lstm2_best_val_loss_per_run)
print('lstm2 standard deviation of lowest val loss across 10 runs of 20 epochs = ', std_lstm2_best_val_loss)
average_lstm2_best_val_accuracy = np.mean(lstm2_best_corresponding_val_acc_per_run)
print('lstm2 average corresponding val accuracy across 10 runs of 20 epochs = ', average_lstm2_best_val_accuracy)
std_lstm2_best_val_accuracy = np.std(lstm2_best_corresponding_val_acc_per_run)
print('lstm2 standard deviation of corresponding val accuracy across 10 rund of 20 epochs = ',
      std_lstm2_best_val_accuracy)

print('lstm2 average val loss at epoch 10 (assuming starting from 1) = ', np.mean(lstm2_val_losses_10_epochs))
print('lstm2 average val loss at epoch 15 (assuming starting from 1) = ', np.mean(lstm2_val_losses_15_epochs))
print('lstm2 average val loss at epoch 20 (assuming starting from 1) = ', np.mean(lstm2_val_losses_20_epochs))
print('lstm2 average val loss at epoch 25 (assuming starting from 1) = ', np.mean(lstm2_val_losses_25_epochs))
print('lstm2 average val loss at epoch 30 (assuming starting from 1) = ', np.mean(lstm2_val_losses_30_epochs))

print('lstm2 average val loss change from epoch 10 to epoch 15 = ',
      (np.mean(lstm2_val_losses_10_epochs) - np.mean(lstm2_val_losses_15_epochs)) / np.mean(
          lstm2_val_losses_10_epochs) * 100, '%')
print('lstm2 average val loss change from epoch 15 to epoch 20 = ',
      (np.mean(lstm2_val_losses_15_epochs) - np.mean(lstm2_val_losses_20_epochs)) / np.mean(
          lstm2_val_losses_15_epochs) * 100, '%')
print('lstm2 average val loss change from epoch 20 to epoch 25 = ',
      (np.mean(lstm2_val_losses_20_epochs) - np.mean(lstm2_val_losses_25_epochs)) / np.mean(
          lstm2_val_losses_20_epochs) * 100, '%')
print('lstm2 average val loss change from epoch 25 to epoch 30 = ',
      (np.mean(lstm2_val_losses_25_epochs) - np.mean(lstm2_val_losses_30_epochs)) / np.mean(
          lstm2_val_losses_25_epochs) * 100, '%')

min_validation_losses = []
max_validation_losses = []
avg_validation_losses = []
std_validation_losses = []

epochs = [i for i in range(1, 30, 1)]

for j in range(1, 30, 1):
    losses = []
    for i in range(len(lstm2_dfs)):
        losses.append(lstm2_dfs[i]['Average validation losses'][j])

    min_validation_losses.append(min(losses))
    max_validation_losses.append(max(losses))
    avg_validation_losses.append(np.mean(losses))
    std_validation_losses.append(np.std(losses))

plt.subplots()
plt.plot(epochs[10:], min_validation_losses[10:], label='Minimum validation loss')
plt.plot(epochs[10:], max_validation_losses[10:], label='Maximum validation loss')
plt.plot(epochs[10:], avg_validation_losses[10:], label='Avergage validation loss')
plt.legend()
plt.savefig('lstm2_loss_plots_30_epochs.png')
plt.close()

plt.subplots()
plt.rcParams['font.size'] = '12'
plt.plot(epochs, np.log10(min_validation_losses), label='Minimum validation loss', color='red', linestyle='dashed')
plt.plot(epochs, np.log10(max_validation_losses), label='Maximum validation loss', color='green', linestyle='dotted')
plt.plot(epochs, np.log10(avg_validation_losses), label='Avergage validation loss', color='blue')
plt.ylabel('Validation loss (log)', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
# plt.title('Validation loss for LSTM models (log scale)', fontsize=16)
plt.legend(prop={'size': 12})
plt.savefig('lstm2_loss_plots_30_epochs_LOGARITHMIC.png')
plt.close()

print('Train Accuracy = (avg, min, max)', np.mean(lstm2_best_model_train_val_accuracies),
      np.min(lstm2_best_model_train_val_accuracies), np.max(lstm2_best_model_train_val_accuracies))
print('Validation Accuracy = (avg, min, max)', np.mean(lstm2_best_model_val_accuracies),
      np.min(lstm2_best_model_val_accuracies), np.max(lstm2_best_model_val_accuracies))
print('Long Accuracy = (avg, min, max)', np.mean(lstm2_best_model_long_val_accuracies),
      np.min(lstm2_best_model_long_val_accuracies), np.max(lstm2_best_model_long_val_accuracies))

print('************************************')