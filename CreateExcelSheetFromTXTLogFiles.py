import pandas as pd
import re

prefix = '/Users/nadineelnaggar/Desktop/SemiDyck1Logs/'

configurations = ['MRI', 'BRI', 'BCIB', 'BCIN', 'MCIB', 'MCIN']

path_semi_dyck_1_BRI = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_BRI_EXCEL = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
path_semi_dyck_1_MRI = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_MRI_EXCEL = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
# path_semi_dyck_1_BRI_colab = ''
# path_semi_dyck_1_MRI_colab = ''
path_semi_dyck_1_BCIB = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNNCorrectInitialisationWithBias_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_BCIN = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNNCorrectInitialisation_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_MCIB = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNNCorrectInitialisationWithBias_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'
path_semi_dyck_1_MCIN = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNNCorrectInitialisation_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_TRAIN_LOG.txt'

path_semi_dyck_1_BCIB_EXCEL = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNNCorrectInitialisationWithBias_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
path_semi_dyck_1_BCIN_EXCEL = prefix+'Dyck1_SemiDyck1BCE_25_bracket_pairs_VanillaReLURNNCorrectInitialisation_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
path_semi_dyck_1_MCIB_EXCEL = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNNCorrectInitialisationWithBias_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
path_semi_dyck_1_MCIN_EXCEL = prefix+'Dyck1_SemiDyck1MSE_25_bracket_pairs_VanillaReLURNNCorrectInitialisation_Feedback_EveryTimestep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'


for c in range(len(configurations)):
    if configurations[c]=='MRI':
        path = path_semi_dyck_1_MRI
        excel_name = path_semi_dyck_1_MRI_EXCEL
    elif configurations[c]=='BRI':
        path = path_semi_dyck_1_BRI
        excel_name=path_semi_dyck_1_BRI_EXCEL
    elif configurations[c]=='BCIB':
        path=path_semi_dyck_1_BCIB
        excel_name=path_semi_dyck_1_BCIB_EXCEL
    elif configurations[c]=='BCIN':
        path=path_semi_dyck_1_BCIN
        excel_name=path_semi_dyck_1_BCIN_EXCEL
    elif configurations[c]=='MCIB':
        path=path_semi_dyck_1_MCIB
        excel_name=path_semi_dyck_1_MCIB_EXCEL
    elif configurations[c]=='MCIN':
        path=path_semi_dyck_1_MCIN
        excel_name=path_semi_dyck_1_MCIN_EXCEL


    epochs = []
    train_accs = []
    train_losses = []
    train_val_accs = []
    train_val_losses = []
    val_accs = []
    val_losses = []
    long_val_accs = []
    long_val_losses = []


    lines = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('Accuracy for epoch'):
                lines.append(line)

    # print(lines[0])


    for i in range(len(lines)):
        line = lines[i].split(',')
        # print(line)
        # print(line[0])
        epoch = line[0].split('epoch ')[1]
        avg_train_acc = epoch.split('=')[1]
        epoch = epoch.split('=')[0]
        # print(epoch)
        avg_train_acc=avg_train_acc.split('%')[0]
        avg_train_loss = line[1].split('= ')[1].split(' ')[0]
        train_val_loss = line[2].split('= ')[1]
        train_val_acc = line[3].split('= ')[1].split('%')[0]
        val_loss = line[4].split('= ')[1]
        val_acc = line[5].split('= ')[1].split('%')[0]
        long_val_loss = line[6].split('= ')[1]
        long_val_acc = line[7].split('= ')[1].split('%')[0]

        epochs.append(int(epoch))
        train_losses.append(float(avg_train_loss))
        train_accs.append(float(avg_train_acc))
        train_val_losses.append(float(train_val_loss))
        train_val_accs.append(float(train_val_acc))
        val_accs.append(float(val_acc))
        val_losses.append(float(val_loss))
        long_val_accs.append(float(long_val_acc))
        long_val_losses.append(float(long_val_loss))
        # if int(epoch)==29:
        #     epochs.append(int(epoch))
        #     train_losses.append(float(avg_train_loss))
        #     train_accs.append(float(avg_train_acc))
        #     train_val_losses.append(float(train_val_loss))
        #     train_val_accs.append(float(train_val_acc))
        #     val_accs.append(float(val_acc))
        #     val_losses.append(float(val_loss))
        #     long_val_accs.append(float(long_val_acc))
        #     long_val_losses.append(float(long_val_loss))


        # print(epoch)
        # # print(avg_train_acc)
        # print(avg_train_loss)
        # print(avg_train_acc)
        # print(train_val_acc)
        # print(train_val_loss)
        # print(val_loss)
        # print(val_acc)
        # print(long_val_loss)
        # print(long_val_acc)
        # break

    runs = []
    train_dataframes = []
    num_full_runs = epochs.count(29)
    print(num_full_runs)
    # count_runs = 0

    # epochs_run_1 = epochs[30:60]
    # print(epochs_run_1)

    for i in range(num_full_runs):
        runs.append("run"+str(i))
        df = pd.DataFrame()
        start_index = i*30
        end_index = start_index+30

        run_epochs = epochs[start_index:end_index]
        run_epochs.append(run_epochs[-1])
        # print(run_epochs)
        run_train_accs = train_accs[start_index:end_index]
        run_train_accs.append(run_train_accs[-1])
        run_train_losses = train_losses[start_index:end_index]
        run_train_losses.append(run_train_losses[-1])
        run_train_val_accs = train_val_accs[start_index:end_index]
        run_train_val_accs.append(run_train_accs[-1])
        run_train_val_losses = train_val_losses[start_index:end_index]
        run_train_val_losses.append(run_train_val_losses[-1])
        run_val_losses = val_losses[start_index:end_index]
        run_val_losses.append(run_val_losses[-1])
        run_val_accs = val_accs[start_index:end_index]
        run_val_accs.append(run_val_accs[-1])
        run_long_val_losses = long_val_losses[start_index:end_index]
        run_long_val_losses.append(run_long_val_losses[-1])
        run_long_val_accs = long_val_accs[start_index:end_index]
        run_long_val_accs.append(run_long_val_accs[-1])
        
        


        df['epoch'] =run_epochs
        df['Training accuracies'] =run_train_accs
        df['Average training losses'] =run_train_losses
        df['Train validation accuracies']=run_train_val_accs
        df['Average train validation losses']=run_train_val_losses
        df['Average validation losses']=run_val_losses
        df['Validation accuracies'] =run_val_accs
        df['Average long validation losses']=run_long_val_losses
        df['Long validation accuracies'] =run_long_val_accs
        train_dataframes.append(df)





    dfs = dict(zip(runs, train_dataframes))
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    for sheet_name in dfs.keys():
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()

