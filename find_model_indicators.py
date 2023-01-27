import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from models_batch import VanillaLSTM, VanillaReLURNN
import os
import scipy
from scipy import stats




parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, help='type of model, LSTM, ReLU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


relu_excel_path = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
relu_excel_path_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
relu_excel_path_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs.xlsx'

relu_prefix_1 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
relu_prefix_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
relu_prefix_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'

excel_name = relu_prefix_2+'INFERENCE_1000token_1checkpoint_step_upto5000checkpoints_EXCEL INFERENCE CHECKPOINTS ONLY GOOD MODELS WITH INDICATORS.xlsx'


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



args = parser.parse_args()

model_type = args.model_type

def getReLUModels():
    best_runs_10runs_50epochs = [4, 5, 8]
    best_runs_20runs_30epochs = [4, 5, 8, 10, 12]
    best_runs_5runs_30epochs = [0, 3]
    checkpoints = []

    # relu_prefix_1 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    # relu_prefix_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
    # relu_prefix_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'

    relu_checkpoint_prefix_1 = relu_prefix_1 + 'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_CHECKPOINT'
    relu_checkpoint_prefix_2 = relu_prefix_2 + 'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_CHECKPOINT'
    relu_checkpoint_prefix_3 = relu_prefix_3 + 'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs_CHECKPOINT'

    for i in range(len(best_runs_10runs_50epochs)):
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch25.pth')

    for i in range(len(best_runs_20runs_30epochs)):
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch25.pth')

    for i in range(len(best_runs_5runs_30epochs)):
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch0.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch5.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch10.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch15.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch20.pth')
        checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch25.pth')

    # extract the excel sheets corresponding to the good runs
    # pass

    # relu_dfs_1 = read_sheets(10, relu_excel_path)  # 10 runs, 50 epochs
    # relu_dfs_2 = read_sheets(20, relu_excel_path_2)  # 20 runs, 30 epochs
    # relu_dfs_3 = read_sheets(5, relu_excel_path_3)  # 5 runs, 30 epochs
    # relu_dfs = []
    # relu_dfs.append(relu_dfs_1[4])
    # relu_dfs.append(relu_dfs_1[5])
    # relu_dfs.append(relu_dfs_1[8])
    # relu_dfs.append(relu_dfs_2[4])
    # relu_dfs.append(relu_dfs_2[5])
    # relu_dfs.append(relu_dfs_2[8])
    # relu_dfs.append(relu_dfs_2[10])
    # relu_dfs.append(relu_dfs_2[12])
    # relu_dfs.append(relu_dfs_3[0])
    # relu_dfs.append(relu_dfs_3[3])

    # excel_name = relu_prefix_2+'INFERENCE_1000token_1checkpoint_step_upto5000checkpoints_EXCEL INFERENCE CHECKPOINTS ONLY GOOD MODELS WITH INDICATORS.xlsx'

    df = pd.read_excel(excel_name, sheet_name='Sheet1')

    return checkpoints, df


def extractModelIndicators():

    if model_type=='ReLU':
        input_size = 2
        hidden_size = 1
        num_classes = 2
        output_activation='Sigmoid'


        # best_runs_10runs_50epochs = [4, 5, 8]
        # best_runs_20runs_30epochs = [4, 5, 8, 10, 12]
        # best_runs_5runs_30epochs = [0, 3]
        # checkpoints = []
        #
        # relu_prefix_1 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
        # relu_prefix_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
        # relu_prefix_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'
        #
        # relu_checkpoint_prefix_1 = relu_prefix_1+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_CHECKPOINT'
        # relu_checkpoint_prefix_2 = relu_prefix_2+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs_CHECKPOINT'
        # relu_checkpoint_prefix_3 = relu_prefix_3+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs_CHECKPOINT'
        #
        #
        #
        # for i in range(len(best_runs_10runs_50epochs)):
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i])+'_epoch0.pth')
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch5.pth')
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch10.pth')
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch15.pth')
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch20.pth')
        #     checkpoints.append(relu_checkpoint_prefix_1 + '_run' + str(best_runs_10runs_50epochs[i]) + '_epoch25.pth')
        #
        # for i in range(len(best_runs_20runs_30epochs)):
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i])+'_epoch0.pth')
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch5.pth')
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch10.pth')
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch15.pth')
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch20.pth')
        #     checkpoints.append(relu_checkpoint_prefix_2 + '_run' + str(best_runs_20runs_30epochs[i]) + '_epoch25.pth')
        #
        # for i in range(len(best_runs_5runs_30epochs)):
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i])+'_epoch0.pth')
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch5.pth')
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch10.pth')
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch15.pth')
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch20.pth')
        #     checkpoints.append(relu_checkpoint_prefix_3 + '_run' + str(best_runs_5runs_30epochs[i]) + '_epoch25.pth')

        checkpoints, df = getReLUModels()

        a_values = []
        b_values = []
        ab_ratios = []
        u_values = []
        weights_a = []
        weights_b = []
        biases_input = []
        weights_u = []
        biases_u = []


        for i in range(len(checkpoints)):
            checkpt = torch.load(checkpoints[i])
            model = VanillaReLURNN(input_size, hidden_size, num_layers=1, batch_size=1, output_size=num_classes,
                                   output_activation=output_activation)

            model.load_state_dict(checkpt['model_state_dict'])
            model.to(device)

            for param in model.rnn.named_parameters():
                # if 'weight' in param[0]:
                #     print('parameter name = ',param[0])
                #     print('weight = ',param[1])
                #     print(param[1][0][0].item())
                #     # print(param[1][0][1])
                # elif 'bias' in param[0]:
                #     print('parameter name = ',param[0])
                #     print('bias = ',param[1])
                #     print(param[1][0].item())
                if 'weight_ih' in param[0]:
                    # print(param[0])
                    # print(param[1][0][0].item())
                    # print(param[1][0][1].item())
                    weights_a.append(param[1][0][0].item())
                    weights_b.append(param[1][0][1].item())
                elif 'weight_hh' in param[0]:
                    weights_u.append(param[1][0][0].item())
                elif 'bias_ih' in param[0]:
                    biases_input.append(param[1][0].item())
                elif 'bias_hh' in param[0]:
                    biases_u.append(param[1][0].item())


                # if 'weight_ih_10' in param[0]:


        print('len weights a = ',len(weights_a))
        print('len weights b = ',len(weights_b))
        print('len weights u = ',len(weights_u))
        print('len biases input = ',len(biases_input))
        print('len biases u = ',len(biases_u))

        for i in range(len(weights_a)):
            ab_ratio = 0
            a_value = weights_a[i]+biases_u[i]+biases_input[i]
            b_value = weights_b[i]+biases_u[i]+biases_input[i]
            ab_ratio=a_value/b_value
            ab_ratios.append(ab_ratio)
            a_values.append(a_value)
            b_values.append(b_value)
            u_values.append(weights_u[i])


        print(ab_ratios)
        print(u_values)
        plt.subplots()
        plt.hist(ab_ratios, bins=100)
        plt.savefig(relu_prefix_2+'INDICATORS_histogram_ab_ratio.png')
        plt.show()
        plt.subplots()
        plt.hist(u_values, bins=100)
        plt.savefig(relu_prefix_2+'INDICATORS_histogram_u_values.png')
        plt.show()

        fpfs = df['average first point of failure (2000 tokens)']
        val_losses = df['avg validation losses']
        log_val_losses = df['log of avg validation losses']
        neg_log_val_losses = df['log of inverse avg validation losses']
        
        res_ab_ratio_val_loss = stats.linregress(ab_ratios,val_losses)
        res_ab_ratio_log_val_loss = stats.linregress(ab_ratios,log_val_losses)
        res_ab_ratio_neg_log_val_loss=stats.linregress(ab_ratios,neg_log_val_losses)
        res_ab_ratio_fpfs = stats.linregress(ab_ratios,fpfs)
        
        print('AB_RATIO LINEAR REGRESSION WITH VAL LOSS')
        print('r value = ',res_ab_ratio_val_loss.rvalue)
        print('p value = ',res_ab_ratio_val_loss.pvalue)
        print('slope = ',res_ab_ratio_val_loss.slope)
        print('intercept = ',res_ab_ratio_val_loss.intercept)
        print('AB_RATIO LINEAR REGRESSION WITH LOG VAL LOSS')
        print('r value = ', res_ab_ratio_log_val_loss.rvalue)
        print('p value = ', res_ab_ratio_log_val_loss.pvalue)
        print('slope = ', res_ab_ratio_log_val_loss.slope)
        print('intercept = ', res_ab_ratio_log_val_loss.intercept)
        print('AB_RATIO LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
        print('r value = ', res_ab_ratio_neg_log_val_loss.rvalue)
        print('p value = ', res_ab_ratio_neg_log_val_loss.pvalue)
        print('slope = ', res_ab_ratio_neg_log_val_loss.slope)
        print('intercept = ', res_ab_ratio_neg_log_val_loss.intercept)
        print('AB_RATIO LINEAR REGRESSION WITH AVERAGE FPF')
        print('r value = ', res_ab_ratio_fpfs.rvalue)
        print('p value = ', res_ab_ratio_fpfs.pvalue)
        print('slope = ', res_ab_ratio_fpfs.slope)
        print('intercept = ', res_ab_ratio_fpfs.intercept)


        u_values_dev = []
        ab_ratios_dev = []

        for i in range(len(ab_ratios)):
            u_dev = abs(u_values[i]-1)
            u_values_dev.append(u_dev)
            ab_dev = abs(ab_ratios[i]--1)
            ab_ratios_dev.append(ab_dev)

        res_u_value_val_loss = stats.linregress(u_values, val_losses)
        res_u_value_log_val_loss = stats.linregress(u_values, log_val_losses)
        res_u_value_neg_log_val_loss = stats.linregress(u_values, neg_log_val_losses)
        res_u_value_fpfs = stats.linregress(u_values, fpfs)

        print('U_VALUE LINEAR REGRESSION WITH VAL LOSS')
        print('r value = ', res_u_value_val_loss.rvalue)
        print('p value = ', res_u_value_val_loss.pvalue)
        print('slope = ', res_u_value_val_loss.slope)
        print('intercept = ', res_u_value_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH LOG VAL LOSS')
        print('r value = ', res_u_value_log_val_loss.rvalue)
        print('p value = ', res_u_value_log_val_loss.pvalue)
        print('slope = ', res_u_value_log_val_loss.slope)
        print('intercept = ', res_u_value_log_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
        print('r value = ', res_u_value_neg_log_val_loss.rvalue)
        print('p value = ', res_u_value_neg_log_val_loss.pvalue)
        print('slope = ', res_u_value_neg_log_val_loss.slope)
        print('intercept = ', res_u_value_neg_log_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH AVERAGE FPF')
        print('r value = ', res_u_value_fpfs.rvalue)
        print('p value = ', res_u_value_fpfs.pvalue)
        print('slope = ', res_u_value_fpfs.slope)
        print('intercept = ', res_u_value_fpfs.intercept)
        
        
        plt.subplots()
        plt.plot(ab_ratios, val_losses, 'o', label='Models')
        plt.ylabel('Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(-0.7)
        # plt.plot(val_losses,ab_ratios,'o',label='Models')
        # # plt.plot(val_losses,res_ab_ratio_val_loss.intercept+res_ab_ratio_val_loss.slope*val_losses,'r',label='Fitted Line')
        # plt.xlabel('Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2+'INDICATORS_linear_regression_ab_values_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(ab_ratios, log_val_losses, 'o', label='Models')
        plt.ylabel('Log Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(-0.7)
        # plt.plot(log_val_losses, ab_ratios, 'o', label='Models')
        # # plt.plot(log_val_losses, res_ab_ratio_log_val_loss.intercept + res_ab_ratio_log_val_loss.slope * log_val_losses, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Log Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_log_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(ab_ratios, neg_log_val_losses, 'o', label='Models')
        plt.ylabel('Negative Log Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(-0.7)
        # plt.plot(neg_log_val_losses, ab_ratios, 'o', label='Models')
        # # plt.plot(neg_log_val_losses, res_ab_ratio_neg_log_val_loss.intercept + res_ab_ratio_neg_log_val_loss.slope * neg_log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Negative Log Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_neg_log_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(ab_ratios, fpfs, 'o', label='Models')
        plt.ylabel('Average FPF')
        plt.xlabel('AB Ratio')
        plt.xlim(-0.7)
        # plt.plot(fpfs, ab_ratios, 'o', label='Models')
        # # plt.plot(fpfs, res_ab_ratio_val_loss.intercept + res_ab_ratio_val_loss.slope * fpfs, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Average FPF')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_val_losses.png')
        plt.close()
        
        ####################################

        plt.subplots()
        plt.plot(u_values, val_losses, 'o', label='Models')
        plt.ylabel('Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(1.1)
        # plt.plot(val_losses, u_values, 'o', label='Models')
        # plt.plot(val_losses, res_u_value_val_loss.intercept + res_u_value_val_loss.slope * val_losses, 'r',
        #          label='Fitted Line')
        # plt.xlabel('Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(u_values, log_val_losses, 'o', label='Models')
        plt.ylabel('Log Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(1.1)
        # plt.plot(log_val_losses, u_values, 'o', label='Models')
        # # plt.plot(log_val_losses, res_u_value_log_val_loss.intercept + res_u_value_log_val_loss.slope * log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Log Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_log_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(u_values, neg_log_val_losses, 'o', label='Models')
        plt.ylabel('Negative Log Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(1.1)
        # plt.plot(neg_log_val_losses, u_values, 'o', label='Models')
        # # plt.plot(neg_log_val_losses,
        # #          res_u_value_neg_log_val_loss.intercept + res_u_value_neg_log_val_loss.slope * neg_log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Negative Log Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_neg_log_val_losses.png')
        plt.close()


        plt.subplots()
        plt.plot(u_values, fpfs, 'o', label='Models')
        plt.ylabel('Average FPF')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(1.1)
        # plt.plot(fpfs, u_values, 'o', label='Models')
        # # plt.plot(fpfs, res_u_value_fpfs.intercept + res_u_value_fpfs.slope * fpfs, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Average FPF')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_val_losses.png')
        plt.close()
        
        
        ######################
        res_u_value_val_loss = stats.linregress(u_values, val_losses)
        res_u_value_log_val_loss = stats.linregress(u_values, log_val_losses)
        res_u_value_neg_log_val_loss = stats.linregress(u_values, neg_log_val_losses)
        res_u_value_fpfs = stats.linregress(u_values, fpfs)

        print('U_VALUE LINEAR REGRESSION WITH VAL LOSS')
        print('r value = ', res_u_value_val_loss.rvalue)
        print('p value = ', res_u_value_val_loss.pvalue)
        print('slope = ', res_u_value_val_loss.slope)
        print('intercept = ', res_u_value_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH LOG VAL LOSS')
        print('r value = ', res_u_value_log_val_loss.rvalue)
        print('p value = ', res_u_value_log_val_loss.pvalue)
        print('slope = ', res_u_value_log_val_loss.slope)
        print('intercept = ', res_u_value_log_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
        print('r value = ', res_u_value_neg_log_val_loss.rvalue)
        print('p value = ', res_u_value_neg_log_val_loss.pvalue)
        print('slope = ', res_u_value_neg_log_val_loss.slope)
        print('intercept = ', res_u_value_neg_log_val_loss.intercept)
        print('U_VALUE LINEAR REGRESSION WITH AVERAGE FPF')
        print('r value = ', res_u_value_fpfs.rvalue)
        print('p value = ', res_u_value_fpfs.pvalue)
        print('slope = ', res_u_value_fpfs.slope)
        print('intercept = ', res_u_value_fpfs.intercept)

        plt.subplots()
        plt.plot(ab_ratios_dev, val_losses, 'o', label='Models')
        plt.ylabel('Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(0.3)
        # plt.plot(val_losses, ab_ratios_dev, 'o', label='Models')
        # # plt.plot(val_losses,res_ab_ratio_val_loss.intercept+res_ab_ratio_val_loss.slope*val_losses,'r',label='Fitted Line')
        # plt.xlabel('Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_dev_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(ab_ratios_dev, log_val_losses, 'o', label='Models')
        plt.ylabel('Log Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(0.3)
        # plt.plot(log_val_losses, ab_ratios_dev, 'o', label='Models')
        # # plt.plot(log_val_losses, res_ab_ratio_log_val_loss.intercept + res_ab_ratio_log_val_loss.slope * log_val_losses, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Log Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_dev_log_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(ab_ratios_dev, neg_log_val_losses, 'o', label='Models')
        plt.ylabel('Negative Log Validation Loss')
        plt.xlabel('AB Ratio')
        plt.xlim(0.3)
        # plt.plot(neg_log_val_losses, ab_ratios_dev, 'o', label='Models')
        # # plt.plot(neg_log_val_losses, res_ab_ratio_neg_log_val_loss.intercept + res_ab_ratio_neg_log_val_loss.slope * neg_log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Negative Log Validation Loss')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_dev_neg_log_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(ab_ratios_dev, fpfs, 'o', label='Models')
        plt.ylabel('Average FPF')
        plt.xlabel('AB Ratio')
        plt.xlim(0.3)
        # plt.plot(fpfs, ab_ratios_dev, 'o', label='Models')
        # # plt.plot(fpfs, res_ab_ratio_val_loss.intercept + res_ab_ratio_val_loss.slope * fpfs, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Average FPF')
        # plt.ylabel('AB Ratio')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_dev_val_losses.png')
        plt.close()

        ####################################

        plt.subplots()
        plt.plot(u_values_dev, val_losses, 'o', label='Models')
        plt.ylabel('Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(0.1)
        # plt.plot(val_losses, u_values_dev, 'o', label='Models')
        # # plt.plot(val_losses, res_u_values_dev_val_loss.intercept + res_u_values_dev_val_loss.slope * val_losses, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_devs_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(u_values_dev, log_val_losses, 'o', label='Models')
        plt.ylabel('Log Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(0.1)
        # plt.plot(log_val_losses, u_values_dev, 'o', label='Models')
        # # plt.plot(log_val_losses, res_u_values_dev_log_val_loss.intercept + res_u_values_dev_log_val_loss.slope * log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Log Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_devs_log_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(u_values_dev, neg_log_val_losses, 'o', label='Models')
        plt.ylabel('Negative Log Validation Loss')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(0.1)
        # plt.plot(neg_log_val_losses, u_values_dev, 'o', label='Models')
        # # plt.plot(neg_log_val_losses,
        # #          res_u_values_dev_neg_log_val_loss.intercept + res_u_values_dev_neg_log_val_loss.slope * neg_log_val_losses,
        # #          'r',
        # #          label='Fitted Line')
        # plt.xlabel('Negative Log Validation Loss')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_devs_neg_log_val_losses.png')
        plt.close()

        plt.subplots()
        plt.plot(u_values_dev, fpfs, 'o', label='Models')
        plt.ylabel('Average FPF')
        plt.xlabel('Recurrent Weight U')
        plt.xlim(0.1)
        # plt.plot(fpfs, u_values_dev, 'o', label='Models')
        # # plt.plot(fpfs, res_u_values_dev_fpfs.intercept + res_u_values_dev_fpfs.slope * fpfs, 'r',
        # #          label='Fitted Line')
        # plt.xlabel('Average FPF')
        # plt.ylabel('Recurrent Weight U')
        plt.legend()
        plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_devs_val_losses.png')
        plt.close()
        #
        # filename = relu_prefix_2 + 'INDICATORS_values_devs_dev.txt'
        # with open(filename, 'w') as f:
        #     f.write('')
        #
        # with open(filename, 'a') as f:
        #     f.write('AB_RATIO LINEAR REGRESSION WITH VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_ab_ratio_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_ab_ratio_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_ab_ratio_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_ab_ratio_val_loss.intercept) + '\n')
        #     f.write('AB_RATIO LINEAR REGRESSION WITH LOG VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_ab_ratio_log_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_ab_ratio_log_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_ab_ratio_log_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_ab_ratio_log_val_loss.intercept) + '\n')
        #     f.write('AB_RATIO LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_ab_ratio_neg_log_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_ab_ratio_neg_log_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_ab_ratio_neg_log_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_ab_ratio_neg_log_val_loss.intercept) + '\n')
        #     f.write('AB_RATIO LINEAR REGRESSION WITH AVERAGE FPF' + '\n')
        #     f.write('r value = ' + str(res_ab_ratio_fpfs.rvalue) + '\n')
        #     f.write('p value = ' + str(res_ab_ratio_fpfs.pvalue) + '\n')
        #     f.write('slope = ' + str(res_ab_ratio_fpfs.slope) + '\n')
        #     f.write('intercept = ' + str(res_ab_ratio_fpfs.intercept) + '\n')
        #
        #     f.write('U_values_dev LINEAR REGRESSION WITH VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_u_values_devs_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_u_values_dev_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_u_values_dev_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_u_values_dev_val_loss.intercept) + '\n')
        #     f.write('U_values_dev LINEAR REGRESSION WITH LOG VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_u_values_dev_log_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_u_values_dev_log_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_u_values_dev_log_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_u_values_dev_log_val_loss.intercept) + '\n')
        #     f.write('U_values_dev LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS' + '\n')
        #     f.write('r value = ' + str(res_u_values_dev_neg_log_val_loss.rvalue) + '\n')
        #     f.write('p value = ' + str(res_u_values_dev_neg_log_val_loss.pvalue) + '\n')
        #     f.write('slope = ' + str(res_u_values_dev_neg_log_val_loss.slope) + '\n')
        #     f.write('intercept = ' + str(res_u_values_dev_neg_log_val_loss.intercept) + '\n')
        #     f.write('U_values_dev LINEAR REGRESSION WITH AVERAGE FPF' + '\n')
        #     f.write('r value = ' + str(res_u_values_dev_fpfs.rvalue) + '\n')
        #     f.write('p value = ' + str(res_u_values_dev_fpfs.pvalue) + '\n')
        #     f.write('slope = ' + str(res_u_values_dev_fpfs.slope) + '\n')
        #     f.write('intercept = ' + str(res_u_values_dev_fpfs.intercept) + '\n')
        # ##############################

        filename = relu_prefix_2+'INDICATORS_VALUES.txt'
        with open(filename,'w') as f:
            f.write('')
        
        with open(filename,'a') as f:
            f.write('AB_RATIO LINEAR REGRESSION WITH VAL LOSS'+'\n')
            f.write('r value = '+str(res_ab_ratio_val_loss.rvalue)+'\n')
            f.write('p value = '+str(res_ab_ratio_val_loss.pvalue)+'\n')
            f.write('slope = '+ str(res_ab_ratio_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_ab_ratio_val_loss.intercept)+ '\n')
            f.write('AB_RATIO LINEAR REGRESSION WITH LOG VAL LOSS'+ '\n')
            f.write('r value = '+ str(res_ab_ratio_log_val_loss.rvalue)+ '\n')
            f.write('p value = '+ str(res_ab_ratio_log_val_loss.pvalue)+ '\n')
            f.write('slope = '+ str(res_ab_ratio_log_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_ab_ratio_log_val_loss.intercept)+ '\n')
            f.write('AB_RATIO LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS'+ '\n')
            f.write('r value = '+ str(res_ab_ratio_neg_log_val_loss.rvalue)+ '\n')
            f.write('p value = '+ str(res_ab_ratio_neg_log_val_loss.pvalue)+ '\n')
            f.write('slope = '+ str(res_ab_ratio_neg_log_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_ab_ratio_neg_log_val_loss.intercept)+ '\n')
            f.write('AB_RATIO LINEAR REGRESSION WITH AVERAGE FPF'+ '\n')
            f.write('r value = '+ str(res_ab_ratio_fpfs.rvalue)+ '\n')
            f.write('p value = '+ str(res_ab_ratio_fpfs.pvalue)+ '\n')
            f.write('slope = '+ str(res_ab_ratio_fpfs.slope)+ '\n')
            f.write('intercept = '+ str(res_ab_ratio_fpfs.intercept)+ '\n')

            f.write('U_VALUE LINEAR REGRESSION WITH VAL LOSS'+ '\n')
            f.write('r value = '+ str(res_u_value_val_loss.rvalue)+ '\n')
            f.write('p value = '+ str(res_u_value_val_loss.pvalue)+ '\n')
            f.write('slope = '+ str(res_u_value_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_u_value_val_loss.intercept)+ '\n')
            f.write('U_VALUE LINEAR REGRESSION WITH LOG VAL LOSS'+ '\n')
            f.write('r value = '+ str(res_u_value_log_val_loss.rvalue)+ '\n')
            f.write('p value = '+ str(res_u_value_log_val_loss.pvalue)+ '\n')
            f.write('slope = '+ str(res_u_value_log_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_u_value_log_val_loss.intercept)+ '\n')
            f.write('U_VALUE LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS'+ '\n')
            f.write('r value = '+ str(res_u_value_neg_log_val_loss.rvalue)+ '\n')
            f.write('p value = '+ str(res_u_value_neg_log_val_loss.pvalue)+ '\n')
            f.write('slope = '+ str(res_u_value_neg_log_val_loss.slope)+ '\n')
            f.write('intercept = '+ str(res_u_value_neg_log_val_loss.intercept)+ '\n')
            f.write('U_VALUE LINEAR REGRESSION WITH AVERAGE FPF'+ '\n')
            f.write('r value = '+ str(res_u_value_fpfs.rvalue)+ '\n')
            f.write('p value = '+ str(res_u_value_fpfs.pvalue)+ '\n')
            f.write('slope = '+ str(res_u_value_fpfs.slope)+ '\n')
            f.write('intercept = '+ str(res_u_value_fpfs.intercept)+ '\n')

        if 'weights_a' not in df.columns:
            df['weights_a'] = weights_a
        if 'weights_b' not in df.columns:
            df['weights_b'] = weights_b
        if 'biases_input' not in df.columns:
            df['biases_input']=biases_input
        if 'biases_u' not in df.columns:
            df['biases_u']=biases_u
        if 'a_values' not in df.columns:
            df['a_values'] = a_values
        if 'b_values' not in df.columns:
            df['b_values']=b_values
        if 'ab_ratios' not in df.columns:
            df['ab_ratios']=ab_ratios
        if 'weights_u' not in df.columns:
            df['weights_u'] = weights_u
        if 'u_values' not in df.columns:
            df['u_values'] = u_values

        writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

        df.to_excel(writer, index=False)
        writer.save()

        # relu_dfs_1 = read_sheets(10, relu_excel_path)  # 10 runs, 50 epochs
        # relu_dfs_2 = read_sheets(20, relu_excel_path_2)  # 20 runs, 30 epochs
        # relu_dfs_3 = read_sheets(5, relu_excel_path_3)  # 5 runs, 30 epochs
        #
        # relu_dfs = []
        # relu_dfs.append(relu_dfs_1[4])
        # relu_dfs.append(relu_dfs_1[5])
        # relu_dfs.append(relu_dfs_1[8])
        # relu_dfs.append(relu_dfs_2[4])
        # relu_dfs.append(relu_dfs_2[5])
        # relu_dfs.append(relu_dfs_2[8])
        # relu_dfs.append(relu_dfs_2[10])
        # relu_dfs.append(relu_dfs_2[12])
        # relu_dfs.append(relu_dfs_3[0])
        # relu_dfs.append(relu_dfs_3[3])
        #
        # print(len(relu_dfs))

    # elif model_type=='LSTM':
    #     pass

    # return df


# def createLinearRegression(df):
#     fpfs = df['average first point of failure (2000 tokens)']
#     ab_ratios = df['ab_ratios']
#     u_values = df['u_values']
#     val_losses = df['avg validation losses']
#     log_val_losses = df['log of avg validation losses']
#     neg_log_val_losses = df['log of inverse avg validation losses']
    

extractModelIndicators()




