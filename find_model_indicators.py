import numpy
import torch
import torch.nn as nn
import pandas as pd
# import matplotlib as mpl
# # mpl.use('tkagg')
import matplotlib.pyplot as plt
import argparse
from models_batch import VanillaLSTM, VanillaReLURNN
import os
import scipy
from scipy import stats
from mpl_toolkits import mplot3d
import numpy as np
import math
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import scipy
import pandas.util.testing as tm
from matplotlib.patches import Rectangle



parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, help='type of model, LSTM, ReLU')
# parser.add_argument('--')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device==torch.device("cuda"):
    relu_excel_path = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
    relu_excel_path_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
    relu_excel_path_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs.xlsx'

    relu_prefix_1 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    relu_prefix_2 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
    relu_prefix_3 = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'

else:
    relu_excel_path = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_NEW.xlsx'
    relu_excel_path_2 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_20runs.xlsx'
    relu_excel_path_3 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_5runs.xlsx'

    relu_prefix_1 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    relu_prefix_2 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
    relu_prefix_3 = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/5_runs/shuffle_True/'

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



#only done once to insert the indicators into the excel sheet, after that, excel sheet can be used for plotting
#without the need to load and inspect the final models
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
        u_values_dev = []
        ab_ratios_dev = []
        fpfs = df['average first point of failure (2000 tokens)']
        val_losses = df['avg validation losses']
        log_val_losses = df['log of avg validation losses']
        neg_log_val_losses = df['log of inverse avg validation losses']





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
            u_dev = abs(weights_u[i] - 1)

            ab_dev = abs(ab_ratio - -1)


            ab_ratios_dev.append(ab_dev)
            u_values_dev.append(u_dev)

            ab_ratios.append(ab_ratio)
            a_values.append(a_value)
            b_values.append(b_value)
            u_values.append(weights_u[i])
            # if ab_dev<=0.3 and u_dev<=0.1:
            #     ab_ratios_dev.append(ab_dev)
            #     u_values_dev.append(u_dev)
            #
            #     ab_ratios.append(ab_ratio)
            #     a_values.append(a_value)
            #     b_values.append(b_value)
            #     u_values.append(weights_u[i])
            # else:
            #     del fpfs[i]
            #     del val_losses[i]
            #     del log_val_losses[i]
            #     del neg_log_val_losses[i]

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
        if 'ab_ratios_dev' not in df.columns:
            df['ab_ratios_dev'] = ab_ratios_dev
        if 'u_dev' not in df.columns:
            df['u_values_dev'] = u_values_dev

        writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

        df.to_excel(writer, index=False)
        writer.save()

def plotModelIndicators():
    _,df = getReLUModels()
    ab_ratios = df['ab_ratios']
    u_values = df['u_values']
    ab_ratios_dev = df['ab_ratios_dev']
    u_values_dev = df['u_values_dev']
    fpfs = df['average first point of failure (2000 tokens)']
    val_losses = df['avg validation losses']
    log_val_losses = df['log of avg validation losses']
    neg_log_val_losses = df['log of inverse avg validation losses']


    for i in range(len(ab_ratios)):
        ab_dev = ab_ratios_dev[i]
        u_dev = u_values_dev[i]
        if ab_dev > 0.3 and u_dev > 0.1:
            del ab_ratios_dev[i]
            del u_values_dev[i]
            del ab_ratios[i]
            del u_values[i]
            del fpfs[i]
            del val_losses[i]
            del log_val_losses[i]
            del neg_log_val_losses[i]


    # print('ab_ratios = ',ab_ratios)
    print('len(ab_ratios) = ',len(ab_ratios))
    print('len(ab_ratios_dev) = ', len(ab_ratios_dev))
    # print('u_values = ',u_values)
    print('len(u_values) = ',len(u_values))

# def other():
    plt.subplots()
    plt.hist(ab_ratios, bins=100)
    plt.savefig(relu_prefix_2+'INDICATORS_histogram_ab_ratio.png')
    # plt.show()
    plt.subplots()
    plt.hist(u_values, bins=100)
    plt.savefig(relu_prefix_2+'INDICATORS_histogram_u_values.png')
    # plt.show()

    # # fpfs = df['average first point of failure (2000 tokens)']
    # # val_losses = df['avg validation losses']
    # # log_val_losses = df['log of avg validation losses']
    # # neg_log_val_losses = df['log of inverse avg validation losses']
    # res_ab_ratio_val_loss = stats.linregress(ab_ratios,val_losses)
    # res_ab_ratio_log_val_loss = stats.linregress(ab_ratios,log_val_losses)
    # res_ab_ratio_neg_log_val_loss=stats.linregress(ab_ratios,neg_log_val_losses)
    # res_ab_ratio_fpfs = stats.linregress(ab_ratios,fpfs)
    #
    # print('AB_RATIO LINEAR REGRESSION WITH VAL LOSS')
    # print('r value = ',res_ab_ratio_val_loss.rvalue)
    # print('p value = ',res_ab_ratio_val_loss.pvalue)
    # print('slope = ',res_ab_ratio_val_loss.slope)
    # print('intercept = ',res_ab_ratio_val_loss.intercept)
    # print('AB_RATIO LINEAR REGRESSION WITH LOG VAL LOSS')
    # print('r value = ', res_ab_ratio_log_val_loss.rvalue)
    # print('p value = ', res_ab_ratio_log_val_loss.pvalue)
    # print('slope = ', res_ab_ratio_log_val_loss.slope)
    # print('intercept = ', res_ab_ratio_log_val_loss.intercept)
    # print('AB_RATIO LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
    # print('r value = ', res_ab_ratio_neg_log_val_loss.rvalue)
    # print('p value = ', res_ab_ratio_neg_log_val_loss.pvalue)
    # print('slope = ', res_ab_ratio_neg_log_val_loss.slope)
    # print('intercept = ', res_ab_ratio_neg_log_val_loss.intercept)
    # print('AB_RATIO LINEAR REGRESSION WITH AVERAGE FPF')
    # print('r value = ', res_ab_ratio_fpfs.rvalue)
    # print('p value = ', res_ab_ratio_fpfs.pvalue)
    # print('slope = ', res_ab_ratio_fpfs.slope)
    # print('intercept = ', res_ab_ratio_fpfs.intercept)
    #
    #
    # # u_values_dev = []
    # # ab_ratios_dev = []
    # #
    # # for i in range(len(ab_ratios)):
    # #     u_dev = abs(u_values[i]-1)
    # #     u_values_dev.append(u_dev)
    # #     ab_dev = abs(ab_ratios[i]--1)
    # #     ab_ratios_dev.append(ab_dev)
    #
    # res_u_value_val_loss = stats.linregress(u_values, val_losses)
    # res_u_value_log_val_loss = stats.linregress(u_values, log_val_losses)
    # res_u_value_neg_log_val_loss = stats.linregress(u_values, neg_log_val_losses)
    # res_u_value_fpfs = stats.linregress(u_values, fpfs)
    #
    # print('U_VALUE LINEAR REGRESSION WITH VAL LOSS')
    # print('r value = ', res_u_value_val_loss.rvalue)
    # print('p value = ', res_u_value_val_loss.pvalue)
    # print('slope = ', res_u_value_val_loss.slope)
    # print('intercept = ', res_u_value_val_loss.intercept)
    # print('U_VALUE LINEAR REGRESSION WITH LOG VAL LOSS')
    # print('r value = ', res_u_value_log_val_loss.rvalue)
    # print('p value = ', res_u_value_log_val_loss.pvalue)
    # print('slope = ', res_u_value_log_val_loss.slope)
    # print('intercept = ', res_u_value_log_val_loss.intercept)
    # print('U_VALUE LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
    # print('r value = ', res_u_value_neg_log_val_loss.rvalue)
    # print('p value = ', res_u_value_neg_log_val_loss.pvalue)
    # print('slope = ', res_u_value_neg_log_val_loss.slope)
    # print('intercept = ', res_u_value_neg_log_val_loss.intercept)
    # print('U_VALUE LINEAR REGRESSION WITH AVERAGE FPF')
    # print('r value = ', res_u_value_fpfs.rvalue)
    # print('p value = ', res_u_value_fpfs.pvalue)
    # print('slope = ', res_u_value_fpfs.slope)
    # print('intercept = ', res_u_value_fpfs.intercept)
    #
    #
    # plt.subplots()
    # plt.plot(ab_ratios, val_losses, 'o', label='Models')
    # plt.ylabel('Validation Loss')
    # plt.xlabel('AB Ratio')
    # # plt.xlim(-0.7)
    # # plt.plot(val_losses,ab_ratios,'o',label='Models')
    # # # plt.plot(val_losses,res_ab_ratio_val_loss.intercept+res_ab_ratio_val_loss.slope*val_losses,'r',label='Fitted Line')
    # # plt.xlabel('Validation Loss')
    # # plt.ylabel('AB Ratio')
    # plt.legend()
    # plt.savefig(relu_prefix_2+'INDICATORS_linear_regression_ab_values_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(ab_ratios, log_val_losses, 'o', label='Models')
    # plt.ylabel('Log Validation Loss')
    # plt.xlabel('AB Ratio')
    # # plt.xlim(-0.7)
    # # plt.plot(log_val_losses, ab_ratios, 'o', label='Models')
    # # # plt.plot(log_val_losses, res_ab_ratio_log_val_loss.intercept + res_ab_ratio_log_val_loss.slope * log_val_losses, 'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Log Validation Loss')
    # # plt.ylabel('AB Ratio')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_log_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(ab_ratios, neg_log_val_losses, 'o', label='Models')
    # plt.ylabel('Negative Log Validation Loss')
    # plt.xlabel('AB Ratio')
    # # plt.xlim(-0.7)
    # # plt.plot(neg_log_val_losses, ab_ratios, 'o', label='Models')
    # # # plt.plot(neg_log_val_losses, res_ab_ratio_neg_log_val_loss.intercept + res_ab_ratio_neg_log_val_loss.slope * neg_log_val_losses,
    # # #          'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Negative Log Validation Loss')
    # # plt.ylabel('AB Ratio')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_neg_log_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(ab_ratios, fpfs, 'o', label='Models')
    # plt.ylabel('Average FPF')
    # plt.xlabel('AB Ratio')
    # # plt.xlim(-0.7)
    # # plt.plot(fpfs, ab_ratios, 'o', label='Models')
    # # # plt.plot(fpfs, res_ab_ratio_val_loss.intercept + res_ab_ratio_val_loss.slope * fpfs, 'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Average FPF')
    # # plt.ylabel('AB Ratio')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_ab_values_val_losses.png')
    # plt.close()
    #
    # ####################################
    #
    # plt.subplots()
    # plt.plot(u_values, val_losses, 'o', label='Models')
    # plt.ylabel('Validation Loss')
    # plt.xlabel('Recurrent Weight U')
    # # plt.xlim(1.1)
    # # plt.plot(val_losses, u_values, 'o', label='Models')
    # # plt.plot(val_losses, res_u_value_val_loss.intercept + res_u_value_val_loss.slope * val_losses, 'r',
    # #          label='Fitted Line')
    # # plt.xlabel('Validation Loss')
    # # plt.ylabel('Recurrent Weight U')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(u_values, log_val_losses, 'o', label='Models')
    # plt.ylabel('Log Validation Loss')
    # plt.xlabel('Recurrent Weight U')
    # # plt.xlim(1.1)
    # # plt.plot(log_val_losses, u_values, 'o', label='Models')
    # # # plt.plot(log_val_losses, res_u_value_log_val_loss.intercept + res_u_value_log_val_loss.slope * log_val_losses,
    # # #          'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Log Validation Loss')
    # # plt.ylabel('Recurrent Weight U')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_log_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(u_values, neg_log_val_losses, 'o', label='Models')
    # plt.ylabel('Negative Log Validation Loss')
    # plt.xlabel('Recurrent Weight U')
    # # plt.xlim(1.1)
    # # plt.plot(neg_log_val_losses, u_values, 'o', label='Models')
    # # # plt.plot(neg_log_val_losses,
    # # #          res_u_value_neg_log_val_loss.intercept + res_u_value_neg_log_val_loss.slope * neg_log_val_losses,
    # # #          'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Negative Log Validation Loss')
    # # plt.ylabel('Recurrent Weight U')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_neg_log_val_losses.png')
    # plt.close()
    #
    #
    # plt.subplots()
    # plt.plot(u_values, fpfs, 'o', label='Models')
    # plt.ylabel('Average FPF')
    # plt.xlabel('Recurrent Weight U')
    # # plt.xlim(1.1)
    # # plt.plot(fpfs, u_values, 'o', label='Models')
    # # # plt.plot(fpfs, res_u_value_fpfs.intercept + res_u_value_fpfs.slope * fpfs, 'r',
    # # #          label='Fitted Line')
    # # plt.xlabel('Average FPF')
    # # plt.ylabel('Recurrent Weight U')
    # plt.legend()
    # plt.savefig(relu_prefix_2 + 'INDICATORS_linear_regression_u_values_val_losses.png')
    # plt.close()


    ##########################################################################

    res_ab_ratio_dev_val_loss = stats.linregress(ab_ratios_dev, val_losses)
    res_ab_ratio_dev_log_val_loss = stats.linregress(ab_ratios_dev, log_val_losses)
    res_ab_ratio_dev_neg_log_val_loss = stats.linregress(ab_ratios_dev, neg_log_val_losses)
    res_ab_ratio_dev_fpfs = stats.linregress(ab_ratios_dev, fpfs)
    # res_ab_ratio_dev_val_loss = stats.linregress(val_losses, ab_ratios_dev)
    # res_ab_ratio_dev_log_val_loss = stats.linregress(log_val_losses, ab_ratios_dev)
    # res_ab_ratio_dev_neg_log_val_loss = stats.linregress(neg_log_val_losses, ab_ratios_dev)
    # res_ab_ratio_dev_fpfs = stats.linregress(fpfs, ab_ratios_dev)

    print('ab_ratio DEV LINEAR REGRESSION WITH VAL LOSS')
    print('r value = ', res_ab_ratio_dev_val_loss.rvalue)
    print('p value = ', res_ab_ratio_dev_val_loss.pvalue)
    print('slope = ', res_ab_ratio_dev_val_loss.slope)
    print('intercept = ', res_ab_ratio_dev_val_loss.intercept)
    print('ab_ratio DEV LINEAR REGRESSION WITH LOG VAL LOSS')
    print('r value = ', res_ab_ratio_dev_log_val_loss.rvalue)
    print('p value = ', res_ab_ratio_dev_log_val_loss.pvalue)
    print('slope = ', res_ab_ratio_dev_log_val_loss.slope)
    print('intercept = ', res_ab_ratio_dev_log_val_loss.intercept)
    print('ab_ratio DEV LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
    print('r value = ', res_ab_ratio_dev_neg_log_val_loss.rvalue)
    print('p value = ', res_ab_ratio_dev_neg_log_val_loss.pvalue)
    print('slope = ', res_ab_ratio_dev_neg_log_val_loss.slope)
    print('intercept = ', res_ab_ratio_dev_neg_log_val_loss.intercept)
    print('ab_ratio DEV LINEAR REGRESSION WITH AVERAGE FPF')
    print('r value = ', res_ab_ratio_dev_fpfs.rvalue)
    print('p value = ', res_ab_ratio_dev_fpfs.pvalue)
    print('slope = ', res_ab_ratio_dev_fpfs.slope)
    print('intercept = ', res_ab_ratio_dev_fpfs.intercept)
    
    
    ###########################################################################

    res_u_value_dev_val_loss = stats.linregress(u_values_dev, val_losses)
    res_u_value_dev_log_val_loss = stats.linregress(u_values_dev, log_val_losses)
    res_u_value_dev_neg_log_val_loss = stats.linregress(u_values_dev, neg_log_val_losses)
    res_u_value_dev_fpfs = stats.linregress(u_values_dev, fpfs)
    # res_u_value_dev_val_loss = stats.linregress(val_losses, u_values_dev)
    # res_u_value_dev_log_val_loss = stats.linregress(log_val_losses, u_values_dev)
    # res_u_value_dev_neg_log_val_loss = stats.linregress(neg_log_val_losses, u_values_dev)
    # res_u_value_dev_fpfs = stats.linregress(fpfs, u_values_dev)

    print('U_VALUE DEV LINEAR REGRESSION WITH VAL LOSS')
    print('r value = ', res_u_value_dev_val_loss.rvalue)
    print('p value = ', res_u_value_dev_val_loss.pvalue)
    print('slope = ', res_u_value_dev_val_loss.slope)
    print('intercept = ', res_u_value_dev_val_loss.intercept)
    print('U_VALUE DEV LINEAR REGRESSION WITH LOG VAL LOSS')
    print('r value = ', res_u_value_dev_log_val_loss.rvalue)
    print('p value = ', res_u_value_dev_log_val_loss.pvalue)
    print('slope = ', res_u_value_dev_log_val_loss.slope)
    print('intercept = ', res_u_value_dev_log_val_loss.intercept)
    print('U_VALUE DEV LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS')
    print('r value = ', res_u_value_dev_neg_log_val_loss.rvalue)
    print('p value = ', res_u_value_dev_neg_log_val_loss.pvalue)
    print('slope = ', res_u_value_dev_neg_log_val_loss.slope)
    print('intercept = ', res_u_value_dev_neg_log_val_loss.intercept)
    print('U_VALUE DEV LINEAR REGRESSION WITH AVERAGE FPF')
    print('r value = ', res_u_value_dev_fpfs.rvalue)
    print('p value = ', res_u_value_dev_fpfs.pvalue)
    print('slope = ', res_u_value_dev_fpfs.slope)
    print('intercept = ', res_u_value_dev_fpfs.intercept)

    # print('slope type = ',type(res_u_value_dev_neg_log_val_loss.slope))
    # print(type(u_values_dev))

    plt.subplots()
    plt.plot(ab_ratios_dev, val_losses, 'o', label='Models')
    plt.ylabel('Validation Loss')
    plt.xlabel('AB Ratio')
    # plt.xlim(0.3)
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
    # plt.xlim(0.3)
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
    plt.plot(ab_ratios_dev,(res_ab_ratio_dev_neg_log_val_loss.slope * ab_ratios_dev) + res_ab_ratio_dev_neg_log_val_loss.intercept,'r',label='Fitted Line')
    plt.ylabel('Negative Log Validation Loss')
    plt.xlabel('AB Ratio')
    # plt.xlim(0.3)
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
    plt.plot(ab_ratios_dev, res_ab_ratio_dev_fpfs.intercept+ res_ab_ratio_dev_fpfs.slope*ab_ratios_dev,'r', label='Fitted Line')
    plt.ylabel('Average FPF')
    plt.xlabel('AB Ratio')
    # plt.xlim(0.3)
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
    # plt.xlim(0.1)
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
    # plt.xlim(0.1)
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
    plt.plot(u_values_dev,(res_u_value_dev_neg_log_val_loss.slope * u_values_dev) + res_u_value_dev_neg_log_val_loss.intercept,'r',label='Fitted Line')

    plt.ylabel('Negative Log Validation Loss')
    plt.xlabel('Recurrent Weight U')

    # plt.xlim(0.1)
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
    plt.plot(u_values_dev,res_u_value_dev_fpfs.intercept+res_ab_ratio_dev_fpfs.slope*u_values_dev)
    plt.ylabel('Average FPF')
    plt.xlabel('Recurrent Weight U')
    # plt.xlim(0.1)
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

    # filename = relu_prefix_2+'INDICATORS_VALUES.txt'
    # with open(filename,'w') as f:
    #     f.write('')
    #
    # with open(filename,'a') as f:
    #     f.write('AB_RATIO LINEAR REGRESSION WITH VAL LOSS'+'\n')
    #     f.write('r value = '+str(res_ab_ratio_val_loss.rvalue)+'\n')
    #     f.write('p value = '+str(res_ab_ratio_val_loss.pvalue)+'\n')
    #     f.write('slope = '+ str(res_ab_ratio_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_ab_ratio_val_loss.intercept)+ '\n')
    #     f.write('AB_RATIO LINEAR REGRESSION WITH LOG VAL LOSS'+ '\n')
    #     f.write('r value = '+ str(res_ab_ratio_log_val_loss.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_ab_ratio_log_val_loss.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_ab_ratio_log_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_ab_ratio_log_val_loss.intercept)+ '\n')
    #     f.write('AB_RATIO LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS'+ '\n')
    #     f.write('r value = '+ str(res_ab_ratio_neg_log_val_loss.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_ab_ratio_neg_log_val_loss.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_ab_ratio_neg_log_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_ab_ratio_neg_log_val_loss.intercept)+ '\n')
    #     f.write('AB_RATIO LINEAR REGRESSION WITH AVERAGE FPF'+ '\n')
    #     f.write('r value = '+ str(res_ab_ratio_fpfs.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_ab_ratio_fpfs.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_ab_ratio_fpfs.slope)+ '\n')
    #     f.write('intercept = '+ str(res_ab_ratio_fpfs.intercept)+ '\n')
    #
    #     f.write('U_VALUE LINEAR REGRESSION WITH VAL LOSS'+ '\n')
    #     f.write('r value = '+ str(res_u_value_val_loss.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_u_value_val_loss.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_u_value_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_u_value_val_loss.intercept)+ '\n')
    #     f.write('U_VALUE LINEAR REGRESSION WITH LOG VAL LOSS'+ '\n')
    #     f.write('r value = '+ str(res_u_value_log_val_loss.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_u_value_log_val_loss.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_u_value_log_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_u_value_log_val_loss.intercept)+ '\n')
    #     f.write('U_VALUE LINEAR REGRESSION WITH NEGATIVE LOG VAL LOSS'+ '\n')
    #     f.write('r value = '+ str(res_u_value_neg_log_val_loss.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_u_value_neg_log_val_loss.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_u_value_neg_log_val_loss.slope)+ '\n')
    #     f.write('intercept = '+ str(res_u_value_neg_log_val_loss.intercept)+ '\n')
    #     f.write('U_VALUE LINEAR REGRESSION WITH AVERAGE FPF'+ '\n')
    #     f.write('r value = '+ str(res_u_value_fpfs.rvalue)+ '\n')
    #     f.write('p value = '+ str(res_u_value_fpfs.pvalue)+ '\n')
    #     f.write('slope = '+ str(res_u_value_fpfs.slope)+ '\n')
    #     f.write('intercept = '+ str(res_u_value_fpfs.intercept)+ '\n')

    #3D Plots
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z1 = neg_log_val_losses
    x = u_values_dev
    y = ab_ratios_dev
    ax.scatter(x,y,z1)
    ax.set_xlabel('deviation in U values')
    ax.set_ylabel('deviation in AB ratio')
    ax.set_zlabel('negative log validation loss')
    plt.savefig(relu_prefix_2+'INDICATORS_3D_deviations_neg_log_val_loss.png')
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z1 = fpfs
    x = u_values_dev
    y = ab_ratios_dev
    ax.scatter(x, y, z1)
    ax.set_xlabel('deviation in U values')
    ax.set_ylabel('deviation in AB ratio')
    ax.set_zlabel('Average FPFs')
    plt.savefig(relu_prefix_2+'INDICATORS_3D_deviations_FPF.png')
    plt.show()
    plt.close()

    euclidean_distances = []
    ab_ratios_dev_list = ab_ratios_dev.to_list()
    u_values_dev_list = u_values_dev.to_list()

    for i in range(len(ab_ratios_dev)):
        euclidean_distance = math.sqrt((ab_ratios_dev_list[i]**2)+(u_values_dev_list[i]**2))
        euclidean_distances.append(euclidean_distance)


        # print(ab_ratios_dev[i])
        # print(u_values_dev[i])
        # euclidean_distance = np.sqrt((ab_ratios_dev[i]*ab_ratios_dev[i])+(u_values_dev[i]*u_values_dev[i]))
        # euclidean_distance = np.sqrt(ab_ratios_dev[i]**2+u_values_dev[i]**2)
        # euclidean_distances.append(euclidean_distance)
    # print('euclidean_distances = ',euclidean_distances)
    # print(len(euclidean_distances))
    print('euclidean distances = ',euclidean_distances)

    res_euclidean_neg_log_val_loss = stats.linregress(euclidean_distances,neg_log_val_losses)
    res_euclidean_fpf = stats.linregress(euclidean_distances,fpfs)

    print('EUCLIDEAN_DISTANCE LINEAR REGRESSION NEGATIVE LOG VAL LOSSES')
    print('rvalue = ',res_euclidean_neg_log_val_loss.rvalue)
    print('pvalue = ',res_euclidean_neg_log_val_loss.pvalue)
    print('slope = ',res_euclidean_neg_log_val_loss.slope)
    print('intercept = ',res_euclidean_neg_log_val_loss.intercept)

    print('EUCLIDEAN_DISTANCE LINEAR REGRESSION FPFs')
    print('rvalue = ',res_euclidean_fpf.rvalue)
    print('pvalue = ',res_euclidean_fpf.pvalue)
    print('slope = ',res_euclidean_fpf.slope)
    print('intercept = ',res_euclidean_fpf.intercept)

    # print('slope type = ',type(res_euclidean_fpf.slope))



    plt.subplots()
    plt.plot(euclidean_distances,neg_log_val_losses, 'o', label='Models')
    plt.plot(euclidean_distances, res_euclidean_neg_log_val_loss.intercept + res_euclidean_neg_log_val_loss.slope*pd.Series(euclidean_distances) ,'r',label='Fitted Line')
    # plt.plot(euclidean_distances, res_euclidean_neg_log_val_loss.slope*euclidean_distances + res_euclidean_neg_log_val_loss.intercept)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('negative log validation loss')
    plt.legend()
    plt.savefig(relu_prefix_2+'INDICATORS_euclidean_distance_neg_log_val_losses.png')
    plt.show()
    plt.close()

    plt.subplots()
    plt.plot(euclidean_distances, fpfs, 'o', label='Models')
    plt.plot(euclidean_distances,res_euclidean_fpf.intercept + res_euclidean_fpf.slope*pd.Series(euclidean_distances),'r',label='Fitted Line')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Average FPFs')
    plt.legend()
    plt.savefig(relu_prefix_2 + 'INDICATORS_euclidean_distance_fpfs.png')
    plt.show()
    plt.close()
    #
    # neg_log_fpfs = -1*np.log(fpfs)
    # res_euclidean_neg_log_fpf = stats.linregress(euclidean_distances)

    plt.subplots()
    plt.hist(euclidean_distances, bins=100)
    plt.xlabel('Euclidean Distance')
    plt.savefig(relu_prefix_2+'INDICATORS_histogram_euclidean_distances.png')







# def createLinearRegression(df):
#     fpfs = df['average first point of failure (2000 tokens)']
#     ab_ratios = df['ab_ratios']
#     u_values = df['u_values']
#     val_losses = df['avg validation losses']
#     log_val_losses = df['log of avg validation losses']
#     neg_log_val_losses = df['log of inverse avg validation losses']
    

def plot_model_indicators_good_models():
    x_labels=[]
    y_labels = []
    for i in range(6):
        x_labels.append(round(-1.02+(0.02*i),2))

    for i in range(8):
        y_labels.append(0.999+(0.001*i))
    print(x_labels)
    print(y_labels)
    plt.rcParams.update({'font.size':11})
    _, df = getReLUModels()
    df1 = df.loc[df['avg validation losses']<1e-6]
    df2 = df.loc[df['average first point of failure (2000 tokens)']>700]

    df3 = df.nsmallest(5, 'avg validation losses')
    df4 = df.nlargest(5,'average first point of failure (2000 tokens)')
    # df5=df3.copy()
    # df5[df5==df4]
    print(df3)
    print(df3['average first point of failure (2000 tokens)'])
    print(df3['avg validation losses'])
    print(df4)
    print(df4['average first point of failure (2000 tokens)'])
    print(df4['avg validation losses'])
    # annotations = []
    # for i in range(len(df3)):
    #     annotation = '('+df3['average validaion']
    fig, ax=plt.subplots()
    plt.plot(df3['ab_ratios'],df3['u_values'],'o',label='Models')
    # plt.text(df3['ab_ratios']+0.3, df3['u_values']+0.3, df3[''])
    plt.plot(-1,1,'x', color='r',label='Correct Model')
    # plt.plot(df5['ab_ratios'], df5['u_values'],'^', label='Common Models')
    rect = Rectangle((-1.02,0.9995), 0.04, 0.001, linewidth=1, edgecolor='r', fill=False, facecolor='w')
    ax.add_patch(rect)
    plt.xlabel('AB Ratio')
    plt.ylabel('U Value')
    plt.axvline(x=-1, color='g')
    plt.axhline(y=1, color='g')
    # plt.xticks(x_labels)
    # plt.yticks(y_labels)

    # ax.set_xticklabels(x_labels)
    # ax.set_yticklabels(y_labels)
    # ax.set_xticks(x_labels)
    # ax.set_yticks(y_labels)
    plt.plot(df3['ab_ratios'].iloc[1], df3['u_values'].iloc[1],'D',color='m')
    plt.plot(df4['ab_ratios'].iloc[0], df4['u_values'].iloc[0], 'D',color='m',label='Common Models')
    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)
    plt.legend(loc='upper left')

    annotations = []
    for i in range(len(df3)):
        annotation = '('+str(round(df3['avg validation losses'].iloc[i],8))+', '+str(round(df3['average first point of failure (2000 tokens)'].iloc[i],2))+')'
        annotations.append(annotation)
    # for i in value(len(df3)):
    #     ax.annotate(df3['avg validation losses'][i], (df3['ab_ratios'][i],df3['u_values'][i]))

    # plt.plot(-1, 1, 'x', color='r', label='Correct Model')
    print(annotations)

    for i, txt in enumerate(annotations):
        ax.annotate(txt, (df3['ab_ratios'].iloc[i],df3['u_values'].iloc[i]))
    plt.subplots_adjust(left=0.15, right=0.8)
    plt.savefig(relu_prefix_2+'INDICATORS_scatter_plots_best_5_models_selected_by_avg_val_loss.png')
    plt.show()
    plt.close()


    x_labels=[]
    y_labels = []
    for i in range(6):
        x_labels.append(round(-1.02+(0.02*i),2))

    for i in range(8):
        y_labels.append(0.999+(0.001*i))

    fig, ax = plt.subplots()
    plt.plot(df4['ab_ratios'], df4['u_values'], 'o', label='Models')
    plt.plot(-1,1,'x', color='r',label='Correct Model')
    # plt.plot(df5['ab_ratios'], df5['u_values'], '^', label='Common Models')
    rect = Rectangle((-1.02, 0.9995), 0.04, 0.001, linewidth=1, edgecolor='r', fill=False, facecolor='w')
    ax.add_patch(rect)
    plt.xlabel('AB Ratio')
    plt.ylabel('U Value')
    plt.plot(df3['ab_ratios'].iloc[1], df3['u_values'].iloc[1],'D',color='m')
    plt.plot(df4['ab_ratios'].iloc[0], df4['u_values'].iloc[0], 'D',color='m',label='Common Models')
    # plt.xticks(x_labels)
    # plt.yticks(y_labels)
    # ax.set_xticks(x_labels)
    # ax.set_yticks(y_labels)
    # ax.set_xticklabels(x_labels)
    # ax.set_yticklabels(y_labels)
    plt.axvline(x=-1, color='g')
    plt.axhline(y=1, color='g')
    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)
    plt.legend(loc='upper left')

    annotations = []
    for i in range(len(df4)):
        annotation = '(' + str(round(df4['avg validation losses'].iloc[i], 8)) + ', ' + str(
            round(df4['average first point of failure (2000 tokens)'].iloc[i], 2)) + ')'
        annotations.append(annotation)

    # for i in value(len(df3)):
    #     ax.annotate(df3['avg validation losses'][i], (df3['ab_ratios'][i],df3['u_values'][i]))

    print(annotations)

    for i, txt in enumerate(annotations):
        ax.annotate(txt, (df4['ab_ratios'].iloc[i], df4['u_values'].iloc[i]))
    plt.subplots_adjust(left=0.15, right=0.8)
    plt.savefig(relu_prefix_2 + 'INDICATORS_scatter_plots_best_5_models_selected_by_fpf.png')
    plt.show()
    plt.close()




plot_model_indicators_good_models()

# extractModelIndicators()
# plotModelIndicators()


def create_2DLinearRegression(par):

    _, df = getReLUModels()
    neg_log_losses = df['log of inverse avg validation losses']
    # fpfs = df[]

    print('par = ',par)
    if par=='NegLogLoss':
        Y=df['log of inverse avg validation losses']
        # X = df[['ab_ratios_dev', 'u_values_dev']]
        # # pass
        # X = sm.add_constant(X)
        # reg_res = sm.OLS(Y, X).fit()
        # print(reg_res.summary())
    elif par=='FPF':
        Y=df['average first point of failure (2000 tokens)']
        # X = df[['ab_ratios_dev', 'u_values_dev']]
        # pass
        # X = sm.add_constant(X)
        # reg_res = sm.OLS(Y, X).fit()
        # print(reg_res.summary())
    elif par=='NegLogFPF':
        Y = -1*np.log(df['average first point of failure (2000 tokens)'])
    elif par=='Loss':
        Y = df['avg validation losses']
    elif par == 'epowNegFPF':
        Y = np.exp(-1*df['average first point of failure (2000 tokens)'])

    X=df[['ab_ratios_dev', 'u_values_dev']]
    # pass
    X = sm.add_constant(X)
    reg_res = sm.OLS(Y,X).fit()
    print(reg_res.summary())

def create2DCustomRegression():
    _, df = getReLUModels()
    df['avg_fpfs'] = df['average first point of failure (2000 tokens)']
    df['ab_ratios_dev_squared'] = df['ab_ratios_dev']**2
    df['u_values_dev_squared'] = df['u_values_dev']**2
    df['ePowNegFPF'] = np.exp(-1*df['avg_fpfs'])
    df['ab_u_combined'] = df['ab_ratios_dev']*df['u_values_dev']
    # mod = smf.ols(formula='ePowNegFPF ~ np.sqrt(ab_ratios_dev_squared + u_values_dev_squared)', data=df)
    mod = smf.ols(formula='ePowNegFPF ~ ab_ratios_dev + u_values_dev + ab_u_combined', data=df)
    res=mod.fit()
    print(res.summary())



    # pass
# def scaledEuclidean():
#     _, df = getReLUModels()
#     ab_ratios_dev = df['ab_ratios_dev']
#     u_dev = df['u_values_dev']



create_2DLinearRegression('NegLogLoss')
create_2DLinearRegression('FPF')
create_2DLinearRegression('NegLogFPF')
create_2DLinearRegression('Loss')
create_2DLinearRegression('epowNegFPF')

create2DCustomRegression()


# print(0.25--1)