# import math

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
# import pandas.util.testing as tm
# import pandas.testing
# import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if os.path.exists('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'):
    prefix='/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'
elif os.path.exists('/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'):
    prefix='/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'


#
# excel_name_a_dev_models = prefix+'INDICATORS_handmade_models_a_dev_test.xlsx'
# excel_name_u_dev_models = prefix+'INDICATORS_handmade_models_u_dev_test.xlsx'
# prefix='/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/30_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/20_runs/shuffle_True/'

excel_name = prefix+'INDICATORS_handmade_models_test.xlsx'


df = pd.read_excel(excel_name, sheet_name='Sheet1')
print(df.columns)
def correlate_model_indicators():
    print('*************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND FPF')
    res_dev_signed_ab_ratio_fpf = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], df.loc[df['model_u_devs']==0]['long_avg_point_of_failure'])
    print('rvalue = ',res_dev_signed_ab_ratio_fpf.rvalue)
    print('rsquared = ',res_dev_signed_ab_ratio_fpf.rvalue**2)
    print('pvalue = ',res_dev_signed_ab_ratio_fpf.pvalue)
    print('intercept = ',res_dev_signed_ab_ratio_fpf.intercept)
    print('slope = ',res_dev_signed_ab_ratio_fpf.slope)

    fig, ax=plt.subplots()
    df1=df.loc[df['model_u_devs']==0]
    plt.plot(df1['model_ab_ratios_devs_signed'],
             df1['long_avg_point_of_failure'], 'o', label='Models')
    plt.axvline(x=0, color='r')

    # plt.plot(1,color='r')

    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], df.loc[df['model_u_devs']==0]['long_avg_point_of_failure'], 'o', label='Models')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_fpf.intercept+(res_dev_signed_ab_ratio_fpf.slope*df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Average FPF')
    y_labels = [700, 750, 800, 850, 900, 950, 1000]
    ax.set_yticks(y_labels)
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_fpf.png')
    plt.close()



    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND LOG FPF')
    res_dev_signed_ab_ratio_log_fpf = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], np.log(df.loc[df['model_u_devs']==0]['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_signed_ab_ratio_log_fpf.rvalue)
    print('rsquared = ', res_dev_signed_ab_ratio_log_fpf.rvalue**2)
    print('pvalue = ', res_dev_signed_ab_ratio_log_fpf.pvalue)
    print('intercept = ', res_dev_signed_ab_ratio_log_fpf.intercept)
    print('slope = ', res_dev_signed_ab_ratio_log_fpf.slope)

    plt.subplots()
    df1 = df.loc[df['model_u_devs'] == 0]
    plt.plot(df1['model_ab_ratios_devs_signed'], np.log(df1['long_avg_point_of_failure']), 'o', label='Models')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_log_fpf.intercept+(res_dev_signed_ab_ratio_log_fpf.slope*df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Log Average FPF')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND NEG LOG FPF')
    res_dev_signed_ab_ratio_neg_log_fpf = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], -1*np.log(df.loc[df['model_u_devs']==0]['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_signed_ab_ratio_neg_log_fpf.rvalue)
    print('rsquared = ', res_dev_signed_ab_ratio_neg_log_fpf.rvalue**2)
    print('pvalue = ', res_dev_signed_ab_ratio_neg_log_fpf.pvalue)
    print('intercept = ', res_dev_signed_ab_ratio_neg_log_fpf.intercept)
    print('slope = ', res_dev_signed_ab_ratio_neg_log_fpf.slope)

    plt.subplots()
    df1 = df.loc[df['model_u_devs'] == 0]
    plt.plot(df1['model_ab_ratios_devs_signed'], -1*np.log(df1['long_avg_point_of_failure']), 'o', label='Models')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_neg_log_fpf.intercept+(res_dev_signed_ab_ratio_neg_log_fpf.slope*df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Negative Log Average FPF')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_neg_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND FPF')
    res_dev_absolute_ab_ratio_fpf = stats.linregress(df['model_ab_ratios_devs_absolute'], df['long_avg_point_of_failure'])
    print('rvalue = ', res_dev_absolute_ab_ratio_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_fpf.rvalue**2)
    print('pvalue = ', res_dev_absolute_ab_ratio_fpf.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_fpf.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_fpf.slope)
    
    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], df['long_avg_point_of_failure'], 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_fpf.intercept+(res_dev_absolute_ab_ratio_fpf.slope*df['model_ab_ratios_devs_absolute']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('Average FPF')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND e^FPF')
    res_dev_absolute_ab_ratio_e_pow_fpf = stats.linregress(df['model_ab_ratios_devs_absolute'],
                                                     np.exp(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_ab_ratio_e_pow_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_e_pow_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_ab_ratio_e_pow_fpf.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_e_pow_fpf.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_e_pow_fpf.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], np.exp(df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_e_pow_fpf.intercept + (
                res_dev_absolute_ab_ratio_e_pow_fpf.slope * df['model_ab_ratios_devs_absolute']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('e pow(Average FPF)')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_e_pow_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND e^-FPF')
    res_dev_absolute_ab_ratio_e_pow_neg_fpf = stats.linregress(df['model_ab_ratios_devs_absolute'],
                                                     np.exp(-1*df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_ab_ratio_e_pow_neg_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_e_pow_neg_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_ab_ratio_e_pow_neg_fpf.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_e_pow_neg_fpf.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_e_pow_neg_fpf.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], np.exp(-1*df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_e_pow_neg_fpf.intercept + (
                res_dev_absolute_ab_ratio_e_pow_neg_fpf.slope * df['model_ab_ratios_devs_absolute']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('e pow(Average neg_fpf)')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_e_pow_neg_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND LOG FPF')
    res_dev_absolute_ab_ratio_log_fpf = stats.linregress(df['model_ab_ratios_devs_absolute'],np.log(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_ab_ratio_log_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_log_fpf.rvalue**2)
    print('pvalue = ', res_dev_absolute_ab_ratio_log_fpf.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_log_fpf.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_log_fpf.slope)
    
    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], np.log(df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_log_fpf.intercept+(res_dev_absolute_ab_ratio_log_fpf.slope*df['model_ab_ratios_devs_absolute']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('Log Average FPF')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND NEG LOG FPF')
    res_dev_absolute_ab_ratio_neg_log_fpf = stats.linregress(df['model_ab_ratios_devs_absolute'],-1*np.log(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_ab_ratio_neg_log_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_neg_log_fpf.rvalue**2)
    print('pvalue = ', res_dev_absolute_ab_ratio_neg_log_fpf.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_neg_log_fpf.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_neg_log_fpf.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], -1*np.log(df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_neg_log_fpf.intercept+(res_dev_absolute_ab_ratio_neg_log_fpf.slope*df['model_ab_ratios_devs_absolute']),label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (absolute)')
    plt.ylabel('Negative Log Average FPF')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_neg_log_fpf.png')
    plt.close()
    
    print('**********************************************************')
    print('*************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND val_loss')
    res_dev_signed_ab_ratio_val_loss = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], df.loc[df['model_u_devs']==0]['val_losses'])
    print('rvalue = ', res_dev_signed_ab_ratio_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_ab_ratio_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_ab_ratio_val_loss.pvalue)
    print('intercept = ', res_dev_signed_ab_ratio_val_loss.intercept)
    print('slope = ', res_dev_signed_ab_ratio_val_loss.slope)

    fig, ax=plt.subplots()
    df1 = df.loc[df['model_u_devs'] == 0]
    plt.plot(df1['model_ab_ratios_devs_signed'], df1['val_losses'], 'o', label='Models')
    # plt.plot(df1['model_a_devs'], df1['val_losses'], 'o', c='r', label='Models1')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_val_loss.intercept + (res_dev_signed_ab_ratio_val_loss.slope * df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']), label='Fitted Line')
    plt.axvline(x=0, color='r')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Average Validation Loss')
    y_labels = [0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022]
    ax.set_yticks(y_labels)
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND LOG val_loss')
    res_dev_signed_ab_ratio_log_val_loss = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'],
                                                       np.log(df.loc[df['model_u_devs']==0]['val_losses']))
    print('rvalue = ', res_dev_signed_ab_ratio_log_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_ab_ratio_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_ab_ratio_log_val_loss.pvalue)
    print('intercept = ', res_dev_signed_ab_ratio_log_val_loss.intercept)
    print('slope = ', res_dev_signed_ab_ratio_log_val_loss.slope)

    plt.subplots()
    plt.plot(df1['model_ab_ratios_devs_signed'], np.log(df1['val_losses']), 'o', label='Models')
    plt.plot(df1['model_a_devs'], np.log(df1['val_losses']), 'o', c='r', label='Models')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_log_val_loss.intercept + (
    #             res_dev_signed_ab_ratio_log_val_loss.slope * df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED ABR DEVIATIONS AND NEG LOG val_loss')
    res_dev_signed_ab_ratio_neg_log_val_loss = stats.linregress(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'],
                                                           -1 * np.log(df.loc[df['model_u_devs']==0]['val_losses']))
    print('rvalue = ', res_dev_signed_ab_ratio_neg_log_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_ab_ratio_neg_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_ab_ratio_neg_log_val_loss.pvalue)
    print('intercept = ', res_dev_signed_ab_ratio_neg_log_val_loss.intercept)
    print('slope = ', res_dev_signed_ab_ratio_neg_log_val_loss.slope)

    plt.subplots()
    df1 = df.loc[df['model_u_devs'] == 0]
    plt.plot(df1['model_ab_ratios_devs_signed'], -1 * np.log(df1['val_losses']), 'o', label='Models')
    plt.plot(df1['model_a_devs'], -1 * np.log(df1['val_losses']), 'o', c='r', label='Models 2')
    # plt.plot(df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed'], res_dev_signed_ab_ratio_neg_log_val_loss.intercept + (
    #             res_dev_signed_ab_ratio_neg_log_val_loss.slope * df.loc[df['model_u_devs']==0]['model_ab_ratios_devs_signed']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Signed)')
    plt.ylabel('Negative Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_ab_ratios_neg_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND val_loss')
    res_dev_absolute_ab_ratio_val_loss = stats.linregress(df['model_ab_ratios_devs_absolute'],
                                                     df['val_losses'])
    print('rvalue = ', res_dev_absolute_ab_ratio_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_ab_ratio_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_val_loss.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_val_loss.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], df['val_losses'], 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_val_loss.intercept + (
                res_dev_absolute_ab_ratio_val_loss.slope * df['model_ab_ratios_devs_absolute']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND LOG val_loss')
    res_dev_absolute_ab_ratio_log_val_loss = stats.linregress(df['model_ab_ratios_devs_absolute'],
                                                         np.log(df['val_losses']))
    print('rvalue = ', res_dev_absolute_ab_ratio_log_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_ab_ratio_log_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_log_val_loss.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_log_val_loss.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], np.log(df['val_losses']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_log_val_loss.intercept + (
                res_dev_absolute_ab_ratio_log_val_loss.slope * df['model_ab_ratios_devs_absolute']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (Absolute)')
    plt.ylabel('Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE ABR DEVIATIONS AND NEG LOG val_loss')
    res_dev_absolute_ab_ratio_neg_log_val_loss = stats.linregress(df['model_ab_ratios_devs_absolute'],
                                                             -1 * np.log(df['val_losses']))
    print('rvalue = ', res_dev_absolute_ab_ratio_neg_log_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_ab_ratio_neg_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_ab_ratio_neg_log_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_ab_ratio_neg_log_val_loss.intercept)
    print('slope = ', res_dev_absolute_ab_ratio_neg_log_val_loss.slope)

    plt.subplots()
    plt.plot(df['model_ab_ratios_devs_absolute'], -1 * np.log(df['val_losses']), 'o', label='Models')
    plt.plot(df['model_ab_ratios_devs_absolute'], res_dev_absolute_ab_ratio_neg_log_val_loss.intercept + (
                res_dev_absolute_ab_ratio_neg_log_val_loss.slope * df['model_ab_ratios_devs_absolute']), label='Fitted Line')
    plt.xlabel('AB Ratio Deviations (absolute)')
    plt.ylabel('Negative Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_ab_ratios_neg_log_val_loss.png')
    plt.close()
    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    
    print('*************************************************')
    print('LINEAR REGRESSION -- SIGNED U DEVIATIONS AND FPF')
    res_dev_signed_u_value_fpf = stats.linregress(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], df.loc[df['model_ab_ratios_devs_absolute']==0]['long_avg_point_of_failure'])
    print('rvalue = ', res_dev_signed_u_value_fpf.rvalue)
    print('rsquared = ', res_dev_signed_u_value_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_fpf.pvalue)
    print('intercept = ', res_dev_signed_u_value_fpf.intercept)
    print('slope = ', res_dev_signed_u_value_fpf.slope)

    fig, ax=plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'],
             df1['long_avg_point_of_failure'], 'o', label='Models')

    # plt.plot(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], df.loc[df['model_ab_ratios_devs_absolute']==0]['long_avg_point_of_failure'], 'o', label='Models')
    # plt.plot(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], res_dev_signed_u_value_fpf.intercept + (
    #             res_dev_signed_u_value_fpf.slope * df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs']), label='Fitted Line')
    plt.axvline(x=0, color='r')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Average FPF')
    y_labels = [700, 750, 800, 850, 900, 950, 1000]
    ax.set_yticks(y_labels)
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED U  DEVIATIONS AND LOG FPF')
    res_dev_signed_u_value_log_fpf = stats.linregress(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'],
                                                       np.log(df.loc[df['model_ab_ratios_devs_absolute']==0]['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_signed_u_value_log_fpf.rvalue)
    print('rsquared = ', res_dev_signed_u_value_log_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_log_fpf.pvalue)
    print('intercept = ', res_dev_signed_u_value_log_fpf.intercept)
    print('slope = ', res_dev_signed_u_value_log_fpf.slope)

    plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'],
             np.log(df1['long_avg_point_of_failure']), 'o', label='Models')
    # plt.plot(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], np.log(df.loc[df['model_ab_ratios_devs_absolute']==0]['long_avg_point_of_failure']), 'o', label='Models')
    # plt.plot(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], res_dev_signed_u_value_log_fpf.intercept + (
    #             res_dev_signed_u_value_log_fpf.slope * df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs']), label='Fitted Line')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Log Average FPF')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED U  DEVIATIONS AND NEG LOG FPF')
    res_dev_signed_u_value_neg_log_fpf = stats.linregress(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'],
                                                           -1 * np.log(df.loc[df['model_ab_ratios_devs_absolute']==0]['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_signed_u_value_neg_log_fpf.rvalue)
    print('rsquared = ', res_dev_signed_u_value_neg_log_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_neg_log_fpf.pvalue)
    print('intercept = ', res_dev_signed_u_value_neg_log_fpf.intercept)
    print('slope = ', res_dev_signed_u_value_neg_log_fpf.slope)

    plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'], -1 * np.log(df1['long_avg_point_of_failure']), 'o', label='Models')
    # plt.plot(df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs'], res_dev_signed_u_value_neg_log_fpf.intercept + (
    #             res_dev_signed_u_value_neg_log_fpf.slope * df.loc[df['model_ab_ratios_devs_absolute']==0]['model_u_devs']), label='Fitted Line')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Negative Log Average FPF')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_neg_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND FPF')
    res_dev_absolute_u_value_fpf = stats.linregress(abs(df['model_u_devs']),
                                                     df['long_avg_point_of_failure'])
    print('rvalue = ', res_dev_absolute_u_value_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_fpf.pvalue)
    print('intercept = ', res_dev_absolute_u_value_fpf.intercept)
    print('slope = ', res_dev_absolute_u_value_fpf.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), df['long_avg_point_of_failure'], 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_fpf.intercept + (
                res_dev_absolute_u_value_fpf.slope * abs(df['model_u_devs'])), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Average FPF')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_fpf.png')
    plt.close()
    
    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND e^FPF')
    res_dev_absolute_u_value_e_pow_fpf = stats.linregress(abs(df['model_u_devs']),
                                                     np.exp(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_u_value_e_pow_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_e_pow_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_e_pow_fpf.pvalue)
    print('intercept = ', res_dev_absolute_u_value_e_pow_fpf.intercept)
    print('slope = ', res_dev_absolute_u_value_e_pow_fpf.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), df['long_avg_point_of_failure'], 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_e_pow_fpf.intercept + (
                res_dev_absolute_u_value_e_pow_fpf.slope * np.exp(abs(df['model_u_devs']))), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Average e_pow_fpf')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_e_pow_fpf.png')
    plt.close()
    
    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND e^-FPF')
    res_dev_absolute_u_value_e_pow_neg_fpf = stats.linregress(abs(df['model_u_devs']),
                                                     np.exp(-1*df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_u_value_e_pow_neg_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_e_pow_neg_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_e_pow_neg_fpf.pvalue)
    print('intercept = ', res_dev_absolute_u_value_e_pow_neg_fpf.intercept)
    print('slope = ', res_dev_absolute_u_value_e_pow_neg_fpf.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), df['long_avg_point_of_failure'], 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_e_pow_neg_fpf.intercept + (
                res_dev_absolute_u_value_e_pow_neg_fpf.slope * np.exp(-1*abs(df['model_u_devs']))), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Average e_pow_neg_fpf')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_e_pow_neg_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND LOG FPF')
    res_dev_absolute_u_value_log_fpf = stats.linregress(abs(df['model_u_devs']),
                                                         np.log(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_u_value_log_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_log_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_log_fpf.pvalue)
    print('intercept = ', res_dev_absolute_u_value_log_fpf.intercept)
    print('slope = ', res_dev_absolute_u_value_log_fpf.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), np.log(df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_log_fpf.intercept + (
                res_dev_absolute_u_value_log_fpf.slope * abs(df['model_u_devs'])), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Log Average FPF')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_log_fpf.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND NEG LOG FPF')
    res_dev_absolute_u_value_neg_log_fpf = stats.linregress(abs(df['model_u_devs']),
                                                             -1 * np.log(df['long_avg_point_of_failure']))
    print('rvalue = ', res_dev_absolute_u_value_neg_log_fpf.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_neg_log_fpf.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_neg_log_fpf.pvalue)
    print('intercept = ', res_dev_absolute_u_value_neg_log_fpf.intercept)
    print('slope = ', res_dev_absolute_u_value_neg_log_fpf.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), -1 * np.log(df['long_avg_point_of_failure']), 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_neg_log_fpf.intercept + (
                res_dev_absolute_u_value_neg_log_fpf.slope * abs(df['model_u_devs'])), label='Fitted Line')
    plt.xlabel('U Value Deviations (absolute)')
    plt.ylabel('Negative Log Average FPF')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_neg_log_fpf.png')
    plt.close()

    print('**********************************************************')
    print('*************************************************')
    print('LINEAR REGRESSION -- SIGNED U  DEVIATIONS AND val_loss')
    res_dev_signed_u_value_val_loss = stats.linregress(df.loc[df['model_a_devs']==0]['model_u_devs'], df.loc[df['model_a_devs']==0]['val_losses'])
    print('rvalue = ', res_dev_signed_u_value_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_u_value_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_val_loss.pvalue)
    print('intercept = ', res_dev_signed_u_value_val_loss.intercept)
    print('slope = ', res_dev_signed_u_value_val_loss.slope)

    fig, ax=plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'], df1['val_losses'], 'o', label='Models')
    # plt.plot(df.loc[df['model_a_devs']==0]['model_u_devs'], res_dev_signed_u_value_val_loss.intercept + (
    #             res_dev_signed_u_value_val_loss.slope * df.loc[df['model_a_devs']==0]['model_u_devs']), label='Fitted Line')
    plt.axvline(x=0, color='r')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Average Validation Loss')
    # y_labels = [0.017, 0.0172, 0.0174, 0.0176, 0.0178, 0.018, 0.0182, 0.0184, 0.0186, 0.0188, 0.019, 0.0192, 0.0194, 0.0196, 0.0198, 0.02, 0.0202, 0.0204]
    y_labels = [0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022]
    ax.set_yticks(y_labels)
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED U  DEVIATIONS AND LOG val_loss')
    res_dev_signed_u_value_log_val_loss = stats.linregress(df.loc[df['model_a_devs']==0]['model_u_devs'],
                                                            np.log(df.loc[df['model_a_devs']==0]['val_losses']))
    print('rvalue = ', res_dev_signed_u_value_log_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_u_value_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_log_val_loss.pvalue)
    print('intercept = ', res_dev_signed_u_value_log_val_loss.intercept)
    print('slope = ', res_dev_signed_u_value_log_val_loss.slope)

    plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'], np.log(df1['val_losses']), 'o', label='Models')
    # plt.plot(df.loc[df['model_a_devs']==0]['model_u_devs'], res_dev_signed_u_value_log_val_loss.intercept + (
    #         res_dev_signed_u_value_log_val_loss.slope * df.loc[df['model_a_devs']==0]['model_u_devs']), label='Fitted Line')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- SIGNED U  DEVIATIONS AND NEG LOG val_loss')
    res_dev_signed_u_value_neg_log_val_loss = stats.linregress(df.loc[df['model_a_devs']==0]['model_u_devs'],
                                                                -1 * np.log(df.loc[df['model_a_devs']==0]['val_losses']))
    print('rvalue = ', res_dev_signed_u_value_neg_log_val_loss.rvalue)
    print('rsquared = ', res_dev_signed_u_value_neg_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_signed_u_value_neg_log_val_loss.pvalue)
    print('intercept = ', res_dev_signed_u_value_neg_log_val_loss.intercept)
    print('slope = ', res_dev_signed_u_value_neg_log_val_loss.slope)

    plt.subplots()
    df1 = df.loc[df['model_a_devs'] == 0]
    plt.plot(df1['model_u_devs'], -1 * np.log(df1['val_losses']), 'o', label='Models')
    # plt.plot(df.loc[df['model_a_devs']==0]['model_u_devs'], res_dev_signed_u_value_neg_log_val_loss.intercept + (
    #         res_dev_signed_u_value_neg_log_val_loss.slope * df.loc[df['model_a_devs']==0]['model_u_devs']), label='Fitted Line')
    plt.xlabel('U Value Deviations (Signed)')
    plt.ylabel('Negative Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_signed_u_value_neg_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND val_loss')
    res_dev_absolute_u_value_val_loss = stats.linregress(abs(df['model_u_devs']),
                                                          df['val_losses'])
    print('rvalue = ', res_dev_absolute_u_value_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_u_value_val_loss.intercept)
    print('slope = ', res_dev_absolute_u_value_val_loss.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), df['val_losses'], 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_val_loss.intercept + (
            res_dev_absolute_u_value_val_loss.slope * abs(df['model_u_devs'])), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND LOG val_loss')
    res_dev_absolute_u_value_log_val_loss = stats.linregress(abs(df['model_u_devs']),
                                                              np.log(df['val_losses']))
    print('rvalue = ', res_dev_absolute_u_value_log_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_log_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_u_value_log_val_loss.intercept)
    print('slope = ', res_dev_absolute_u_value_log_val_loss.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), np.log(df['val_losses']), 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_log_val_loss.intercept + (
            res_dev_absolute_u_value_log_val_loss.slope * abs(df['model_u_devs'])), label='Fitted Line')
    plt.xlabel('U Value Deviations (Absolute)')
    plt.ylabel('Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_log_val_loss.png')
    plt.close()

    print('****************************************************')
    print('LINEAR REGRESSION -- ABSOLUTE U  DEVIATIONS AND NEG LOG val_loss')
    res_dev_absolute_u_value_neg_log_val_loss = stats.linregress(abs(df['model_u_devs']),
                                                                  -1 * np.log(df['val_losses']))
    print('rvalue = ', res_dev_absolute_u_value_neg_log_val_loss.rvalue)
    print('rsquared = ', res_dev_absolute_u_value_neg_log_val_loss.rvalue ** 2)
    print('pvalue = ', res_dev_absolute_u_value_neg_log_val_loss.pvalue)
    print('intercept = ', res_dev_absolute_u_value_neg_log_val_loss.intercept)
    print('slope = ', res_dev_absolute_u_value_neg_log_val_loss.slope)

    plt.subplots()
    plt.plot(abs(df['model_u_devs']), -1 * np.log(df['val_losses']), 'o', label='Models')
    plt.plot(abs(df['model_u_devs']), res_dev_absolute_u_value_neg_log_val_loss.intercept + (
            res_dev_absolute_u_value_neg_log_val_loss.slope * abs(df['model_u_devs'])),
             label='Fitted Line')
    plt.xlabel('U Value Deviations (absolute)')
    plt.ylabel('Negative Log Average Validation Loss')
    plt.legend()
    plt.savefig(prefix + 'INDICATORS_handmade_models_linear_regression_absolute_u_value_neg_log_val_loss.png')
    plt.close()



    print('EUCLIDEAN NORM WITH VAL LOSS')
    plt.subplots()
    plt.plot(df['model_euclidean_norms'], df['val_losses'], 'o', label='Models')
    plt.xlabel('Euclidean Norm')
    plt.ylabel('Average Validation Loss')
    plt.legend()
    plt.savefig(prefix+'INDICATORS_handmade_models_linear_regression_euclidean_norm_val_loss.png')

    
def create3dCorrelations():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z1 = df['neg_log_val_losses']
    x = df['model_u_devs']
    y = df['model_ab_ratios_devs_signed']
    ax.scatter(x, y, z1)
    ax.set_xlabel('deviation in U values')
    ax.set_ylabel('deviation in AB ratio')
    ax.set_zlabel('negative log validation loss')
    plt.savefig(prefix + 'INDICATORS_handmade_models_3D_deviations_neg_log_val_loss.png')
    plt.show()
    plt.close()


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z1 = df['long_avg_point_of_failure']
    x = df['model_u_devs']
    y = df['model_ab_ratios_devs_signed']
    ax.scatter(x, y, z1)
    ax.set_xlabel('deviation in U values')
    ax.set_ylabel('deviation in AB ratio')
    ax.set_zlabel('FPF')
    plt.savefig(prefix + 'INDICATORS_handmade_models_3D_deviations_fpf.png')
    plt.show()
    plt.close()

def correlate2d_model_indicators(par):
    print('******************************************************')

    if par=='val_loss':
        Y = df['val_losses']
    elif par=='fpf':
        Y = df['long_avg_point_of_failure']

    X=df[['ab_ratios_dev', 'u_values_dev']]



def correlate_model_indicators_check():
    df1 = df.loc[df['model_u_devs']==0]
    print(df1)
    print(len(df1))

def create_heatmap():
    df1=df[['model_a_devs', 'model_u_devs', 'val_losses']]
    x = pd.DataFrame(df1['model_a_devs'].unique())
    heatmap_pt = pd.pivot_table(df1, values='val_losses', index=['model_u_devs'], columns='model_a_devs')
    # headmap_pt
    fig, ax = plt.subplots()
    sns.set()
    ax=sns.heatmap(heatmap_pt, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    plt.xlabel('AB Ratio Deviations')
    plt.ylabel('U Value Deviations')
    plt.axvline(x=5.5, color='g')
    plt.axhline(y=5.5, color='g')
    # plt.xticks(rotation=90)
    # plt.xticks()
    # plt.yticks()
    plt.subplots_adjust(bottom=0.175, left=0.15)
    plt.savefig(prefix + 'INDICATORS_handmade_models_heatmap_devs_val_loss.png')
    plt.show()
    # fig.close()

    df1 = df[['model_a_devs', 'model_u_devs', 'long_avg_point_of_failure']]
    x = pd.DataFrame(df1['model_a_devs'].unique())
    heatmap_pt = pd.pivot_table(df1, values='long_avg_point_of_failure', index=['model_u_devs'], columns='model_a_devs')
    # headmap_pt
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(heatmap_pt, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    plt.xlabel('AB Ratio Deviations')
    plt.ylabel('U Value Deviations')
    rect = Rectangle((5, 5), 1, 1, linewidth=1, edgecolor='w', fill=True, facecolor='w')
    ax.add_patch(rect)
    plt.axvline(x=5.5, color='g')
    plt.axhline(y=5.5, color='g')
    # plt.xticks()
    # plt.yticks()
    # plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.175, left=0.15)
    plt.savefig(prefix+'INDICATORS_handmade_models_heatmap_devs_fpf.png')
    plt.show()

    df1=df[['model_u_weights', 'long_avg_point_of_failure']]
    df1['ab_ratios'] = df['model_a_weights']/df['model_b_weights']
    x = pd.DataFrame(df1['ab_ratios'].unique())
    heatmap_pt=pd.pivot_table(df1, values='long_avg_point_of_failure', index=['model_u_weights'], columns='ab_ratios')
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(heatmap_pt, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    # plt.xticks(rotation=90)
    plt.xlabel('AB Ratio')
    plt.ylabel('U Value')
    rect = Rectangle((5,5), 1,1,linewidth=1,edgecolor='w', fill=True, facecolor='w')
    ax.add_patch(rect)
    plt.axvline(x=5.5, color='g')
    plt.axhline(y=5.5, color='g')
    plt.subplots_adjust(bottom=0.175, left=0.15)
    plt.savefig(prefix + 'INDICATORS_handmade_models_heatmap_fpf.png')
    plt.show()

    df1=df[['model_u_weights', 'val_losses']]
    df1['ab_ratios'] = df['model_a_weights']/df['model_b_weights']
    x = pd.DataFrame(df1['ab_ratios'].unique())
    heatmap_pt=pd.pivot_table(df1, values='val_losses', index=['model_u_weights'], columns='ab_ratios')
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(heatmap_pt, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    # plt.xticks(rotation=90)
    plt.xlabel('AB Ratio')
    plt.ylabel('U Value')

    plt.axvline(x=5.5, color='g')
    plt.axhline(y=5.5, color='g')
    plt.subplots_adjust(bottom=0.175, left=0.15)
    plt.savefig(prefix + 'INDICATORS_handmade_models_heatmap_val_loss.png')
    plt.show()

    df1 = df[['model_u_weights', 'val_bce_losses']]
    df1['ab_ratios'] = df['model_a_weights'] / df['model_b_weights']
    x = pd.DataFrame(df1['ab_ratios'].unique())
    heatmap_pt = pd.pivot_table(df1, values='val_bce_losses', index=['model_u_weights'], columns='ab_ratios')
    fig, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(heatmap_pt, xticklabels=True, yticklabels=True)
    ax.invert_yaxis()
    # plt.xticks(rotation=90)
    plt.xlabel('AB Ratio')
    plt.ylabel('U Value')

    plt.axvline(x=5.5, color='g')
    plt.axhline(y=5.5, color='g')
    plt.subplots_adjust(bottom=0.175, left=0.15)
    plt.savefig(prefix + 'INDICATORS_handmade_models_heatmap_val_bce_loss.png')
    plt.show()



    # plt.subplots()
    # sns.heatmap(df1, annot=True, fmt=".1f").pivot(index="model_a_devs",columns="model_u_devs",values="val_accuracies")
    # plt.show()


correlate_model_indicators()
create3dCorrelations()
create_heatmap()

# correlate_model_indicators_check()