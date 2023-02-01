import torch
import torch.nn as nn
from models_batch import VanillaLSTM
import os
import numpy
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_10runs_MODEL_run9.pth'):
    prefix = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    # model_name = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_10runs_MODEL_run9.pth'
elif os.path.exists('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_10runs_MODEL_run9.pth'):
    prefix = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/'
    # model_name = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_10runs_MODEL_run9.pth'

excel_name = prefix+'VanillaLSTM_METRICS.xlsx'

# model=VanillaLSTM(input_size=2,hidden_size=1,num_layers=1,batch_size=1,output_size=2,output_activation='Sigmoid')

# model.load_state_dict(torch.load(model_name))
# model.to(device)
# print(model)
# print(model.parameters())
num_runs = 10
num_epochs=30

# a_values = []
# b_values = []

###############################################################################################

# IGNORE


#no need to extract weights because we have them stored in an excel sheet already
def extract_lstm_weights(model):
    for param in model.lstm.named_parameters():
        if 'weight_hh' in param[0]:
            weights_hh = param[1]
            weights_hi = weights_hh[0]
            weights_hf=weights_hh[1]
            weights_hg = weights_hh[2]
            weights_ho = weights_hh[3]



        elif 'weight_ih' in param[0]:
            weights_ih = param[1]
            weights_ii = weights_ih[0]
            weights_if = weights_ih[1]
            weights_ig = weights_ih[2]
            weights_io = weights_ih[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]

        elif 'bias_hh' in param[0]:
            biases_hh = param[1]
            biases_hi = biases_hh[0]
            biases_hf = biases_hh[1]
            biases_hg = biases_hh[2]
            biases_ho = biases_hh[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]
    
    return weights_if.cpu().detach().numpy(), weights_ii.cpu().detach().numpy(), weights_ig.cpu().detach().numpy(), weights_io.cpu().detach().numpy(), biases_if.item(), biases_ii.item(), biases_ig.item(), biases_io.item(), \
           weights_hf.item(), weights_hi.item(), weights_hg.item(), weights_ho.item(), biases_hf.item(), biases_hi.item(), biases_hg.item(), biases_ho.item()

# def extract_lstm_weights_all_models():
#     model = VanillaLSTM(input_size=2, hidden_size=1, num_layers=1, batch_size=1, output_size=2,
#                         output_activation='Sigmoid')
#
#     weights_all_if = []
#     weights_all_ii = []
#     weights_all_ig = []
#     weights_all_io = []
#     biases_all_if = []
#     biases_all_ii = []
#     biases_all_ig = []
#     biases_all_io = []
#
#     weights_all_hf = []
#     weights_all_hi = []
#     weights_all_hg = []
#     weights_all_ho = []
#     biases_all_hf = []
#     biases_all_hi = []
#     biases_all_hg = []
#     biases_all_ho = []
#
#
#     for run in range(num_runs):
#         for epoch in range(num_epochs):
#
#             model_name = prefix+'Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_30epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs_CHECKPOINT_run'+str(run)+'_epoch'+str(epoch)+'.pth'
#
#             model.load_state_dict(torch.load(model_name)['model_state_dict'])
#             model.to(device)
#             weights_if, weights_ii, weights_ig, weights_io, biases_if, biases_ii, biases_ig, biases_io, \
#                 weights_hf, weights_hi, weights_hg, weights_ho, biases_hf, biases_hi, biases_hg, biases_ho = extract_lstm_weights(model)
#
#             weights_all_if.append(weights_if)
#             weights_all_ii.append(weights_ii)
#             weights_all_ig.append(weights_all_ig)
#             weights_all_io.append(weights_io)
#             biases_all_if.append(biases_if)
#             biases_all_ii.append(biases_ii)
#             biases_all_ig.append(biases_all_ig)
#             biases_all_io.append(biases_io)
#
#             weights_all_hf.append(weights_hf)
#             weights_all_hi.append(weights_hi)
#             weights_all_hg.append(weights_all_hg)
#             weights_all_ho.append(weights_ho)
#             biases_all_hf.append(biases_hf)
#             biases_all_hi.append(biases_hi)
#             biases_all_hg.append(biases_all_hg)
#             biases_all_ho.append(biases_ho)
#
#
#
# def extract_lstm_weights_checkpoints():
#     pass


def calculate_lstm_indicators(seq):
    weights_if, weights_ii, weights_ig, weights_io, biases_if, biases_ii, biases_ig, biases_io, \
        weights_hf, weights_hi, weights_hg, weights_ho, biases_hf, biases_hi, biases_hg, biases_ho = extract_lstm_weights()
    
    
    print('weights_if = ',weights_if)
    print('weights_ig = ', weights_ig)
    print('weights_io = ', weights_io)
    print('weights_ii = ', weights_ii)
    print('biases_if = ', biases_if)
    print('biases_ig = ', biases_ig)
    print('biases_io = ', biases_io)
    print('biases_ii = ', biases_ii)
    print('weights_hf = ', weights_hf)
    print('weights_hg = ', weights_hg)
    print('weights_ho = ', weights_ho)
    print('weights_hi = ', weights_hi)
    print('biases_hf = ', biases_hf)
    print('biases_hg = ', biases_hg)
    print('biases_ho = ', biases_ho)
    print('biases_hi = ', biases_hi)
    
    
    a_values = []
    b_values = []
    c_values=[]
    h_values = []
    h_prev_values = []
    a_b_ratios = []
    i_t_values = []


    #tokenise sequence
    #input the tokens one at a time and calculate the parameters and store in arrays

    tokens = [char for char in seq]

    for i in range(len(tokens)):
        a = 0
        b = 0
        c = 0
        h = 0
        h_prev = h_prev_values[i]

        tok = tokens[i]
        if tok == '(':
            #write calculations here
            pass
        elif tok == ')':
            #write calculations here
            pass

    #calculate the average a, b, and a_b_ratio
    #calculate the average i_t value (recurrent weight)



calculate_lstm_indicators('()()')


def inspect_lstm(model):
    print(model)

    for param in model.lstm.named_parameters():
        if 'weight_hh' in param[0]:
            weights_hh = param[1]
            weights_hi = weights_hh[0]
            weights_hf=weights_hh[1]
            weights_hg = weights_hh[2]
            weights_ho = weights_hh[3]



        elif 'weight_ih' in param[0]:
            weights_ih = param[1]
            weights_ii = weights_ih[0]
            weights_if = weights_ih[1]
            weights_ig = weights_ih[2]
            weights_io = weights_ih[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]

        elif 'bias_hh' in param[0]:
            biases_hh = param[1]
            biases_hi = biases_hh[0]
            biases_hf = biases_hh[1]
            biases_hg = biases_hh[2]
            biases_ho = biases_hh[3]

        elif 'bias_ih' in param[0]:
            biases_ih = param[1]
            biases_ii = biases_ih[0]
            biases_if = biases_ih[1]
            biases_ig = biases_ih[2]
            biases_io = biases_ih[3]

    print('\n')
    print('TO CALCULATE IT')
    print('weight_ii = ', weights_ii)
    print('bias_ii = ', biases_ii)
    print('weight_hi = ', weights_hi)
    print('bias_hi = ', biases_hi)




    metric_it_1 = min(weights_ii[0].item(), weights_ii[1].item()) + biases_ii.item() + biases_hi.item() - torch.abs(weights_hi).item()

    # metrics_it_1.append(metric_it_1)
    print('metric_it_1 = ', metric_it_1)
    print('sigmoid(metric_it_1_min) = ', torch.sigmoid(torch.tensor(metric_it_1, dtype=torch.float32)))
    metric_it_1_best_case = max(weights_ii[0].item(), weights_ii[1].item()) + biases_ii.item() + \
                            biases_hi.item() + torch.abs(weights_hi).item()
    print('metric_it_1_best_case = ', metric_it_1_best_case)
    print('sigmoid(metric_it_1_best_case) = ', torch.sigmoid(torch.tensor(metric_it_1_best_case, dtype=torch.float32)))

    sigmoid_metric_it_1_best = torch.sigmoid(torch.tensor(metric_it_1_best_case, dtype=torch.float32)).item()
    sigmoid_metric_it_1_worst = torch.sigmoid(torch.tensor(metric_it_1, dtype=torch.float32)).item()

    print('\n')
    print('TO CALCULATE FT')
    print('weight_if = ', weights_if)
    print('bias_if = ', biases_if)
    print('weight_hf = ', weights_hf)
    print('bias_hf = ', biases_hf)

    metric_ft_1 = min(weights_if[0].item(), weights_if[1].item()) + biases_if.item() + biases_hf.item() - torch.abs(weights_hf).item()
    # metrics_ft.append(metric_ft_1)
    print('metric_ft_1 = ', metric_ft_1)
    print('sigmoid(metric_ft_1) = ', torch.sigmoid(torch.tensor(metric_ft_1, dtype=torch.float32)))
    metric_ft_1_best_case = max(weights_if[0].item(), weights_if[1].item()) + biases_if.item() + \
                            biases_hf.item() + torch.abs(weights_hf).item()
    print('metric_ft_1_best_case = ', metric_ft_1_best_case)
    print('sigmoid(metric_ft_1_best_case) = ', torch.sigmoid(torch.tensor(metric_ft_1_best_case, dtype=torch.float32)))

    sigmoid_metric_ft_1_best = torch.sigmoid(torch.tensor(metric_ft_1_best_case, dtype=torch.float32)).item()
    sigmoid_metric_ft_1_worst = torch.sigmoid(torch.tensor(metric_ft_1, dtype=torch.float32)).item()

    print('\n')
    print('TO CALCULATE GT (C TILDE IN THE PAPER)')
    print('weight_ig = ', weights_ig)
    print('bias_ig = ', biases_ig)
    print('weight_hg = ', weights_hg)
    print('bias_hg = ', biases_hg)

    metric_ctilde_open = weights_ig[0].item() + biases_ig.item() + biases_hg.item() + torch.abs(
        weights_hg).item()
    # metrics_ctilde_open.append(metric_ctilde_open)
    metric_ctilde_open_worst_case = weights_ig[0].item() + biases_ig.item() + biases_hg.item() - torch.abs(
        weights_hg).item()
    # metrics_ctilde_open_worst_case.append(metric_ctilde_open_worst_case)

    metric_ctilde_close = weights_ig[1].item() + biases_ig.item() + biases_hg.item() - torch.abs(
        weights_hg).item()
    # metrics_ctilde_close.append(metric_ctilde_close)

    metric_ctilde_close_worst_case = weights_ig[1].item() + biases_ig.item() + biases_hg.item() + torch.abs(
        weights_hg).item()
    # metrics_ctilde_close_worst_case.append(metric_ctilde_close_worst_case)

    print('metric_ctilde_open = ', metric_ctilde_open)
    print('metric_ctilde_close = ', metric_ctilde_close)

    print('tanh(metric_ctilde_open) = ', torch.tanh(torch.tensor(metric_ctilde_open, dtype=torch.float32)))
    print('tanh(metric_ctilde_close) = ', torch.tanh(torch.tensor(metric_ctilde_close, dtype=torch.float32)))

    print('metric_ctilde_open worst case = ', metric_ctilde_open_worst_case)
    print('metric_ctilde_close worst case = ', metric_ctilde_close_worst_case)

    print('tanh(metric_ctilde_open worst case) = ',
          torch.tanh(torch.tensor(metric_ctilde_open_worst_case, dtype=torch.float32)))
    print('tanh(metric_ctilde_close worst case) = ',
          torch.tanh(torch.tensor(metric_ctilde_close_worst_case, dtype=torch.float32)))

    tanh_metric_ctilde_open_best = torch.tanh(torch.tensor(metric_ctilde_open, dtype=torch.float32)).item()
    tanh_metric_ctilde_open_worst = torch.tanh(torch.tensor(metric_ctilde_open_worst_case, dtype=torch.float32)).item()
    tanh_metric_ctilde_close_best = torch.tanh(torch.tensor(metric_ctilde_close, dtype=torch.float32)).item()
    tanh_metric_ctilde_close_worst = torch.tanh(torch.tensor(metric_ctilde_close_worst_case, dtype=torch.float32)).item()

    print('\n')
    print('TO CALCULATE OT')
    print('weight_io = ', weights_io)
    print('bias_io = ', biases_io)
    print('weight_ho = ', weights_ho)
    print('bias_ho = ', biases_ho)

    metric_ot = biases_io.item() + biases_ho.item() - max(weights_io[0].item(),
                                                                    weights_io[1].item()) - weights_ho.item()
    # metrics_ot.append(metric_ot)
    print('metric_ot = ', metric_ot)
    print('sigmoid(metric_ot) = ', torch.sigmoid(torch.tensor(metric_ot, dtype=torch.float32)))
    print('\n')
    sigmoid_metric_ot = torch.sigmoid(torch.tensor(metric_ot, dtype=torch.float32)).item()

    return weights_if.cpu().detach().numpy(), weights_ii.cpu().detach().numpy(), weights_ig.cpu().detach().numpy(), weights_io.cpu().detach().numpy(), biases_if.item(), biases_ii.item(), biases_ig.item(), biases_io.item(), \
           weights_hf.item(), weights_hi.item(), weights_hg.item(), weights_ho.item(), biases_hf.item(), biases_hi.item(), biases_hg.item(), biases_ho.item(), \
           metric_ft_1, metric_ft_1_best_case, metric_it_1, metric_it_1_best_case, metric_ctilde_open, metric_ctilde_open_worst_case, \
           metric_ctilde_close, metric_ctilde_close_worst_case, metric_ot, sigmoid_metric_ft_1_best, sigmoid_metric_ft_1_worst, \
           sigmoid_metric_it_1_best, sigmoid_metric_it_1_worst, tanh_metric_ctilde_open_best, tanh_metric_ctilde_open_worst, \
           tanh_metric_ctilde_close_best, tanh_metric_ctilde_close_worst, sigmoid_metric_ot

###################################################################################################################

#RELEVANT CODE STARTS HERE

def read_sheets():
    sheet_name='Sheet1'
    df = pd.read_excel(excel_name, sheet_name=sheet_name)
    # df = pd.read_excel(path_name,sheet_name=sheet_names)
    # dfs = []
    # for i in range(num_runs):
    #     dfs.append(df.get(sheet_names[i]))
    # return dfs
    return df




