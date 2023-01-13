import torch
import torch.nn as nn
from models_batch import VanillaLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = '/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaLSTM/1_batch_size/0.01_learning_rate/20_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaLSTM_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_20epochs_10runs_MODEL_run9.pth'


model=VanillaLSTM(input_size=2,hidden_size=1,num_layers=1,batch_size=1,output_size=2,output_activation='Sigmoid')

model.load_state_dict(torch.load(model_name))
model.to(device)
print(model)
