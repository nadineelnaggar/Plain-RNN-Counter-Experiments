import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Dyck1_Datasets import NextTokenPredictionDataset990to1000tokens, NextTokenPredictionDataset502to1000tokens, NextTokenPredictionDataset2000tokens

path_name = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Dyck1_NextTokenPrediction/Minibatch_Training/VanillaReLURNN/1_batch_size/0.01_learning_rate/50_epochs/50_lr_scheduler_step/1.0_lr_scheduler_gamma/1_hidden_units/10_runs/shuffle_True/Dyck1_NextTokenPrediction_25_bracket_pairs_VanillaReLURNN_Feedback_EveryTimeStep_1_batch_size__1hidden_units_Adam_lr=0.01_50epochs_50lr_scheduler_step_1.0lr_scheduler_gamma_10runs.xlsx'

num_runs = 10
sheet_names = []
for i in range(num_runs):
    sheet_names.append('run'+str(i))

dict_df = pd.read_excel(path_name,sheet_name=sheet_names)

dfs = []
for i in range(num_runs):
    dfs.append(dict_df.get(sheet_names[i]))


def read_sheets():
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run"+str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(path_name,sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs

frames=read_sheets()
print(frames[0].head())

print(dfs[1].head())
print(dfs[0].head())

df = dfs[4]
print(df.head())

print(df.at[0,"epoch"])
print(df.at[10,"Average training losses"])


arr = df['Average training losses']
print(len(arr))

print(arr[49])
print(arr[0])
arr2 = arr.tolist()
print(arr2[49])
print(arr2[0])
arr = arr.tolist()
print(arr[0])

print(len(dfs))



arr3 = df['Training accuracies']
arr3=arr3.tolist()
print(arr3)
print(len(arr))
print(len(arr3))

# plt.subplots()
# plt.scatter(x=arr3,y=arr)
# plt.show()


# dataset = NextTokenPredictionDataset1000tokens()
# print(len(dataset))

dataset = NextTokenPredictionDataset990to1000tokens()
print(len(dataset))
count_1000=0

for i in range(len(dataset)):
    if len(dataset.x[i])==1000:
        count_1000+=1

print(count_1000)
print(len(dataset.x[0]))
print(len(dataset.x[-1]))

# print(df.iloc([0][0]))
# print(df.iloc['1','accuracy'])

# print(df.loc['12','average training loss'])
#
# print(df.at([0,'average training loss']).head())

# print(df['average training loss'].head())


dataset2 = NextTokenPredictionDataset2000tokens()
print(len(dataset2.x[0]))
print(len(dataset2))
# print(max(dataset2.max_depth))

# for i in range(len(dataset2)):
#     print(len(dataset2.x[i]))

def get_timestep_depths(x):
    max_depth=0
    current_depth=0
    timestep_depths = []
    for i in range(len(x)):

        if x[i] == '(':
            current_depth += 1
            timestep_depths.append(current_depth)
            if current_depth > max_depth:
                max_depth = current_depth
        elif x[i] == ')':
            current_depth -= 1
            timestep_depths.append(current_depth)
    return max_depth, timestep_depths

# print(get_timestep_depths(dataset2.x[0]))

max_depth, timestep_depths = get_timestep_depths(dataset2.x[0])
print(max_depth)
print(timestep_depths)