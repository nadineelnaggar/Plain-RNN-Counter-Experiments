import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from random import randint
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

sample_fpf = []
hist_bins = []
sample_max_depths = []
for i in range(100):
    num = randint(0,100)
    sample_fpf.append(num)
    # hist_bins.append(i)
    depth = randint(0,50)
    sample_max_depths.append(depth)

for i in range(101):
    hist_bins.append(i)
# plt.hist(sample_fpf,bins=timestep_depths)
# plt.show()
#
# plt.hist(timestep_depths,sample_fpf)
# plt.show()

print(hist_bins)
print(sample_fpf)

plt.hist(sample_fpf,bins=hist_bins, range=[0,100])
# plt.show()
# plt.hist2d(sample_max_depths,sample_fpf,bins=hist_bins)
# plt.show()


#############################################################


from Dyck_Generator_Suzgun_Batch import DyckLanguage
vocab = ['(', ')']

NUM_PAR = 1
MIN_SIZE = 950
MAX_SIZE = 1000
P_VAL = 0.5
Q_VAL = 0.25


epsilon=0.5

Dyck = DyckLanguage(NUM_PAR, P_VAL, Q_VAL)

def encode_batch(sentences, labels, lengths, batch_size):
    max_length = max(lengths)
    # print(max_length)
    sentence_tensor = torch.zeros(batch_size, max_length, len(vocab))

    labels_tensor = torch.tensor([])
    for i in range(len(sentences)):
        # for i in range(batch_size):

        sentence = sentences[i]
        labels_tensor = torch.cat((labels_tensor, Dyck.lineToTensorSigmoid(labels[i], max_len=max_length)))
        # labels_tensor = torch.cat((labels_tensor,Dyck.batchToTensorSigmoid(labels,lengths,batch_size,max_length)))
        if len(sentence) < max_length:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab.index(char)
                sentence_tensor[i][index][pos] = 1

    # num_sequences = len(lengths)
    # if len(lengths)<batch_size:
    #     for j in range(batch_size-num_sequences):
    #         lengths.append(0)

    sentence_tensor.requires_grad_(True)
    # lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64).cpu()

    if len(lengths_tensor)<batch_size:
        for j in range(batch_size - len(sentences)):
            lengths_tensor=torch.cat((lengths_tensor,torch.tensor(0,dtype=torch.int64)))
    # print('labels tensor = ',labels_tensor)
    return sentence_tensor, labels_tensor, lengths_tensor


def collate_fn(batch):
    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    # max_depth = [batch[i]['max_depth'] for i in range(len(batch))]
    # print('labels in collate function  = ',labels)
    lengths = [len(sentence) for sentence in sentences]
    # batch.sort(key=lambda x:len(batch['x']), reverse=True)
    # batch.sort(key=lambda x:len(batch['length']), reverse=True)
    # max_depths = [batch[i]['max_depth'] for i in range(len(batch))]
    # # timestep_depths = [batch_size[i]['timestep_depths'] for i in range(len(batch))]
    # batch.sort(key=lambda x: lengths, reverse=True)


    # sentences = [batch[i]['x'] for i in range(len(batch))]
    # labels = [batch[i]['y'] for i in range(len(batch))]
    # # max_depth = [batch[i]['max_depth'] for i in range(len(batch))]
    # # print('labels in collate function  = ',labels)
    # lengths = [len(sentence) for sentence in sentences]
    # max_depths = [batch[i]['max_depth'] for i in range(len(batch))]
    # timestep_depths = [batch_size[i]['timestep_depths'] for i in range(len(batch))]

    print(len(sentences))
    print(len(labels))
    print(len(lengths))
    # print(len(max_depths))

    sentences.sort(key=len, reverse=True)
    labels.sort(key=len, reverse=True)
    lengths.sort(reverse=True)

    # max_depth.sort(reverse=True)

    # seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels,lengths, batch_size=len(sentences))
    seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=batch_size)

    # max_depths = []
    # timestep_depths = []
    # for i in range(len(batch)):
    #     max_depth, timestep_depth = get_timestep_depths(sentences[i])
    #     max_depths.append(max_depth)
    #     timestep_depths.append(timestep_depth)

    # max_depths_tensor = torch.tensor(max_depths,dtype=torch.float32)
    # timestep_depths_tensor = torch.tensor(timestep_depths,dtype=torch.float32)
    # return seq_tensor.to(device), labels_tensor.to(device), lengths_tensor.to(device)
    # return seq_tensor.to(device), labels_tensor.to(device), lengths_tensor.to(device), max_depths
    return sentences, labels, seq_tensor.to(device), labels_tensor.to(device), lengths_tensor


test_dataset = dataset2

# print(len(test_dataset))
batch_size = 50
shuffle_dataset = True
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=collate_fn)

print(test_loader)

for i, (sentences, labels, input_seq, target_seq, length) in enumerate(test_loader):
    print(length)
    print(sentences[i])
    print(labels[i])
    # print(length)
    print(get_timestep_depths(sentences[i]))
    break
