import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
from Dyck_Generator_Suzgun_Batch import DyckLanguage

vocab2 = ['(',')']

def encode_sentence_onehot(sentence, dataset='short'):
    max_length = 50
    # seq_lengths = [len(sentence) for sentence in sentences]
    # print(seq_lengths)
    # max_length = max(seq_lengths)
    # print(max_length)
    rep = torch.zeros(1,max_length,len(vocab2))
    lengths = []


    for index, char in enumerate(sentence):
        pos = vocab2.index(char)
        rep[0][index][pos] = 1

    rep.requires_grad_(True)
    return rep


Dyck = DyckLanguage(1,0.5, 0.25)

class NextTokenPredictionTrainDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:10000]
        self.y = self.y[:10000]

        self.x_tensor = []
        self.y_tensor = []
        for i in range(len(self.x)):
            self.x_tensor.append(encode_sentence_onehot(self.x[i]))
            self.y_tensor.append(Dyck.lineToTensorSigmoid(self.y[i], max_len=50))

        self.lengths = self.lengths[:10000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}
        # return {'x':encode_sentence_onehot(self.x[index]), 'y': Dyck.lineToTensorSigmoid(str(self.y[index]), max_len=50), 'length': self.lengths[index]}
        # return {'x':self.x[index], 'y':self.y[index]}

    def __len__(self):
        return self.n_samples

# dataset = NextTokenPredictionTrainDataset()
# print(len(dataset))


class NextTokenPredictionShortTestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[10000:15000]
        self.y = self.y[10000:15000]
        self.lengths = self.lengths[10000:15000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples
# dataset_test = NextTokenPrediction_Short_Test_Dataset()
# print(len(dataset_test))


class NextTokenPredictionValidationDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(sentence)

        self.x = self.x[15000:]
        self.y = self.y[15000:]
        self.lengths = self.lengths[15000:]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_val = NextTokenPrediction_Validation_Dataset()
# print(len(dataset_val))


class NextTokenPredictionLongTestDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_test_.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        self.x = self.x[:5000]
        self.y = self.y[:5000]
        self.lengths = self.lengths[:5000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples


class NextTokenPredictionDataset102to500tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_102to500tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples


class NextTokenPredictionDataset502to1000tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                self.x.append(sentence)
                self.y.append(label)
                self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples



class NextTokenPredictionDataset950to1000tokens(Dataset):
    def __init__(self):
        # xy = np.loadtxt('Dyck1_Dataset_Suzgun_train_.txt', delimiter=",")
        # self.x = torch.from_numpy(xy[:,0])
        # self.y = torch.from_numpy(xy[:, [1]])
        self.x = []
        self.y = []
        self.lengths = []
        # self.n_samples = xy.shape[0]
        with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                if len(sentence)>=950:
                    self.x.append(sentence)
                    self.y.append(label)
                    self.lengths.append(len(sentence))

        # self.x = self.x[:5000]
        # self.y = self.y[:5000]
        # self.lengths = self.lengths[:5000]

        # self.x = self.x[:1000]
        # self.y = self.y[:1000]
        # self.lengths = self.lengths[:1000]

        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x':self.x[index], 'y':self.y[index], 'length':self.lengths[index]}

    def __len__(self):
        return self.n_samples

# dataset_long = NextTokenPredictionLongTestDataset()
# print(len(dataset_long))
# print(dataset_long[20])
# print(dataset_long[20]['x'])