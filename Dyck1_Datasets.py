import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader


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
        self.lengths = self.lengths[:10000]
        self.n_samples = len(self.x)



    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return {'x': self.x[index], 'y': self.y[index], 'length': self.lengths[index]}
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

# dataset_long = NextTokenPredictionLongTestDataset()
# print(len(dataset_long))
# print(dataset_long[20])
# print(dataset_long[20]['x'])