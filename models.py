import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VanillaRNN, self).__init__()
        self.model_name='VanillaRNN'
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.fc1=nn.Linear(input_size,hidden_size)
        self.rnn = nn.RNN(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x, h0):
        x = self.fc1(x)
        x, h0 = self.rnn(x.unsqueeze(dim=0), h0)
        x = self.fc2(x)
        x = x.squeeze()
        x = x.squeeze()
        x = self.softmax(x)
        return x, h0

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VanillaLSTM, self).__init__()
        self.model_name='VanillaLSTM'
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.fc1=nn.Linear(input_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x, h0):
        # print('input x to model',x)
        x = self.fc1(x)
        # print('x after input weights',x)
        # print('input hidden',h0)
        x, h0 = self.lstm(x.unsqueeze(dim=0), h0)
        # print('x after lstm',x)
        # print('hidden after lstm',h0)
        x = self.fc2(x)
        x = x.squeeze()
        x = x.squeeze()
        # print('x after output weights',)
        x = self.softmax(x)
        # print('x after softmax',x)

        return x, h0

class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VanillaGRU, self).__init__()
        self.model_name='VanillaLSTM'
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.fc1=nn.Linear(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x, h0):
        # print('input x to model',x)
        x = self.fc1(x)
        # print('x after input weights',x)
        # print('input hidden',h0)
        x, h0 = self.gru(x.unsqueeze(dim=0), h0)
        # print('x after lstm',x)
        # print('hidden after lstm',h0)
        x = self.fc2(x)
        x = x.squeeze()
        x = x.squeeze()
        # print('x after output weights',)
        x = self.softmax(x)
        # print('x after softmax',x)

        return x, h0