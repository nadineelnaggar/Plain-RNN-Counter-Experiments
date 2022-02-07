import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, output_activation='Sigmoid'):
        super(VanillaRNN, self).__init__()
        self.model_name='VanillaRNN'
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.output_activation=output_activation

        self.fc1=nn.Linear(input_size,hidden_size)
        # self.rnn = nn.RNN(hidden_size,hidden_size)
        self.rnn = nn.RNN(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x, h0):
        # x = self.fc1(x)
        x, h0 = self.rnn(x.unsqueeze(dim=0), h0)
        x = self.fc2(x)
        x = x.squeeze()
        x = x.squeeze()
        if self.output_activation=='Softmax':
            x = self.softmax(x)
        elif self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
            # x = x.squeeze()
        return x, h0

# class VanillaLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, output_activation='Sigmoid'):
#         super(VanillaLSTM, self).__init__()
#         self.model_name='VanillaLSTM'
#         self.hidden_size=hidden_size
#         self.num_layers=num_layers
#         self.output_activation=output_activation
#
#         self.fc1=nn.Linear(input_size,hidden_size)
#         # self.lstm = nn.LSTM(hidden_size,hidden_size)
#         self.lstm = nn.LSTM(input_size,hidden_size)
#         self.fc2 = nn.Linear(hidden_size,num_classes)
#         self.softmax=nn.Softmax(dim=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, h0):
#         # print('input x to model',x)
#         # x = self.fc1(x)
#         # print('x after input weights',x)
#         # print('input hidden',h0)
#         x, h0 = self.lstm(x.unsqueeze(dim=0), h0)
#         # print('x after lstm',x)
#         # print('hidden after lstm',h0)
#         x = self.fc2(x)
#         # x = x.squeeze()
#         # x = x.squeeze()
#         # # print('x after output weights',)
#         # if self.output_activation=='Softmax':
#         #     x = self.softmax(x)
#         # elif self.output_activation=='Sigmoid':
#         #     x = self.sigmoid(x)
#         # print('x after softmax',x)
#
#         return x, h0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_activation='Sigmoid'):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.output_activation=output_activation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0):

        x, h0 = self.lstm(x, h0)
        x = self.fc2(x)
        x = self.sigmoid(x).view(-1, self.output_size)
        return x

    def init_hidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, 1, self.hidden_dim).to(device))

class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, output_activation='Sigmoid'):
        super(VanillaGRU, self).__init__()
        self.model_name='VanillaLSTM'
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_activation=output_activation

        self.fc1=nn.Linear(input_size,hidden_size)
        # self.gru = nn.GRU(hidden_size,hidden_size)
        self.gru = nn.GRU(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.softmax=nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0):
        # print('input x to model',x)
        # x = self.fc1(x)
        # print('x after input weights',x)
        # print('input hidden',h0)
        x, h0 = self.gru(x.unsqueeze(dim=0), h0)
        # print('x after lstm',x)
        # print('hidden after lstm',h0)
        x = self.fc2(x)
        x = x.squeeze()
        x = x.squeeze()
        # print('x after output weights',)

        if self.output_activation=='Softmax':
            x = self.softmax(x)
        elif self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
        # print('x after softmax',x)

        return x, h0