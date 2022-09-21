import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Dyck1_Datasets import NextTokenPredictionDataset2000tokens_zigzag


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaLSTM'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.lstm(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x, h0

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))



    def mask(self, Y_hat, Y, X_lengths):
        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        Y_hat2=[]

        return Y_hat_out.to(device)




class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaRNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x, h0

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        Y_hat2=[]

        return Y_hat_out.to(device)



class VanillaReLURNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaReLURNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu')
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x, h0

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)




    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)



        for i in range(self.batch_size):

            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        Y_hat2=[]

        return Y_hat_out.to(device)




class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid'):
        super(VanillaGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaGRU'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()

        x, h0 = self.gru(x, h0)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x, h0

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)


    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)
        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]

        Y_hat2=[]

        return Y_hat_out.to(device)



def get_timestep_depths(x):
    max_depth = 0
    current_depth = 0
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


lstm_model = VanillaLSTM(input_size=2, hidden_size=1, num_layers=1, batch_size=1, output_size=2, output_activation='Sigmoid')
gru_model = VanillaGRU(input_size=2, hidden_size=1, num_layers=1, batch_size=1, output_size=2, output_activation='Sigmoid')
relu_model = VanillaReLURNN(input_size=2, hidden_size=1, num_layers=1, batch_size=1, output_size=2, output_activation='Sigmoid')


"""
1. choose which models are needed of LSTM, RELU, GRU
2. import them and run one zigzag sequence through them
3. keep track of the output activation and store it in a list
4. plot timestep depth of the sequence vs the output activation 
5. save image
"""
