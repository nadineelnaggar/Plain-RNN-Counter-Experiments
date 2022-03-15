import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.



        # flatten all predictions

        print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()



        # count how many tokens we have

        nb_tokens = int(torch.sum(mask).item())


        # pick the values for the label and zero out the rest with the mask

        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask


        return Y_hat



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
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.



        # flatten all predictions

        print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()



        # count how many tokens we have

        nb_tokens = int(torch.sum(mask).item())


        # pick the values for the label and zero out the rest with the mask

        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask


        return Y_hat



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
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.



        # flatten all predictions

        print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()



        # count how many tokens we have

        nb_tokens = int(torch.sum(mask).item())


        # pick the values for the label and zero out the rest with the mask

        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask


        return Y_hat