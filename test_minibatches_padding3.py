import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
# sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']
sent_1_x = ['(', '(', ')', '(', ')', ')', '(', ')']
sent_1_y = ['incomplete', 'incomplete', 'incomplete', 'incomplete','incomplete', 'valid', 'incomplete', 'valid']




sent_2_x = ['(', ')', '(', ')']
sent_2_y = ['incomplete', 'valid','incomplete', 'valid']

sent_3_x = ['(', ')']
sent_3_y = ['incomplete', 'valid']

X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

print(X)
print(Y)


# map sentences to vocab
# vocab = {'': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
vocab = {'':0, '(':1, ')':2}


# fancy nested list comprehension
X =  [[vocab[word] for word in sentence] for sentence in X]

print('X = ')
print(X)

# X now looks like:
# [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]


# tags = {'': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
tags = {'':0, 'valid':1, 'incomplete':2}



# fancy nested list comprehension
Y =  [[tags[tag] for tag in sentence] for sentence in Y]


print('Y = ')
print(Y)

# Y now looks like:
# [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]


X_lengths  = [len(sentence) for sentence in X]
print(X_lengths)

# create an empty matrix with padding tokens
pad_token = vocab['']
longest_sent = max(X_lengths)
batch_size = len(X)
padded_X = np.ones((batch_size, longest_sent)) * pad_token

# copy over the actual sequences
for i, x_len in enumerate(X_lengths):
    sequence = X[i]
    padded_X[i, 0:x_len] = sequence[:x_len]

# # padded_X looks like:
# array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.],
#        [ 8.,  8.,  0.,  0.,  0.,  0.,  0.],
#        [ 7.,  9.,  0.,  0.,  0.,  0.,  0.]])


print('padded_X = ')
print(padded_X)

# get the length of each sentence
Y_lengths = [len(sentence) for sentence in Y]

# create an empty matrix with padding tokens
pad_token = tags['']
longest_sent = max(Y_lengths)
batch_size = len(Y)
padded_Y = np.ones((batch_size, longest_sent)) * pad_token

# copy over the actual sequences
for i, y_len in enumerate(Y_lengths):
    sequence = Y[i]
    padded_Y[i, 0:y_len] = sequence[:y_len]

# padded_Y looks like:
# array([[ 1.,  2.,  3.,  3.,  3.,  1.,  4.],
#        [ 5.,  5.,  0.,  0.,  0.,  0.,  0.],
#        [ 4.,  5.,  0.,  0.,  0.,  0.,  0.]])


print('padded_Y = ')
print(padded_Y)


class BieberLSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=3, batch_size=3):
        super(BieberLSTM, self).__init__()
        # self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8,
        #               'yeah': 9}
        self.vocab = {'<PAD>':0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
        self.tags = {'<PAD>':0, 'valid':1, 'incomplete':2}
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # don't count the padding tag for the classifier output
        self.nb_tags = len(self.tags) - 1

        # when the model is bidirectional we double the output dimension
        self.lstm= nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
        )
        self.nb_vocab_words = len(self.vocab)
        self.padding_idx = self.vocab['<PAD>']

        self.word_embedding = nn.Embedding(
            num_embeddings=self.nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

        # build actual NN
    #     self.__build_model()
    #
    # def __build_model(self):
    #     # build embedding layer first
    #     nb_vocab_words = len(self.vocab)
    #
    #     # whenever the embedding sees the padding index it'll make the whole vector zeros
    #     padding_idx = self.vocab['<PAD>']
    #     self.word_embedding = nn.Embedding(
    #         num_embeddings=nb_vocab_words,
    #         embedding_dim=self.embedding_dim,
    #         padding_idx=padding_idx
    #     )
    #
    #     # design LSTM
    #     self.lstm = nn.LSTM(
    #         input_size=self.embedding_dim,
    #         hidden_size=self.nb_lstm_units,
    #         num_layers=self.nb_lstm_layers,
    #         batch_first=True,
    #     )
    #
    #     # output layer which projects back to tag space
    #     self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_layers, self.batch_size, self.nb_lstm_units).to(device)
        hidden_b = torch.randn(self.nb_layers, self.batch_size, self.nb_lstm_units).to(device)

        # if self.hparams.on_gpu:
        #     hidden_a = hidden_a.cuda()
        #     hidden_b = hidden_b.cuda()
        #
        # hidden_a = Variable(hidden_a)
        # hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        # batch_size, seq_len, _ = X.size()
        # batch_size= seq_len= _ = len(X)
        # seq_len = sum(X_lengths)
        seq_len=7

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(self.batch_size, seq_len, self.nb_tags)
        X = X.view(self.batch_size, max(X_lengths), self.nb_tags)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.view(-1)
        # print('Y = ',Y)
        # print(len(Y))

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.nb_tags)
        # print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()
        # print(mask)


        # count how many tokens we have
        # nb_tokens = int(torch.sum(mask).data[0])
        nb_tokens = int(torch.sum(mask).item())
        # nb_tokens = int(torch.sum(mask))
        # print(nb_tokens)

        # pick the values for the label and zero out the rest with the mask
        # print(Y_hat.shape[0])

        for i in range(Y_hat.shape[0]):
            Y_hat[i] = Y_hat[i]*mask[i]
        # print(Y_hat)

        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        #
        # return ce_loss
        Y_hat = Y_hat.view(self.batch_size, max(X_lengths), self.nb_tags)
        return Y_hat


model = BieberLSTM(nb_layers=1)

model_input = torch.tensor(padded_X, dtype=torch.long)
out = model(model_input, X_lengths)
# out = model(X, X_lengths)
print('out = ')
print(out)
print('padded_Y = ')
print(torch.tensor(padded_Y))
tensor_padded_Y = torch.tensor(padded_Y, dtype=torch.long)
print('tensor_padded_Y = ')
print(tensor_padded_Y)
print('masked output = ')
print(model.loss(out, tensor_padded_Y, X_lengths))
masked_out = model.loss(out, tensor_padded_Y, X_lengths)
print(out.shape)
print(masked_out.shape)
print(padded_Y.shape)
print(tensor_padded_Y.shape)

# criterion = nn.MSELoss()
# optimiser = torch.optim.Adam(model.parameters(),lr=0.01)
# loss = criterion(out, tensor_padded_Y)
# loss.backward()
# print(model.loss(out, torch.tensor(padded_Y,dtype=torch.float32), X_lengths))




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
        self.tags = {'<PAD>':0, 'valid':1, 'incomplete':2}
        self.nb_tags = len(self.vocab)-1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        print('X before packing = ',x)
        x = pack_padded_sequence(x, length, batch_first=True)
        print('packed X = ',x)
        h0 = self.init_hidden()
        print('h0 = ',h0)


        x, h0 = self.lstm(x, h0)
        print('x after lstm = ',x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        print('x after pad packed sequence = ',x)

        x = x.contiguous()
        print('x after contiguous = ',x)

        x = x.view(-1, x.shape[2])
        print('x before output weights = ',x)

        x = self.fc2(x)
        print('x after output weights and before sigmoid = ',x)

        x = self.sigmoid(x).view(-1, self.output_size)
        print('x after sigmoid = ',x)
        x = x.view(self.batch_size, max(X_lengths), self.nb_tags)
        print('x after view = ',x)

        return x

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

        # print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()
        # print(Y_hat.shape)
        # print(Y.shape)
        # print(mask.shape)


        # count how many tokens we have

        nb_tokens = int(torch.sum(mask).item())


        # pick the values for the label and zero out the rest with the mask

        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask


        return Y_hat

print('######################################')
# model = BieberLSTM(nb_layers=1)
model2 = VanillaLSTM(input_size=1, hidden_size=3, num_layers=1, batch_size=3, output_size=1)
print(model2)

model_input = torch.tensor(padded_X, dtype=torch.long)
print(model_input)
out = model2(model_input, X_lengths)
out = model(X, X_lengths)
print('out = ')
print(out)
print('padded_Y = ')
print(torch.tensor(padded_Y))
tensor_padded_Y = torch.tensor(padded_Y, dtype=torch.long)
print('tensor_padded_Y = ')
print(tensor_padded_Y)
print('masked output = ')
# print(model.loss(out, tensor_padded_Y, X_lengths))
masked_out = model2.mask(out, tensor_padded_Y, X_lengths)
print(masked_out)
print(out.shape)
print(masked_out.shape)
print(padded_Y.shape)
print(tensor_padded_Y.shape)
print(tensor_padded_Y.view(-1))


#######################################################################################

# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
# import pandas as pd
# import numpy as np
# # from models import VanillaLSTM, VanillaGRU, VanillaRNN
# from Dyck_Generator_Suzgun_Batch import DyckLanguage
# from Dyck1_Datasets import NextTokenPredictionTrainDataset, NextTokenPredictionLongTestDataset, \
#     NextTokenPredictionShortTestDataset, NextTokenPredictionValidationDataset
# from torch.utils.data import Dataset, DataLoader
# # import Dyck_Generator_Suzgun
# # from Dyck_Generator_Suzgun import DyckLanguage
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # vocab = ['PAD','(', ')']
# vocab = {'PAD':0, '(':1,')':2}
# tags = {'':0, '0':1, '1':2}
# n_letters= len(vocab)-1
# n_tags = len(tags)-1
# num_bracket_pairs = 25
# length_bracket_pairs = 50