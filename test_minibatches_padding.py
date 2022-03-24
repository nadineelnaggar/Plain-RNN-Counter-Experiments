"""
- load data
- Create vocab dictionaries
- Padding and splitting into input/labels
- one-hot encoding
- define model
- train model
- evaluate model

"""


"""

"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import pandas as pd
import numpy as np
# from models import VanillaLSTM, VanillaGRU, VanillaRNN
from Dyck_Generator_Suzgun_Batch import DyckLanguage
from Dyck1_Datasets import NextTokenPredictionTrainDataset, NextTokenPredictionLongTestDataset, \
    NextTokenPredictionShortTestDataset, NextTokenPredictionValidationDataset
from torch.utils.data import Dataset, DataLoader
# import Dyck_Generator_Suzgun
# from Dyck_Generator_Suzgun import DyckLanguage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vocab = ['PAD','(', ')']
vocab = {'PAD':0, '(':1,')':2}
tags = {'':0, '0':1, '1':2}
n_letters= len(vocab)-1
n_tags = len(tags)-1
num_bracket_pairs = 25
length_bracket_pairs = 50

batch_size = 4

pad_token=0

# def encode_sentence(sentence, dataset='short'):
#     # max_length=1
#     # if dataset=='short' and model_name!='FFStack' and task=='BinaryClassification':
#     if dataset == 'short':
#         max_length=2*num_bracket_pairs
#     elif dataset=='long':
#         max_length=2*length_bracket_pairs
#     rep = torch.zeros(max_length,1,n_letters)
#     if len(sentence)<max_length:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[index+(max_length-len(sentence))][0][pos] = 1
#     else:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[index][0][pos]=1
#     rep.requires_grad_(True)
#     return rep, len(sentence)


# def encode_sentence(sentence, dataset='short'):
#     # max_length=1
#     # if dataset=='short' and model_name!='FFStack' and task=='BinaryClassification':
#     if dataset == 'short':
#         max_length=2*num_bracket_pairs
#     elif dataset=='long':
#         max_length=2*length_bracket_pairs
#     rep = torch.zeros(batch_size,max_length,n_letters)
#     if len(sentence)<max_length:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[0][index+(max_length-len(sentence))][pos] = 1
#     else:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[index][0][pos]=1
#     rep.requires_grad_(True)
#     return rep, len(sentence)


def encode_sentence(sentences, dataset='short'):
    if dataset=='short':
        max_length=2*num_bracket_pairs
    elif dataset=='long':
        max_length=2*length_bracket_pairs
    sentence_lengths = []
    seqs = [[vocab[word] for word in list(sentence)] for sentence in sentences]
    print(seqs)

    # rep = torch.zeros(batch_size, max_length)
    rep = np.ones((batch_size,max_length))*pad_token
    for i in range(batch_size):
        sentence = sentences[i]
        seq = seqs[i]
        # sentence = [[vocab[word] for word in sentence] for sentence in sentences]

        sentence_lengths.append(len(sentence))
        count=0
        rep[i][0:len(sentence)]=seq[:len(sentence)]
        # for char, idx in enumerate(sentence):

        # for char, idx in enumerate(sentence):
        #     if count<len(sentence):
        #         rep[i][idx] = vocab.index(char)

    return rep, sentence_lengths

# Dyck = DyckLanguage()

# def collate_fn(batch):
#     # get word tensor, label tensor and lengths tensor for
#     pass


# rep, length = encode_sentence(['()()', '((()))','(())()()', '()()((()))'], dataset='short')
rep, length = encode_sentence(['()()((()))','(())()()','((()))','()()'], dataset='short')
print(rep)
print(length)

rep = torch.from_numpy(rep)
# print(pack_sequence(rep))

packed = pack_padded_sequence(rep, length, batch_first=True)
print(packed)

vocab2 = ['(',')']

def encode_sentence_onehot(sentences, dataset='short'):
    # max_length = 50
    seq_lengths = [len(sentence) for sentence in sentences]
    print(seq_lengths)
    max_length = max(seq_lengths)
    print(max_length)
    rep = torch.zeros(batch_size,max_length,len(vocab2))
    lengths = []
    for i in range(batch_size):
        # for sentence in sentences:
        sentence = sentences[i]
        lengths.append(len(sentence))
        if len(sentence)<max_length:
            for index, char in enumerate(sentence):
                pos = vocab2.index(char)
                rep[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab2.index(char)
                # rep[index][0][pos]=1
                rep[i][index][pos]=1
    rep.requires_grad_(True)
    return rep, lengths

rep2, lengths2 = encode_sentence_onehot(['()()((()))','(())()()','((()))','()()'], dataset='short')
print(rep2)
print(lengths2)
padded2 = pack_padded_sequence(rep2, lengths=lengths2, batch_first=True)
print(padded2)

train_dataset = NextTokenPredictionTrainDataset()
val_dataset = NextTokenPredictionValidationDataset()
short_test_dataset = NextTokenPredictionShortTestDataset()
long_test_dataset = NextTokenPredictionLongTestDataset()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
for elem in train_dataset:
    print(elem)
    break

for elem in train_loader:
    print(elem)
    break


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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0, length):
        print('input x shape = ',x.shape)
        x = pack_padded_sequence(x, length, batch_first=True)
        h0 = self.init_hidden()
        print('packed sequence = ',x)
        x, h0 = self.lstm(x, h0)
        print('lstm output = ',x)
        # print('h0 = ',h0)
        x, _ = pad_packed_sequence(x, batch_first=True)
        print('padded packed sequence = ',x)
        print('len(x) = ',len(x))
        print('x.shape after lstm = ',x.shape)
        print(_)
        print(h0)
        # x = x.contiguous()
        # x = x.view(-1, x.shape[2])
        # x = x.view(batch_size, length)
        x = x.contiguous()
        print('x after contiguous = ',x)
        print('x.shape after contiguous = ',x.shape)
        x = x.view(-1, x.shape[2])
        print('x before linear layer = ',x)
        x = self.fc2(x)
        print('x after linear layer = ',x)
        print('x.shape after linear layer',x.shape)
        x = self.sigmoid(x).view(-1, self.output_size)
        print('x after sigmoid = ',x)
        # x = x.view(batch_size, length[0], self.input_size)

        # x = self.sigmoid(x)
        return x, h0

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))
    # def loss(self, Y_hat, Y, X_lengths):
    #     # TRICK 3 ********************************
    #     # before we calculate the negative log likelihood, we need to mask out the activations
    #     # this means we don't want to take into account padded items in the output vector
    #     # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    #     # and calculate the loss on that.
    #
    #     # flatten all the labels
    #     Y = Y.view(-1)
    #
    #     # flatten all predictions
    #     Y_hat = Y_hat.view(-1, self.nb_tags)
    #
    #     # create a mask by filtering out all tokens that ARE NOT the padding token
    #     tag_pad_token = self.tags['<PAD>']
    #     mask = (Y > tag_pad_token).float()
    #
    #     # count how many tokens we have
    #     nb_tokens = int(torch.sum(mask).data[0])
    #
    #     # pick the values for the label and zero out the rest with the mask
    #     Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
    #
    #     # compute cross entropy loss which ignores all <PAD> tokens
    #     ce_loss = -torch.sum(Y_hat) / nb_tokens
    #
    #     return ce_loss
    def mask(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        # Y = Y.view(-1)
        # print('Y = ',Y)
        # print(len(Y))

        # flatten all predictions
        # Y_hat = Y_hat.view(-1, self.nb_tags)
        print('Y_hat = ,',Y_hat)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()

        print(mask)


        # count how many tokens we have
        # nb_tokens = int(torch.sum(mask).data[0])
        nb_tokens = int(torch.sum(mask).item())
        # nb_tokens = int(torch.sum(mask))
        print(nb_tokens)

        # pick the values for the label and zero out the rest with the mask
        print(Y_hat.shape[0])
        # Y_hat2 = torch.tensor([])
        Y_hat2=[]
        # for i in range(Y_hat.shape[0]):
        #     # Y_hat[i] = Y_hat[i]*mask[i]
        #     Y_hat2.append(Y_hat[i]*mask[i])

        # print(Y_hat)

        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        #
        # return ce_loss
        return Y_hat
        # return torch.tensor([Y_hat2])

    def mask2(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        # Y = Y.view(-1)
        # print('Y = ',Y)
        # print(len(Y))

        # flatten all predictions
        # Y_hat = Y_hat.view(-1, self.nb_tags)
        print('Y.shape = ',Y.shape)
        print('Y_hat = ,',Y_hat)
        print('Y_hat.shape = ',Y_hat.shape)
        Y_hat_out = torch.zeros(Y_hat.shape)
        # Y_hat_out = torch.zeros(Y.shape)
        print('Y_hat_out = ',Y_hat_out)
        max_batch_length = max(X_lengths)

        # for i in range(Y_hat.shape[0]):
        #     for j in range(self.batch_size):

        for i in range(self.batch_size):
            # for j in range(Y_hat.shape[0]):
            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        # create a mask by filtering out all tokens that ARE NOT the padding token
        # tag_pad_token = self.tags['<PAD>']
        # mask = (Y > tag_pad_token).float()
        #
        # print(mask)


        # count how many tokens we have
        # nb_tokens = int(torch.sum(mask).data[0])
        # nb_tokens = int(torch.sum(mask).item())
        # nb_tokens = int(torch.sum(mask))
        # print(nb_tokens)

        # pick the values for the label and zero out the rest with the mask
        print(Y_hat.shape[0])
        # Y_hat2 = torch.tensor([])
        Y_hat2=[]
        # for i in range(Y_hat.shape[0]):
        #     # Y_hat[i] = Y_hat[i]*mask[i]
        #     Y_hat2.append(Y_hat[i]*mask[i])

        # print(Y_hat)

        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        # Y_hat = Y_hat[range(Y_hat.shape[0])] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        #
        # return ce_loss
        print('Y_hat_out.shape = ',Y_hat_out.shape)
        return Y_hat_out
        # return torch.tensor([Y_hat2])


model = VanillaLSTM(input_size=2, hidden_size=2, num_layers=1, output_size=2, batch_size=batch_size, output_activation='Sigmoid')

h0 = model.init_hidden()
print('h0 = ',h0)
out, h0 = model(rep2, h0, lengths2)
print('out = ',out)
print(out.shape)
print('h0 = ',h0)
Dyck = DyckLanguage(1,0.5, 0.25)
# print(model.loss(out))

init_ascii = 48
# def letterToIndex(self, letter):
#     return all_letters.find(letter)
# def lineToTensorSigmoid(line):
#     tensor = torch.zeros(len(line), n_letters)
#     openpar = ['(']
#     closepar = [')']
#     base = 1
#     for li, letter in enumerate(line):
#         for elt in openpar:
#             tensor[li][vocab[elt]-1] = 1.0
#
#         binary_code = ord(letter) - init_ascii
#
#         if binary_code > 0:
#             for base in range(len(closepar) - 1, -1, -1):
#                 if binary_code - (2 ** base) >= 0:
#                     tensor[li][letterToIndex(closepar[base])] = 1.0
#                     binary_code -= (2 ** base)
#     return tensor


print(Dyck.lineToTensorSigmoid('1110', max_len=4))

# targets = ['()()((()))','(())()()','((()))','()()']
targets = ['1010111110', '11101010', '111110', '1010']
# targets_enc = [Dyck.lineToTensorSigmoid(target, max_len=10) for target in targets]


targets_enc = torch.tensor([])
for i in range(len(targets)):
    targets_enc = torch.cat((targets_enc, Dyck.lineToTensorSigmoid(targets[i], max_len=10)))

print('targets_enc = ',targets_enc)

# targets_enc = torch.tensor(targets_enc)
# targets_enc = torch.tensor([])
# for i in range(len(targets)):
#     targets_enc = torch.cat((targets_enc,torch.tensor(Dyck.lineToTensorSigmoid(targets[i],max_len=10))))

# print(targets_enc)

# print(model.mask(out, targets_enc,lengths2))

out2 = model.mask(out, targets_enc, lengths2)
print(out2)

# criterion = nn.MSELoss()
# loss = criterion(out2, targets_enc)
# print(loss)
# loss.backward()

out3 = model.mask2(out, targets_enc, lengths2)
print('out3 = ',out3)
print('out3.shape = ',out3.shape)
criterion2 = nn.MSELoss()
loss2 = criterion2(out3,targets_enc)
print('loss2 = ',loss2)
loss2.backward()
out3 = out3.view(batch_size, length[0], n_letters)
targets_enc = targets_enc.view(batch_size, length[0], n_letters)

print('out3 = ',out3)
print('targets_enc = ',targets_enc)

out3_np = np.int_(out3.detach().cpu().numpy() >= 0.5)
target_np = np.int_(targets_enc.detach().cpu().numpy())

print('out3_np = ',out3_np)
print('target_np = ',target_np)
out4_np = np.copy(target_np)
out4_np[0,0]=0
print('out4_np = ',out4_np)


num_correct = 0
num_correct2 = 0
for j in range(batch_size):
    # print('out_np[j] = ',out_np[j])
    # print('out_np[j].shape = ',out_np[j].shape)
    # print('target_np[j] = ',target_np[j])
    # print('target_np[j].shape = ',target_np[j].shape)
    if np.array_equal(out4_np[j], target_np[j]):
        print('out4_np[', j, '] = target_np[', j, ']')
        print('out4_np[j] = ',out4_np[j])
        print('target_np[j] = ',target_np[j])
        num_correct2+=1


    # if np.equal(out_np[j].all(), target_np[j].all()).all():
    if np.array_equal(out3_np[j], target_np[j]):
        # print('output_np[j] = target_np[j]')
        print('out3_np[',j,'] = target_np[',j,']')
        #     count+=1
        #     print('count correct = ',count)
        # print('np.all(np.equal(out_np[j], target_np[j])) = ',np.all(np.equal(out_np[j],target_np[j])))
        # if np.all(np.equal(out_np[j], target_np[j])):
        # if np.all(np.equal(out_np[j], target_np[j])):
        # if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
        num_correct += 1
print('num_correct = ',num_correct)
print('num_correct2 = ',num_correct2)

def encode_batch(sentences, labels, lengths, batch_size, dataset='short'):
    # max_length = 50
    # seq_lengths = [len(sentence) for sentence in sentences]
    # print(seq_lengths)
    # max_length = max(seq_lengths)
    max_length = max(lengths)
    print(max_length)
    sentence_tensor = torch.zeros(batch_size,max_length,len(vocab2))
    # lengths = []
    labels_tensor = torch.tensor([])
    for i in range(batch_size):
        # for sentence in sentences:
        sentence = sentences[i]
        # labels_tensor = torch.cat((labels_tensor, Dyck.lineToTensorSigmoid(labels[i],max_len=max_length)))
        # labels_tensor = Dyck.batchToTensor(labels, batch_size=batch_size, max_len=max_length)
        # lengths.append(len(sentence))
        labels_tensor = torch.cat((labels_tensor,Dyck.batchToTensorSigmoid(labels,lengths,batch_size,max_length)))
        # print(Dyck.batchToTensorSigmoid([train_dataset2[i]['y'] for i in range(2)],
        #                                 lengths=[train_dataset2[i]['length'] for i in range(2)], batch_size=2,
        #                                 max_len=10))
        if len(sentence)<max_length:
            for index, char in enumerate(sentence):
                pos = vocab2.index(char)
                sentence_tensor[i][index][pos] = 1
        else:
            for index, char in enumerate(sentence):
                pos = vocab2.index(char)
                # rep[index][0][pos]=1
                sentence_tensor[i][index][pos]=1
    sentence_tensor.requires_grad_(True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    print('labels tensor = ',labels_tensor)
    return sentence_tensor, labels_tensor, lengths_tensor



# def sort_batch(seq_tensor, labels, tensor_lengths):
#     seqlen, perm_idx = tensor_lengths.sort(0, descending=True)
#     seq_tensor = seq_tensor[perm_idx]
#     print('seqlen = ',seqlen)
#     print('seq_tensor = ',seq_tensor)
#     print('perm_idx = ',perm_idx)
#     # labels = labels[perm_idx]
#     labels_tensor = torch.tensor([])
#     # max_len = seqlen
#     for i in range(len(perm_idx)):
#         print('i = ',i)
#         # print('val = ',val)
#         print('perm_idx[i] = ',perm_idx[i])
#         # labels_tensor = torch.cat((labels_tensor,labels[i*seqlen[i]]))
#     print('labels tensor after sort = ',labels)
#     print('labels tensor after sort == ',labels_tensor)
#     return seq_tensor, labels, seqlen




def collate_fn(batch):

    # word_tensor, label_tensor, tensor_lengths = [b.get('word') for b in batch], \
    #                                             [b.get('label') for b in batch], [b.get('len') for b in batch]

    # word_tensor =
    # labels = torch.cat(label_tensor)
    # tensor_lengths = torch.tensor(tensor_lengths, dtype=torch.long)

    sentences = [batch[i]['x'] for i in range(len(batch))]
    labels = [batch[i]['y'] for i in range(len(batch))]
    print('labels in collate function  = ',labels)
    # lengths = [batch[i]['length'] for i in range(len(batch))]
    lengths = [len(sentence) for sentence in sentences]

    sentences.sort(key=len, reverse=True)
    labels.sort(key=len,reverse=True)
    lengths.sort(reverse=True)
    # lengths = [len(sentence) for sentence in sentences]

    # sentences = [b['x'] for b in batch]
    # labels = [b['y'] for b in batch]
    # # lengths = [batch[i]['length'] for i in range(len(batch))]
    # lengths = [len(sentence) for sentence in sentences]

    # seq_tensor = self.pad_sequences(word_tensor, tensor_lengths)
    # seq_tensor, labels, tensor_lengths = self.sort_batch(seq_tensor,labels,tensor_lengths)

    seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels,lengths, batch_size=len(sentences))
    # seq_tensor, labels_tensor, lengths_tensor = sort_batch(seq_tensor,labels_tensor, lengths_tensor)

    # seq_tensor, labels_tensor, lengths_tensor = sort_batch(seq_tensor, labels_tensor, lengths_tensor)
    # seq_tensor, labels_tensor, lengths_tensor = encode_batch(sentences, labels, lengths, batch_size=len(sentences))


    return seq_tensor.to(device), labels_tensor.to(device), lengths_tensor.to(device)
    # return seq_tensor.to(device), labels_tensor.to(device)
    # return seq_tensor.to(device), labels.to(device), tensor_lengths.to(device)



train_dataset2 = NextTokenPredictionTrainDataset()
train_loader2 = DataLoader(train_dataset2,batch_size=2, shuffle=False, collate_fn=collate_fn)

# print('test collate function = ')

# print(collate_fn(train_dataset2))
epsilon=0.5

model_ = VanillaLSTM(input_size=2, hidden_size=3, batch_size=2, num_layers=1, output_size=2, output_activation='Sigmoid')
for i, (seq, target_seq, length) in enumerate(train_loader2):
    num_correct = 0
    # print('seq = ',seq)
    print('input_seq shape = ',seq.shape)
    # print('label = ', target_seq)
    print('label shape = ',target_seq.shape)
    print('length = ',length)
    print('seq length = ',seq.shape)
    # out, _ = model(seq,model.init_hidden(), length)
    # print('out = ',out)
    # print('out.shape = ', out.shape)
    print('target seq length = ', target_seq.shape)

    # # print('length = ',length)
    #
    # # output_seq = model(seq, model.init_hidden(), length)
    # # break
    # output_seq, _ = model_(seq, model_.init_hidden(), length)
    # print('output_seq.shape before mask = ',output_seq.shape)
    # output_seq = model_.mask2(output_seq,target_seq,length)
    # print('output_seq.shape after mask = ',output_seq.shape)
    # # output_seq = output_seq.view(batch_size, length[0], n_letters)
    # # target_seq = target_seq.view(batch_size, length[0], n_letters)
    #
    # print('output_seq.shape = ',output_seq.shape)
    # print('target_seq.shape = ',target_seq.shape)
    #
    # out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
    # target_np = np.int_(target_seq.detach().cpu().numpy())
    #
    # print('out_np = ',out_np)
    # print('target_np = ',target_np)
    # print('out_np.shape = ', out_np.shape)
    # print('target_np.shape = ',target_np.shape)
    #
    # # print('out_np = ',out_np)
    # # print('target_np = ',target_np)
    # # print('flattened output np = ',out_np.flatten())
    # # print('flattened target np = ', target_np.flatten())
    # max_len = max(length)
    # print('max_len = ',max_len)
    # for j in range(batch_size):
    #
    #     print('out_np[j*max_len:J*max_len+max_len] = ',out_np[(j*max_len):(j*max_len)+max_len])
    #     print('target_np[j*max_len:J*max_len+max_len] = ',target_np[j*max_len:(j*max_len)+max_len])
    #
    #     if np.all(np.equal(out_np[j*max_len:(j*max_len)+max_len], target_np[j*max_len:(j*max_len)+max_len])):
    #         num_correct += 1
    #         print('TRUE')
    #     print('num_correct = ',num_correct)

        # if np.all(np.equal(out_np[j*max_len:(j*max_len)+max_len], target_np[j*max_len:(j*max_len)+max_len])) and (out_np[j*max_len:(j*max_len)+max_len].flatten() == target_np[j*max_len:(j*max_len)+max_len].flatten()).all():
        #     num_correct += 1
        #     print('TRUE')
        # print('num_correct = ',num_correct)
    if i == 5:
        break

#
# epsilon=0.5
# output_seq = output_seq.view(batch_size, length[0], n_letters)
# target_seq = target_seq.view(batch_size, length[0], n_letters)
#
# out_np = np.int_(output_seq.detach().cpu().numpy() >= epsilon)
# target_np = np.int_(target_seq.detach().cpu().numpy())
#
# num_correct=0
#
# # print('out_np = ',out_np)
# # print('target_np = ',target_np)
# # print('flattened output np = ',out_np.flatten())
# # print('flattened target np = ', target_np.flatten())
# for j in range(batch_size):
#
#     if np.all(np.equal(out_np[j], target_np[j])) and (out_np[j].flatten() == target_np[j].flatten()).all():
#         num_correct += 1
#         # epoch_correct_guesses.append(X[i])
#
# print(num_correct)

# print(Dyck.batchToTensorSigmoid([train_dataset2[i]['y'] for i in range(2)], lengths=[train_dataset2[i]['length'] for i in range(2)],batch_size=2,max_len=10))




# example =

# print(lengths2)
# for i in range(len(rep2)):
#     out, _ = model(rep2[i], lengths2)
#     print(out)



# print(out)


# for elem in train_loader:
#     print(elem)
#     # hidden = model.init_hidden()
#     # elem = torch.tensor(elem)
#     # out, _ = model(elem, hidden, lengths2)
#     # print(out)
#     break



"""
- load data
- Create vocab dictionaries
- create collate function and create dataloader
    - encode sentence with padding already
    - 
- Padding and splitting into input/labels
- one-hot encoding
- define model
- train model
- evaluate model
"""


# train_loader = DataLoader(NextTokenPredictionTrainDataset,batch_size=1, shuffle=False)
# val_loader = DataLoader(NextTokenPredictionValidationDataset, batch_size=1, shuffle=False)
# short_test_loader = DataLoader(NextTokenPredictionShortTestDataset, batch_size=1, shuffle=False)
# long_test_loader = DataLoader(NextTokenPredictionLongTestDataset, batch_size=1, shuffle=False)
#
# for elem, i in enumerate(train_loader):
#     print(elem)
#     break







