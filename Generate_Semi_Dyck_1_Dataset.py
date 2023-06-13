import sys
import numpy as np
import torch
from collections import defaultdict
import random
from random import randint
from random import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

x = []
y = []
lengths = []
with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        x.append(sentence)
        y.append(label)
        lengths.append(len(sentence))


#train set

x_train = x[:10000]
y_train = y[:10000]
lengths_train = lengths[:10000]

#short test (2-50 tokens)
x_short_test = x[10000:15000]
y_short_test = y[10000:15000]
lengths_short_test = lengths[10000:15000]


#validation set (2-50 tokens)
x_val = x[15000:]
y_val = y[15000:]
lengths_val = lengths[15000:]

#test set (52-100 tokens)
x_test = []
y_test = []
lengths_test = []

with open('Dyck1_Dataset_Suzgun_test_.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        x_test.append(sentence)
        y_test.append(label)
        lengths_test.append(len(sentence))

x_test = x_test[:5000]
y_test = y_test[:5000]
lengths_test = lengths_test[:5000]


#long test set (1000 tokens)
x_long = []
y_long = []
lengths_long = []

with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            x_long.append(sentence)
            y_long.append(label)
            lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_2.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_3.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_4.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_5.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_5 (1).txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))

with open('Dyck1_Dataset_Suzgun_1000tokens_6.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence) == 1000:
            if sentence not in x_long:
                x_long.append(sentence)
                y_long.append(label)
                lengths_long.append(len(sentence))



x_long = x_long[:100]
y_long = y_long[:100]
lengths_long = lengths_long[:100]



#zigzag dataset (2000 tokens)

x_zigzag = []
y_zigzag = []
max_depths_zigzag= []


lengths_zigzag = []
# self.n_samples = xy.shape[0]
with open('Dyck1_Dataset_Suzgun_2000tokens_zigzag.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        length = line[2].strip()

        x_zigzag.append(sentence)
        y_zigzag.append(label)
        lengths_zigzag.append(length)



def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
def flipOpeningBrackets(seqs):
    threshold=0.5
    flip_probability=0.0

    for i in range(len(seqs)):
        # seq = seqs[indices[i]]
        seq = seqs[i]
        count_open = len(seq)/2
        indices_open = find(seq, '(')
        # print(indices_open)
        # label = labels[i]


        for j in range(len(seq)):
            if seq[j]=='(':
                flip_probability=random()
                # print('flip probability = ',flip_probability)
                if flip_probability>=threshold:
                    # print('old sequence = ',seq)
                    seq = seq[:j]+')'+seq[j+1:]
                    # print('changed sequence = ',seq)
                # elif flip_probability<threshold:
                    # print('no change')
                    # print('unchanged sequence = ',seq)



        seqs[i] = seq
        # print('seqs[i] = ',seqs[i])

    return seqs

# flipOpeningBrackets([x[0], x[1], x[2], x[3]])

def removeOpeningBracket(seqs):
    # seqs_new = []
    print(len(seqs))
    for i in range(len(seqs)):
        seq = seqs[i]
        # print('original sequence = ',seq)
        # print('original sequence length = ',len(seq))
        open_bracket_indices = find(seq, '(')
        # print('open bracket indices = ',open_bracket_indices)
        rand=randint(0, len(open_bracket_indices)-1)
        # print('rand = ',rand)
        # print('len(opening_bracket_indices) = ',len(open_bracket_indices))
        changed_index = open_bracket_indices[rand]
        # print('changed_index = ',changed_index)
        seq = seq[:changed_index]+''+seq[changed_index+1:]
        # print('changed sequence = ',seq)
        # print('changed sequence length = ',len(seq))
        # print('number of opening brackets = ',seq.count('('))
        # print('number of closing brackets = ',seq.count(')'))
        seqs[i] = seq
        # seqs_new.append(seq)
    return seqs
    # return seqs_new
def addClosingBracket(seqs):
    for i in range(len(seqs)):
        # threshold=0.5
        # addition_probability =
        seq = seqs[i]
        # print('original sequence = ',seq)
        # print('original sequence length = ',len(seq))
        changed_index = randint(0, len(seq))
        # print('changed_index = ',changed_index)
        seq = seq[:changed_index]+')'+seq[changed_index:]
        # print('changed sequence = ',seq)
        # print('changed sequence length = ',len(seq))
        # print('number of opening brackets = ',seq.count('('))
        # print('number of closing brackets = ',seq.count(')'))
        seqs[i] = seq

    return seqs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VanillaReLURNN__(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, output_size, output_activation='Sigmoid', rnn_input_weight=[1,-1], rnn_hidden_weight=[1]):
        super(VanillaReLURNN__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size=batch_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.model_name = 'VanillaReLURNN'

        self.vocab = {'<PAD>': 0, '(':1, ')':2}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}r
        self.tags = {'<PAD>':0, '0':1, '1':2}
        self.nb_tags = len(self.vocab)-1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity='relu', bias=False)
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.rnn.weight_ih_l0=nn.Parameter(torch.tensor([rnn_input_weight], dtype=torch.float32))
        self.rnn.weight_hh_l0=nn.Parameter(torch.tensor([rnn_hidden_weight], dtype=torch.float32))
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.weight=nn.Parameter(torch.tensor([[1],[1]],dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor([1,-0.5], dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True)
        # print(x)
        h0 = self.init_hidden()

        x, h0 = self.rnn(x, h0)
        # print(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])

        x = self.fc2(x)
        # print(x)

        x = self.sigmoid(x).view(-1, self.output_size)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)



    def mask(self, Y_hat, Y, X_lengths):

        Y_hat_out = torch.zeros(Y_hat.shape)

        max_batch_length = max(X_lengths)


        for i in range(self.batch_size):

            Y_hat_out[i*max_batch_length:(i*max_batch_length+X_lengths[i])] = Y_hat[i*max_batch_length:(i*max_batch_length+X_lengths[i])]


        return Y_hat_out.to(device)
        # return torch.tensor([Y_hat2])



def createSemiDyck1Labels(seqs):
    labels = []

    for i in range(len(seqs)):
        # count = 0
        non_negative_count = 0
        seq=seqs[i]
        label=''
        for j in range(len(seq)):
            if seq[j]=='(':
                # count+=1
                non_negative_count+=1
            elif seq[j]==')':
                non_negative_count-=1
                if non_negative_count<0:
                    non_negative_count=0
            if non_negative_count>0:
                label=label+'1'
            elif non_negative_count==0:
                label = label+'0'
        labels.append(label)

    return labels


# print(createSemiDyck1Labels(flipOpeningBrackets([x[0], x[1]])))


def makeSemiDyck1Dataset(dataset_name):
    x = []
    y = []
    # labels = []

    if dataset_name=='train':
        x = x_train
        y = y_train
        filename = 'SemiDyck1_Dataset_train.txt'
    elif dataset_name=='val':
        x = x_val
        y = y_val
        filename = 'SemiDyck1_Dataset_val.txt'
    elif dataset_name=='short_test':
        x = x_short_test
        y = y_short_test
        filename = 'SemiDyck1_Dataset_short_test.txt'
    elif dataset_name=='test':
        x = x_test
        y = y_test
        filename = 'SemiDyck1_Dataset_test.txt'
    elif dataset_name=='long_test':
        x = x_long
        y = y_long
        filename = 'SemiDyck1_Dataset_long_test.txt'
    elif dataset_name=='zigzag':
        x = x_zigzag
        y = y_zigzag
        filename = 'SemiDyck1_Dataset_zigzag.txt'

    n_samples = len(x)
    print('n_samples = ',n_samples)
    n_flipped = int(n_samples/2)
    n_added = int(n_samples/4)
    n_removed = int(n_samples/4)

    x_flipped = x[:n_flipped]
    x_added = x[n_flipped:(n_flipped+n_added)]
    x_removed = x[(n_flipped+n_added):(n_flipped+n_added+n_removed)]
    semiDyckFlipped = []

    # print('x_flipped = ',x_flipped)
    print('len(x_flipped) = ',len(x_flipped))
    # print('n_flipped = ',n_flipped)
    # print('x_added = ',x_added)
    print('len(x_added) = ', len(x_added))
    # print('n_added = ', n_added)
    # print('x_removed = ',x_removed)
    print('len(x_removed) = ', len(x_removed))
    # print('n_removed = ', n_removed)
    
    x_flipped = flipOpeningBrackets(x_flipped)
    x_added = addClosingBracket(x_added)
    x_removed = removeOpeningBracket(x_removed)
    semiDyckAdded = []
    
    y_flipped = createSemiDyck1Labels(x_flipped)
    y_added = createSemiDyck1Labels(x_added)
    y_removed = createSemiDyck1Labels(x_removed)
    semiDyckRemoved = []
    
    x_new = []
    y_new = []
    lengths_new = []

    for i in range(len(x_flipped)):
        x_new.append(x_flipped[i])
        y_new.append(y_flipped[i])
        lengths_new.append(len(x_flipped[i]))
        semiDyckFlipped.append(isSemiDyck1(x_flipped[i], y_flipped[i]))

    
    for i in range(len(x_added)):
        x_new.append(x_added[i])
        y_new.append(y_added[i])
        lengths_new.append(len(x_added[i]))
        semiDyckAdded.append(isSemiDyck1(x_added[i], y_added[i]))
        
    for i in range(len(x_removed)):
        x_new.append(x_removed[i])
        y_new.append(y_removed[i])
        lengths_new.append(len(x_removed[i]))
        semiDyckRemoved.append(isSemiDyck1(x_removed[i], y_removed[i]))
    
    if False not in semiDyckFlipped and False not in semiDyckAdded and False not in semiDyckRemoved:
        print('ALL SEMI DYCK-1 SEQUENCES CORRECT')
    # writeDatasetToFile(x_new, y_new, lengths_new, filename)
    
    return x_new, y_new, lengths_new



def writeDatasetToFile(x,y,lengths,filename):
    with open(filename,'w') as f:
        f.write('')

    with open(filename,'a') as f:
        for i in range(len(x)):
            f.write(str(x[i]) + ',' + str(y[i]) + ',' + str(lengths[i]) + '\n')
        



def isSemiDyck1(seq, label):
    semiDyck=False
    count_open = 0
    count_close = 0
    # count_total = 0
    non_negative_count = 0
    # indices_of_excess_closing = []
    # print(seq)
    # print(seq.count('('))
    # print(seq.count(')'))
    output_label=''
    for i in range(len(seq)):

        if seq[i]=='(':
            count_open+=1
            non_negative_count+=1
            # output_label=output_label+'1'
        elif seq[i] ==')':
            count_close+=1
            non_negative_count-=1
            if non_negative_count<0:
                non_negative_count=0
        
        if non_negative_count>0:
            output_label=output_label+'1'
        else:
            output_label=output_label+'0'
                
        # print('&&&&&&&&&&&&&&&&')
        # print('seq[i] = ', seq[i])
        # print('count_open = ',count_open)
        # print('count_close = ',count_close)
        # print('count_total = ',count_total)
        # print('count_total_rectified = ',count_total_rectified)
    #     if count_total<0:
    #         indices_of_excess_closing.append(i)
    # 
    # print(indices_of_excess_closing)
    # if count_total<=0 and count_total_rectified==0 and count_close>=count_open:
    #     semiDyck=True
    
    if output_label==label:
        semiDyck=True

    return semiDyck


_ = makeSemiDyck1Dataset('train')
_ = makeSemiDyck1Dataset('val')
_ = makeSemiDyck1Dataset('short_test')
_ = makeSemiDyck1Dataset('test')
_ = makeSemiDyck1Dataset('long_test')
_ = makeSemiDyck1Dataset('zigzag')





# removeOpeningBracket([x[2]])
# addClosingBracket([x[2]])


# def flipOneOpeningBracket(indices, seqs, labels):
#     for i in range(len(indices)):
#         # seq = seqs[indices[i]]
#         seq = seqs[i]
#         count_zeros = 0
#         indices_zeros = []
#         label = labels[i]
#
#         for j, char in enumerate(label):
#             if char == '0':
#                 count_zeros += 1
#                 indices_zeros.append(j)
#
#         print('length of original seq = ', len(seq))
#         print('initial number of opening brackets = ', seq.count('('))
#         print('initial number of closing brackets = ', seq.count(')'))
#         print('count_zeros for seq ', seq, ' = ', count_zeros)
#         print('indices_zeros for seq ', seq, ' = ', indices_zeros)
#         rand_idx = randint(0, len(indices_zeros))
#         print('rand_idx for seq', seq, ' = ', rand_idx)
#         old_seq = seq
#         if rand_idx == len(indices_zeros):
#             changed_idx = 0
#         elif rand_idx < len(indices_zeros):
#             changed_idx = indices_zeros[rand_idx]
#         print('changed idx for seq ', seq, ' = ', changed_idx)
#
#         if changed_idx == 0:
#             seq = ')' + seq[1:]
#         elif changed_idx == len(seq) - 1:
#             seq = seq[1:] + ')'
#         elif changed_idx > 0:
#             seq = seq[:changed_idx] + '))' + seq[changed_idx + 2:]
#
#         print('changed seq = ', seq)
#         print('length of changed sequence = ', len(seq))
#         print('final number of opening brackets = ', seq.count('('))
#         print('final number of closing brackets = ', seq.count(')'))
#         print('invalid A')
#         print('***************************')
#         # seqs[indices[i]] = seq
#         seqs[i] = seq
#
#     return seqs

# def addExtraClosingBracket(indices, seqs, labels):
#     for i in range(len(indices)):
#         # seq = seqs[indices[i]]
#         seq = seqs[i]
#         count_zeros = 0
#         indices_zeros = []
#         label = labels[i]
#
#         for j, char in enumerate(label):
#             if char == '0':
#                 count_zeros += 1
#                 indices_zeros.append(j)
#
#         print('length of original seq = ', len(seq))
#         print('initial number of opening brackets = ', seq.count('('))
#         print('initial number of closing brackets = ', seq.count(')'))
#         print('count_zeros for seq ', seq, ' = ', count_zeros)
#         print('indices_zeros for seq ', seq, ' = ', indices_zeros)
#         # rand_idx = randint(0, len(seq))
#         rand_idx = randint(0, len(indices_zeros))
#         print('rand_idx for seq', seq, ' = ', rand_idx)
#         old_seq = seq
#         if rand_idx == len(indices_zeros):
#             changed_idx = 0
#         elif rand_idx < len(indices_zeros):
#             changed_idx = indices_zeros[rand_idx]
#         print('changed idx for seq ', seq, ' = ', changed_idx)
#
#         if changed_idx == 0:
#             seq = ')' + seq[0:]
#         elif changed_idx == len(seq) - 1:
#             seq = seq[0:] + ')'
#         elif changed_idx > 0:
#             seq = seq[:changed_idx] + '))' + seq[changed_idx + 1:]
#
#         print('changed seq = ', seq)
#         print('length of changed sequence = ', len(seq))
#         print('final number of opening brackets = ', seq.count('('))
#         print('final number of closing brackets = ', seq.count(')'))
#         print('Semi Dyck-1 (additional closing bracket)')
#         print('***************************')
#         # seqs[indices[i]] = seq
#         seqs[i] = seq
#     return seqs


# def removeOpeningBracket(indices, seqs, labels):
#     for i in range(len(indices)):
#         # seq = seqs[indices[i]]
#         seq = seqs[i]
#         count_zeros = 0
#         indices_zeros = []
#         label = labels[i]
#
#         for j, char in enumerate(label):
#             if char == '0':
#                 count_zeros += 1
#                 indices_zeros.append(j)
#
#         print('length of original seq = ', len(seq))
#         print('initial number of opening brackets = ', seq.count('('))
#         print('initial number of closing brackets = ', seq.count(')'))
#         print('count_zeros for seq ', seq, ' = ', count_zeros)
#         print('indices_zeros for seq ', seq, ' = ', indices_zeros)
#         rand_idx = randint(0, len(indices_zeros))
#         print('rand_idx for seq', seq, ' = ', rand_idx)
#         old_seq = seq
#         if rand_idx == len(indices_zeros):
#             changed_idx = 0
#         elif rand_idx < len(indices_zeros):
#             changed_idx = indices_zeros[rand_idx]
#         print('changed idx for seq ', seq, ' = ', changed_idx)
#
#         if changed_idx == 0:
#             seq = seq[1:]
#         elif changed_idx == len(seq) - 1:
#             seq = seq[1:]
#         elif changed_idx > 0:
#             seq = seq[:changed_idx+1] + '' + seq[changed_idx + 2:]
#
#         print('changed seq = ', seq)
#         print('length of changed sequence = ', len(seq))
#         print('final number of opening brackets = ', seq.count('('))
#         print('final number of closing brackets = ', seq.count(')'))
#         print('invalid A')
#         print('***************************')
#         # seqs[indices[i]] = seq
#         seqs[i] = seq
#
#     return seqs



# def isSemiDyck1(seq):
#     semiDyck=False
#     count_open = 0
#     count_close = 0
#     count_total = 0
#     count_total_rectified = 0
#     indices_of_excess_closing = []
#     print(seq)
#     print(seq.count('('))
#     print(seq.count(')'))
#     for i in range(len(seq)):
#
#         if seq[i]=='(':
#             count_open+=1
#             count_total+=1
#             count_total_rectified+=1
#         elif seq[i] ==')':
#             count_close+=1
#             count_total-=1
#             count_total_rectified-=1
#             if count_total<0:
#                 count_total_rectified=0
#         print('&&&&&&&&&&&&&&&&')
#         print('seq[i] = ', seq[i])
#         print('count_open = ',count_open)
#         print('count_close = ',count_close)
#         print('count_total = ',count_total)
#         print('count_total_rectified = ',count_total_rectified)
#         if count_total<0:
#             indices_of_excess_closing.append(i)
#
#     print(indices_of_excess_closing)
#     if count_total<=0 and count_total_rectified==0 and count_close>=count_open:
#         semiDyck=True
#
#     return semiDyck


#
# print(x[0])
# indices = [0,1,2,3]
# seqs = x[0:4]
# labels = y[0:4]
#
# indices2 = [0,1,2,3]
# seqs2=x[0:4]
# labels2=y[0:4]
#
# indices3 = [0,1,2,3]
# seqs3=x[0:4]
# labels3=y[0:4]
#
# indices4 = [0,1,2,3]
# seqs4=x[0:4]
# labels4=y[0:4]
#
# print(flipOneOpeningBracket(indices,seqs,labels))
# print(isSemiDyck1(seqs[0]))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print(addExtraClosingBracket(indices2, seqs2, labels2))
# print(isSemiDyck1(seqs2[0]))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print(removeOpeningBracket(indices3,seqs3,labels3))
# print(isSemiDyck1(seqs3[0]))
# print('||||||||||||||||||||||||||||||||||||||||||')
# indices4 = [0,1,2,3]
# seqs4=x[0:4]
# labels4=y[0:4]
# print(flipOpeningBrackets(indices4, seqs4[0:4], labels4[0:4]))
#
#
# indices4 = [0,1,2,3]
# seqs4=x[0:4]
# labels4=y[0:4]
# seqs_new = removeOpeningBracket(indices4,seqs4,labels4)
# print(seqs_new)
# print(isSemiDyck1(seqs_new[0]))
# print('££££££££££££££££££')
# print(isSemiDyck1(seqs_new[1]))
# print('@@@@@@@@@@@@@@@@@@@@@@@')
# print(isSemiDyck1(seqs_new[2]))


