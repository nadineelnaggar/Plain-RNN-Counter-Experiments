import sklearn
import pandas as pd
from sklearn.utils import shuffle
from random import randint
import math


# catalan(n) is the number of valid Dyck-1 sequences of length 2n tokens (n pairs)
def catalan(n):
    return int(math.factorial(2*n)/(math.factorial(n+1)*math.factorial(n)))

print(catalan(1))
print(catalan(2))
print(catalan(3))
print(catalan(4))
print(catalan(5))

"""
- create new non-brute force code to generate Dyck sequences then alter to make them invalid or incomplete
- look at def num_valid_sequences_lenghts(n,m, dataset_size): function and then try to generate a dataset which
    has a somewhat balanced number of varying length Dyck sequences then make them invalid or incomplete
- generate these datasets for suzgun task, ternary classification and binary classification and run the codes




* for binary classification, use 1/4 of dataset for invalid and 1/4 for incomplete to be part of the invalid class
* for ternary classification 1/3 invalid, 1/3 incomplete
* for suzgun task, all valid, no changes
* make sure to generate datasets of size 12k with long set of size 6k or 15k with long set size 7.5k
* make sure to create 2 instances of each dataset (while running at the same time now),
    * one for feedback at the very end
    * one for feedback every time step

"""

def make_invalid(indices, seqs):
    """
    input a valid sequence and distort it to make it invalid by either
        - swapping the order of opening and closing brackets (equal number but wrong order)
        - replacing opening brackets with closing brackets so that there is an excess of closing brackets
    :return:
    """
    incorrect_order = []
    excess_close = []

    for i in range(len(indices)):

        if i%2 !=0:
            excess_close.append(indices[i])
        elif i%2==0 or i==0:
            incorrect_order.append(indices[i])

    for i in range(len(incorrect_order)):
        seq = seqs[incorrect_order[i]]


        # print('***********************')
        # print('index in original array = ', incorrect_order[i])
        # print(seq)
        # print(len(seq))
        # print(seq[0])
        # print(seq.find('('))
        seq = seq[-1:]+seq[:-1]
        # print(seq)
        seqs[incorrect_order[i]]=seq

    for i in range(len(excess_close)):
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('index in original array = ', excess_close[i])
        seq = seqs[excess_close[i]]
        # print(seq)
        # print(len(seq))
        # print(seq[0])
        ind = seq.find('()')
        # print(ind)
        # # seq[ind]= ')'
        # # seq[ind+1]='('
        # print(ind+1)
        seq = seq[:ind]+')'+seq[ind+1:]
        # print(seq)
        seqs[excess_close[i]] = seq
    return seqs





def make_incomplete(indices, seqs):
    """
    input a valid sequence and distort it to make it potentially valid by
        - replacing one or more closing brackets in a random location with an opening bracket
    :return:
    """

    for i in range(len(indices)):
        seq = seqs[indices[i]]


        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # print('index in original array = ', indices[i])
        # print(seq)
        # print(len(seq))
        # print(seq[0])
        # # print(seq.find(')'))

        ind = seq.find(')')
        # print(ind)
        # # seq[ind]= ')'
        # # seq[ind+1]='('
        # print(ind + 1)
        seq = seq[:ind] + '(' + seq[ind + 1:]
        # print(seq)
        # # seqs[excess_close[i]] = seq
        #
        # # seq = seq[-1:]+seq[:-1]
        # # print(seq)
        seqs[indices[i]]=seq
    return seqs



class Dyck1_Generator():
    def generateParenthesis(self,n_pairs, dataset_size=-1):
        stack = []
        sequences = []

        def generate(n_open, n_close):
            if len(sequences)<dataset_size and dataset_size!=-1:
                if n_open==n_close==n_pairs:
                    sequences.append("".join(stack))
                    return
                if n_open<n_pairs:
                    stack.append('(')
                    generate(n_open+1,n_close)
                    stack.pop()
                if n_close < n_open:
                    stack.append(')')
                    generate(n_open,n_close+1)
                    stack.pop()
            elif dataset_size==-1: #no restriction on dataset size, generate all
                if n_open==n_close==n_pairs:
                    sequences.append("".join(stack))
                    return
                if n_open<n_pairs:
                    stack.append('(')
                    generate(n_open+1,n_close)
                    stack.pop()
                if n_close < n_open:
                    stack.append(')')
                    generate(n_open,n_close+1)
                    stack.pop()
        generate(0,0)
        return sequences


# class Dyck1_Generator_Suzgun(object):
#     def generateParenthesis(self, n):
#
#         def generate(A = []):
#             if len(A) == 2*n:
#                 if valid(A):
#                     val.append("".join(A))
#                 elif potentially_valid(A):
#                     potentially_val.append("".join(A))
#                 else:
#                     inval.append("".join(A))
#             else:
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()
#                 # brac = randint(1, 2)
#                 # if brac==1:
#                 #     A.append('(')
#                 #     generate(A)
#                 #     A.pop()
#                 # elif brac==2:
#                 #     A.append(')')
#                 #     generate(A)
#                 #     A.pop()
#
#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0
#
#         def potentially_valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(':
#                     bal += 1
#                 else:
#                     bal -= 1
#                 if bal < 0: return False
#             return bal > 0
#         def invalid(A):
#             bal = 0
#             for c in A:
#                 if c == '(':
#                     bal += 1
#                 else:
#                     bal -= 1
#                 # if bal < 0: return False
#             return bal < 0
#
#         val = []
#         potentially_val=[]
#         inval = []
#         generate()
#         return val, potentially_val, inval
#         # return val, potentially_val

# valid = []
# incomplete=[]
# invalid=[]
# gen = Dyck1_Generator()
# # valid, incomplete, invalid = gen.generateParenthesis(1)
# valid = gen.generateParenthesis(1)
# print(valid)
# print(incomplete)
# print(invalid)
# print(len(valid))
# print(len(valid)+len(incomplete)+len(invalid))
# # valid, incomplete, invalid = gen.generateParenthesis(2)
# valid = gen.generateParenthesis(2)
# print(valid)
# print(incomplete)
# print(invalid)
# print(len(valid))
# print(len(valid)+len(incomplete)+len(invalid))
# # valid, incomplete, invalid = gen.generateParenthesis(3)
# valid = gen.generateParenthesis(3, dataset_size=4)
# print(valid)
# print(incomplete)
# print(invalid)
# print(len(valid))
# print(len(valid)+len(incomplete)+len(invalid))
#
# # valid, incomplete, invalid = gen.generateParenthesis(4)
# valid = gen.generateParenthesis(4, dataset_size=5)
# print(valid)
# print(incomplete)
# print(invalid)
# print(len(valid))
# print(len(valid)+len(incomplete)+len(invalid))
#
# print(catalan(6))
# print(catalan(25))
# print('/////////////////////////////')

def num_valid_sequences_lenghts(n,m, dataset_size):
    available_elems = dataset_size
    num_elems_per_i = available_elems//(m+1-n)
    print('dataset size = initial available elems = ',available_elems)
    for i in range(n, m+1):
        print('##########################################')
        n_elems = min(catalan(i), num_elems_per_i)
        available_elems-=n_elems
        print('i =',i)
        print('catalan number = ',catalan(i))
        print('expected n_elems per i = ',num_elems_per_i)
        if i<m:
            num_elems_per_i = available_elems//(m-i)
        elif i ==m:
            num_elems_per_i = available_elems

        print('actual n_elems for i = ',n_elems)
        print('available elems = ',available_elems)
        print('new num_elems per i = ',num_elems_per_i)


# num_valid_sequences_lenghts(1,25,10000) #5 pairs to 10 pairs
#
# print('////////////////////////////////////////////////')
# num_valid_sequences_lenghts(26, 50, 5000)



def generateDyckSequencesRange(n, m, dataset_size):
    available_elems = dataset_size
    num_elems_per_i = available_elems // (m + 1 - n)
    # print('dataset size = initial available elems = ', available_elems)
    sequences = []
    gen = Dyck1_Generator()
    for i in range(n, m + 1):
        # print('##########################################')
        n_elems = min(catalan(i), num_elems_per_i)
        available_elems -= n_elems
        # print('i =', i)
        # print('catalan number = ', catalan(i))
        # print('expected n_elems per i = ', num_elems_per_i)
        elems = gen.generateParenthesis(i, n_elems)
        # sequences.append(str(elem) for elem in elems)
        for elem in elems:
            sequences.append(elem)
        print(elems)
        # sequences.append(gen.generateParenthesis(i, n_elems))
        if i < m:

            num_elems_per_i = available_elems // (m - i)

        elif i == m:
            num_elems_per_i = available_elems


        # print('actual n_elems for i = ', n_elems)
        # print('available elems = ', available_elems)
        # print('new num_elems per i = ', num_elems_per_i)

    return sequences

# # print(generateDyckSequencesRange(1,5,30))
#
#
# seqs = generateDyckSequencesRange(1,5,30)
# print(seqs)
#
#
# # invalid_seqs = []
# # incomplete_seqs = []
#
# invalid_indices = []
# incomplete_indices = []
# labels = []
# for i in range(len(seqs)):
#     labels.append('valid')
#
# for i in range(100000):
#     value = randint(1, len(seqs)-1)
#     if value not in invalid_indices and value not in incomplete_indices and len(invalid_indices)<len(seqs)/3:
#         invalid_indices.append(value)
#     value2 = randint(1,len(seqs)-1)
#     if value2 not in invalid_indices and value2 not in incomplete_indices and len(incomplete_indices)<len(seqs)/3:
#         incomplete_indices.append(value2)
#
#     if len(incomplete_indices)==len(invalid_indices)==len(seqs)/3:
#         break
#
# print(invalid_indices)
# print(incomplete_indices)
# print(len(invalid_indices))
# print(len(incomplete_indices))
#
# make_invalid(invalid_indices, seqs)
# make_incomplete(incomplete_indices, seqs)
#
# for i in range(len(incomplete_indices)):
#     labels[incomplete_indices[i]]='incomplete'
#
# for i in range(len(invalid_indices)):
#     labels[invalid_indices[i]]='invalid'
#
# print(labels)
#
#
#
#
#
# print(seqs)


def generateTernaryDataset(start_n, end_n, dataset_size):
    seqs = generateDyckSequencesRange(start_n, end_n, dataset_size)
    invalid_indices = []
    incomplete_indices = []
    labels = []
    for i in range(len(seqs)):
        labels.append('valid')

    for i in range(100000):
        value = randint(1, len(seqs) - 1)
        if value not in invalid_indices and value not in incomplete_indices and len(invalid_indices) < len(seqs) / 3:
            invalid_indices.append(value)
        value2 = randint(1, len(seqs) - 1)
        if value2 not in invalid_indices and value2 not in incomplete_indices and len(incomplete_indices) < len(
                seqs) / 3:
            incomplete_indices.append(value2)

        if len(incomplete_indices) == len(invalid_indices) == len(seqs) / 3:
            break

    # print(invalid_indices)
    # print(incomplete_indices)
    # print(len(invalid_indices))
    # print(len(incomplete_indices))

    make_invalid(invalid_indices, seqs)
    make_incomplete(incomplete_indices, seqs)

    for i in range(len(incomplete_indices)):
        labels[incomplete_indices[i]] = 'incomplete'

    for i in range(len(invalid_indices)):
        labels[invalid_indices[i]] = 'invalid'
    document_name = 'Dyck1_TernaryDataset_'+str(start_n)+'to'+str(end_n)+'pairs_'+str(dataset_size)+'elements_balanced.txt'
    with open(document_name,'a') as f:
        for i in range(len(seqs)):
            f.write(seqs[i] + ',' + labels[i] + '\n')
    print('',document_name,' completed')


def generateBinaryDataset(start_n, end_n, dataset_size):
    seqs = generateDyckSequencesRange(start_n, end_n, dataset_size)
    invalid_indices = []
    incomplete_indices = []
    labels = []
    for i in range(len(seqs)):
        labels.append('valid')

    # make 1/3 of invalid sequences have excess opening, 1/3 have excess closing, and 1/3 have incorrect order
    for i in range(100000):
        value = randint(1, len(seqs) - 1)
        if value not in invalid_indices and value not in incomplete_indices and len(invalid_indices) < len(seqs) / 3:
            invalid_indices.append(value)
        value2 = randint(1, len(seqs) - 1)
        if value2 not in invalid_indices and value2 not in incomplete_indices and len(incomplete_indices) < len(
                seqs) / 6:
            incomplete_indices.append(value2)

        if len(incomplete_indices) == len(invalid_indices) == len(seqs) / 3:
            break

    # print(invalid_indices)
    # print(incomplete_indices)
    # print(len(invalid_indices))
    # print(len(incomplete_indices))

    make_invalid(invalid_indices, seqs)
    make_incomplete(incomplete_indices, seqs)

    for i in range(len(incomplete_indices)):
        labels[incomplete_indices[i]] = 'invalid'

    for i in range(len(invalid_indices)):
        labels[invalid_indices[i]] = 'invalid'
    document_name = 'Dyck1_BinaryDataset_'+str(start_n)+'to'+str(end_n)+'pairs_'+str(dataset_size)+'elements_balanced.txt'
    with open(document_name,'a') as f:
        for i in range(len(seqs)):
            f.write(seqs[i] + ',' + labels[i] + '\n')
    print('',document_name,' completed')


def generatePredictionDataset(start_n, end_n, dataset_size):
    seqs = generateDyckSequencesRange(start_n, end_n, dataset_size)
    invalid_indices = []
    incomplete_indices = []
    labels = []
    for i in range(len(seqs)):
        labels.append('valid')

    # make 1/3 of invalid sequences have excess opening, 1/3 have excess closing, and 1/3 have incorrect order
    for i in range(100000):
        # value = randint(1, len(seqs) - 1)
        # if value not in invalid_indices and value not in incomplete_indices and len(invalid_indices) < len(seqs) / 3:
        #     invalid_indices.append(value)
        value2 = randint(1, len(seqs) - 1)
        if value2 not in incomplete_indices and len(incomplete_indices) < len(
                seqs) / 2:
            incomplete_indices.append(value2)

        if len(incomplete_indices) == len(invalid_indices) == len(seqs) / 3:
            break

    # print(invalid_indices)
    # print(incomplete_indices)
    # print(len(invalid_indices))
    # print(len(incomplete_indices))

    make_invalid(invalid_indices, seqs)
    make_incomplete(incomplete_indices, seqs)

    for i in range(len(incomplete_indices)):
        labels[incomplete_indices[i]] = 'incomplete'

    # for i in range(len(invalid_indices)):
    #     labels[invalid_indices[i]] = 'invalid'
    document_name = 'Dyck1_PredictionDataset_'+str(start_n)+'to'+str(end_n)+'pairs_'+str(dataset_size)+'elements_balanced.txt'
    with open(document_name,'a') as f:
        for i in range(len(seqs)):
            f.write(seqs[i] + ',' + labels[i] + '\n')
    print('',document_name,' completed')

generateTernaryDataset(1,25,12000)
generateTernaryDataset(26,50,6000)
generateBinaryDataset(1,25,12000)
generateBinaryDataset(26,50,6000)
generatePredictionDataset(1,25,12000)
generatePredictionDataset(26,50,6000)


# def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
#     gen = Dyck1_Generator_Suzgun()
#     # d1_valid, d1_invalid = gen.generateParenthesis(3)
#     d1_valid = []
#     d1_invalid = []
#     for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
#         x,y = gen.generateParenthesis(i)
#         for elem in x:
#             d1_valid.append(elem)
#         for elem in y:
#             d1_invalid.append(elem)
#     return d1_valid,d1_invalid
#
# valid, incomplete = generateDataset(1,3)
# print(valid)
# print(incomplete)

# valid, incomplete = generateDataset(1,4)
# print(valid)
# print(incomplete)

#
# class Dyck1_Generator_Binary(object):
#     def generateParenthesis(self, n):
#         def generate(A = []):
#             if len(A) == 2*n:
#                 if valid(A):
#                     val.append("".join(A))
#                 elif potentially_valid(A):
#                     potentially_val.append("".join(A))
#                 else:
#                     inval.append("".join(A))
#             else:
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()
#
#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0
#
#         def potentially_valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(':
#                     bal += 1
#                 else:
#                     bal -= 1
#                 if bal < 0: return False
#             return bal > 0
#
#         val = []
#         potentially_val=[]
#         inval = []
#         generate()
#         return val, potentially_val, inval
#
# class Dyck1_Generator_Ternary(object):
#     def generateParenthesis(self, n):
#         def generate(A = []):
#             if len(A) == 2*n:
#                 if valid(A):
#                     val.append("".join(A))
#                 elif potentially_valid(A):
#                     potentially_val.append("".join(A))
#                 else:
#                     inval.append("".join(A))
#             else:
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()
#
#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0
#
#         def potentially_valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(':
#                     bal += 1
#                 else:
#                     bal -= 1
#                 if bal < 0: return False
#             return bal > 0
#
#         val = []
#         potentially_val=[]
#         inval = []
#         generate()
#         return val, potentially_val, inval
#
# def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
#     gen = Dyck1_Generator()
#     # d1_valid, d1_invalid = gen.generateParenthesis(3)
#     d1_valid = []
#     d1_potentially_valid=[]
#     d1_invalid = []
#     for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
#         x,y,z = gen.generateParenthesis(i)
#         for elem in x:
#             d1_valid.append(elem)
#         for elem in y:
#             d1_potentially_valid.append(elem)
#         for elem in z:
#             d1_invalid.append(elem)
#     return d1_valid,d1_potentially_valid,d1_invalid
#
# # d1_valid, d1_invalid = generateDataset(6)
#
#
# # print(d1_valid)
# # print('///////////////////////')
# # print(d1_invalid)
#
# # add the labels and then create the csv file to complete the dataset
#
# def generateLabelledDataset(n_bracket_pairs_start,n_bracket_pairs_end):
#     d1_valid,d1_potentially_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
#     dataset = []
#     sentences = []
#     labels = []
#     for elem in d1_valid:
#         entry = (elem, 'valid')
#         dataset.append(entry)
#         sentences.append(elem)
#         labels.append('valid')
#     for elem in d1_invalid:
#         entry = (elem,'invalid')
#         dataset.append(entry)
#         sentences.append(elem)
#         labels.append('invalid')
#     dataset, sentences, labels = shuffle(dataset, sentences, labels,random_state=0)
#     return dataset, sentences, labels
#
#
#
# def generateBalancedLabelledDataset(n_bracket_pairs_start, n_bracket_pairs_end, size=15000, size_limit=False):
#     d1_valid,d1_potentially_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
#     # size=size
#     # if min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))<size/3:
#     #     size=3*min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))
#     # elif size_limit==False or min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))>size/3:
#     #     size = 3 * min(len(d1_valid), len(d1_potentially_valid), len(d1_invalid))
#     size = 3*max(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))
#     d1_valid = shuffle(d1_valid)
#     d1_potentially_valid=shuffle(d1_potentially_valid)
#     d1_invalid = shuffle(d1_invalid)
#     dataset = []
#     sentences = []
#     labels = []
#     count_valid=0
#     count_potentially_valid=0
#     count_invalid=0
#     for elem in d1_valid:
#         entry = (elem, 'valid')
#         # if elem not in sentences and size_limit==True and count_valid<(size/3):
#         if elem not in sentences and count_valid < (size / 3):
#             dataset.append(entry)
#             sentences.append(elem)
#             labels.append('valid')
#             count_valid+=1
#     for elem in d1_potentially_valid:
#         entry = (elem, 'incomplete')
#         # if elem not in sentences and size_limit==True and count_potentially_valid<(size/3):
#         if elem not in sentences and count_potentially_valid < (size / 3):
#             dataset.append(entry)
#             sentences.append(elem)
#             labels.append('incomplete')
#             count_potentially_valid+=1
#     for elem in d1_invalid:
#         # if elem not in sentences and size_limit==True and count_valid < (size / 3):
#         if elem not in sentences and count_valid < (size / 3):
#             entry = (elem,'invalid')
#             dataset.append(entry)
#             sentences.append(elem)
#             labels.append('invalid')
#             count_invalid+=1
#     print('count valid = ',count_valid)
#     print('count potentially valid = ',count_potentially_valid)
#     print('count invalid = ',count_invalid)
#     print('dataset length = ',len(dataset))
#
#     if count_valid!=count_invalid or count_valid!=count_potentially_valid or count_potentially_valid!=count_invalid:
#         max_elements = max(count_invalid,count_valid,count_potentially_valid)
#         if count_valid<max_elements:
#             for i in range((max_elements-count_valid)):
#                 idx = randint(0,len(d1_valid)-1)
#                 elem = d1_valid[idx]
#                 entry=(elem,'valid')
#                 dataset.append(entry)
#                 sentences.append(elem)
#                 labels.append('valid')
#                 count_valid+=1
#         if count_potentially_valid<max_elements:
#             for i in range(max_elements-count_potentially_valid):
#                 idx = randint(0,len(d1_potentially_valid)-1)
#                 elem = d1_potentially_valid[idx]
#                 entry=(elem,'incomplete')
#                 dataset.append(entry)
#                 sentences.append(elem)
#                 labels.append('incomplete')
#                 count_potentially_valid+=1
#         if count_invalid<max_elements:
#             for i in range(max_elements-count_invalid):
#                 idx = randint(0,len(d1_invalid)-1)
#                 elem = d1_invalid[idx]
#                 entry=(elem,'invalid')
#                 dataset.append(entry)
#                 sentences.append(elem)
#                 labels.append('invalid')
#                 count_invalid+=1
#     print('count valid = ',count_valid)
#     print('count potentially valid = ',count_potentially_valid)
#     print('count invalid = ',count_invalid)
#     print('dataset length = ',len(dataset))
#
#
#     dataset, sentences, labels = shuffle(dataset, sentences, labels,random_state=0)
#
#     return dataset, sentences, labels
#
#
#
#
# # dataset, sentences, labels = generateBalancedLabelledDataset(1, 5, size=12000, size_limit=True)
# # for elem in dataset:
# #     print(elem)
# # print(len(dataset))
# #
# # with open('Dyck1_Ternary_Dataset_1to5pairs_balanced.txt','a') as f:
# #     for i in range(len(sentences)):
# #         f.write(sentences[i]+','+labels[i]+'\n')
#
# # dataset_length, sentences_length, labels_length = generateBalancedLabelledDataset(6, 10, size=5000, size_limit=True)
# # with open('Dyck1_Ternary_Dataset_6to10pairs_balanced_length.txt','a') as f:
# #     for i in range(len(sentences_length)):
# #         f.write(sentences_length[i]+','+labels_length[i]+'\n')