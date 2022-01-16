import sklearn
import pandas as pd
from sklearn.utils import shuffle
from random import randint

class Dyck1_Generator(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    val.append("".join(A))
                elif potentially_valid(A):
                    potentially_val.append("".join(A))
                else:
                    inval.append("".join(A))
            else:
                A.append('(')
                generate(A)
                A.pop()
                A.append(')')
                generate(A)
                A.pop()

        def valid(A):
            bal = 0
            for c in A:
                if c == '(': bal += 1
                else: bal -= 1
                if bal < 0: return False
            return bal == 0

        def potentially_valid(A):
            bal = 0
            for c in A:
                if c == '(':
                    bal += 1
                else:
                    bal -= 1
                if bal < 0: return False
            return bal > 0

        val = []
        potentially_val=[]
        inval = []
        generate()
        return val, potentially_val, inval

def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
    gen = Dyck1_Generator()
    # d1_valid, d1_invalid = gen.generateParenthesis(3)
    d1_valid = []
    d1_potentially_valid=[]
    d1_invalid = []
    for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
        x,y,z = gen.generateParenthesis(i)
        for elem in x:
            d1_valid.append(elem)
        for elem in y:
            d1_potentially_valid.append(elem)
        for elem in z:
            d1_invalid.append(elem)
    return d1_valid,d1_potentially_valid,d1_invalid

# d1_valid, d1_invalid = generateDataset(6)


# print(d1_valid)
# print('///////////////////////')
# print(d1_invalid)

# add the labels and then create the csv file to complete the dataset

def generateLabelledDataset(n_bracket_pairs_start,n_bracket_pairs_end):
    d1_valid,d1_potentially_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    dataset = []
    sentences = []
    labels = []
    for elem in d1_valid:
        entry = (elem, 'valid')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('valid')
    for elem in d1_invalid:
        entry = (elem,'invalid')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('invalid')
    dataset, sentences, labels = shuffle(dataset, sentences, labels,random_state=0)
    return dataset, sentences, labels



def generateBalancedLabelledDataset(n_bracket_pairs_start, n_bracket_pairs_end, size=15000, size_limit=False):
    d1_valid,d1_potentially_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    # size=size
    # if min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))<size/3:
    #     size=3*min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))
    # elif size_limit==False or min(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))>size/3:
    #     size = 3 * min(len(d1_valid), len(d1_potentially_valid), len(d1_invalid))
    size = 3*max(len(d1_valid),len(d1_potentially_valid),len(d1_invalid))
    d1_valid = shuffle(d1_valid)
    d1_potentially_valid=shuffle(d1_potentially_valid)
    d1_invalid = shuffle(d1_invalid)
    dataset = []
    sentences = []
    labels = []
    count_valid=0
    count_potentially_valid=0
    count_invalid=0
    for elem in d1_valid:
        entry = (elem, 'valid')
        # if elem not in sentences and size_limit==True and count_valid<(size/3):
        if elem not in sentences and count_valid < (size / 3):
            dataset.append(entry)
            sentences.append(elem)
            labels.append('valid')
            count_valid+=1
    for elem in d1_potentially_valid:
        entry = (elem, 'incomplete')
        # if elem not in sentences and size_limit==True and count_potentially_valid<(size/3):
        if elem not in sentences and count_potentially_valid < (size / 3):
            dataset.append(entry)
            sentences.append(elem)
            labels.append('incomplete')
            count_potentially_valid+=1
    for elem in d1_invalid:
        # if elem not in sentences and size_limit==True and count_valid < (size / 3):
        if elem not in sentences and count_valid < (size / 3):
            entry = (elem,'invalid')
            dataset.append(entry)
            sentences.append(elem)
            labels.append('invalid')
            count_invalid+=1
    print('count valid = ',count_valid)
    print('count potentially valid = ',count_potentially_valid)
    print('count invalid = ',count_invalid)
    print('dataset length = ',len(dataset))

    if count_valid!=count_invalid or count_valid!=count_potentially_valid or count_potentially_valid!=count_invalid:
        max_elements = max(count_invalid,count_valid,count_potentially_valid)
        if count_valid<max_elements:
            for i in range((max_elements-count_valid)):
                idx = randint(0,len(d1_valid)-1)
                elem = d1_valid[idx]
                entry=(elem,'valid')
                dataset.append(entry)
                sentences.append(elem)
                labels.append('valid')
                count_valid+=1
        if count_potentially_valid<max_elements:
            for i in range(max_elements-count_potentially_valid):
                idx = randint(0,len(d1_potentially_valid)-1)
                elem = d1_potentially_valid[idx]
                entry=(elem,'incomplete')
                dataset.append(entry)
                sentences.append(elem)
                labels.append('incomplete')
                count_potentially_valid+=1
        if count_invalid<max_elements:
            for i in range(max_elements-count_invalid):
                idx = randint(0,len(d1_invalid)-1)
                elem = d1_invalid[idx]
                entry=(elem,'invalid')
                dataset.append(entry)
                sentences.append(elem)
                labels.append('invalid')
                count_invalid+=1
    print('count valid = ',count_valid)
    print('count potentially valid = ',count_potentially_valid)
    print('count invalid = ',count_invalid)
    print('dataset length = ',len(dataset))


    dataset, sentences, labels = shuffle(dataset, sentences, labels,random_state=0)

    return dataset, sentences, labels




# dataset, sentences, labels = generateBalancedLabelledDataset(1, 5, size=12000, size_limit=True)
# for elem in dataset:
#     print(elem)
# print(len(dataset))
#
# with open('Dyck1_Ternary_Dataset_1to5pairs_balanced.txt','a') as f:
#     for i in range(len(sentences)):
#         f.write(sentences[i]+','+labels[i]+'\n')

# dataset_length, sentences_length, labels_length = generateBalancedLabelledDataset(6, 10, size=5000, size_limit=True)
# with open('Dyck1_Ternary_Dataset_6to10pairs_balanced_length.txt','a') as f:
#     for i in range(len(sentences_length)):
#         f.write(sentences_length[i]+','+labels_length[i]+'\n')