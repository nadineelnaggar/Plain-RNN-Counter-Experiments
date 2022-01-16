import pandas as pd
import sklearn
from sklearn.utils import shuffle

class Dyck1_Generator(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    val.append("".join(A))
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

        val = []
        inval = []
        generate()
        return val, inval

def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
    gen = Dyck1_Generator()
    # d1_valid, d1_invalid = gen.generateParenthesis(3)
    d1_valid = []
    d1_invalid = []
    for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
        x,y = gen.generateParenthesis(i)
        for elem in x:
            d1_valid.append(elem)
        for elem in y:
            d1_invalid.append(elem)
    return d1_valid,d1_invalid

# d1_valid, d1_invalid = generateDataset(6)


# print(d1_valid)
# print('///////////////////////')
# print(d1_invalid)

# add the labels and then create the csv file to complete the dataset

def generateLabelledDataset(n_bracket_pairs_start,n_bracket_pairs_end):
    d1_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
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
    sentences, labels = shuffle(dataset, sentences, labels,random_state=0)
    return dataset, sentences, labels


#this function generates a balanced labelled dataset consisting of bracket sequences of lengths ranging from
# n_bracket_pairs_start to n_bracket_pairs_end. if a size limit is defined then
def generateBalancedLabelledDistinctDataset(n_bracket_pairs_start,n_bracket_pairs_end, size=15000, size_limit=False):
    d1_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    d1_valid = shuffle(d1_valid)
    d1_invalid = shuffle(d1_invalid)
    dataset = []
    sentences = []
    labels = []
    count_valid=0
    count_invalid=0
    for elem in d1_valid:
        entry = (elem, 'valid')
        if elem not in sentences and size_limit==True and count_valid<(size/2):
            dataset.append(entry)
            sentences.append(elem)
            labels.append('valid')
            count_valid+=1
    for elem in d1_invalid:
        if elem not in sentences and size_limit==True and count_valid < (size / 2):
            entry = (elem,'invalid')
            dataset.append(entry)
            sentences.append(elem)
            labels.append('invalid')
            count_invalid+=1
    sentences, labels = shuffle(dataset, sentences, labels,random_state=0)
    # if size_limit==True:
    #     sentences1 = sentences
    #     labels1 = labels
    #     dataset1=dataset
    #     sentences=[]
    #     labels=[]
    #     dataset=[]
    #     for i in range(len(sentences1)):
    #         if sentences1[i] not in sentences and labels1[i]=='valid' and count_valid<(size/2):
    #             sentences.append(sentences1[i])
    #             labels.append(labels1[i])
    #             dataset.append(dataset1[i])
    #         elif sentences1[i] not in sentences and labels1[i]=='invalid' and count_invalid<(size/2):
    #             sentences.append(sentences1[i])
    #             labels.append(labels1[i])
    #             dataset.append(dataset1[i])
    return dataset, sentences, labels



#generate dataset similar to Suzgun paper
dataset, sentences, labels = generateBalancedLabelledDistinctDataset(1,25,size=15000,size_limit=True)


with open('Dyck1_Dataset_25pairs_balanced.txt','a') as f:
    for i in range(len(sentences)):
        f.write(sentences[i]+','+labels[i]+'\n')

dataset_length, sentences_length, labels_length = generateBalancedLabelledDistinctDataset(25,50,size=5000,size_limit=True)
with open('Dyck1_Dataset_25pairs_balanced_length.txt','a') as f:
    for i in range(len(sentences_length)):
        f.write(sentences_length[i]+','+labels_length[i]+'\n')


# dataset = generateLabelledDataset(6)
# print(dataset)
#
# dataset_ = pd.DataFrame(dataset,columns=['sentence','label'])
# print(dataset_.head())
#
# dataset_ = sklearn.utils.shuffle(dataset_).reset_index(drop=True)
# print(dataset_.head())
# print(len(dataset_))
# # dataset_=sklearn.utils.shuffle(dataset_)
# # print(dataset_.head())
#
# dataset3=generateLabelledDataset(3)
# print(dataset3)
# print(len(dataset3))
#
# # dataset_.to_csv('Dyck1_Dataset_6pairs.csv',index=False)