import random
from random import randint

def read_dataset():
    x = []
    y = []
    lengths = []

    with open('Dyck1_Dataset_Suzgun_502to1000tokens.txt', 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            if len(sentence)==1000:
                x.append(sentence)
                y.append(label)
                lengths.append(len(sentence))
    return x, y, lengths




def nest_sequences():
    x, y, lengths = read_dataset()

    x_new = []
    y_new = []
    lengths_new = []

    for i in range(len(x)):

        for j in range(len(x)):
            nest_position = randint(0, 999)
            temp = ''
            temp = temp+x[i][:nest_position]+x[j]+x[i][nest_position:]
            # print(len(temp))
            if temp not in x_new:
                x_new.append(temp)
                lengths_new.append(len(temp))
            temp2 = ''
            temp2 = temp2+x[j][:nest_position]+x[i]+x[j][nest_position:]
            if temp2 not in x_new:
                x_new.append(temp2)
                lengths_new.append(len(temp2))
            # print(len(temp2))

    y_new = generate_labels(x_new)




    return x_new, y_new, lengths_new

def generate_labels(sequences):

    sentences = []
    labels = []
    lengths = []

    for j in range(len(sequences)):
        x = sequences[j]
        stack = []
        stack_depth = 0
        label = ''
        for i in range(len(x)):



            if x[i]=='(':
                stack_depth += 1
                label = label + '1'
                stack.append('(')

            elif x[i]==')':

                if stack_depth>0:
                    stack_depth -= 1
                    stack.pop()
                    if stack_depth>0:
                        label = label + '1'
                    elif stack_depth==0:
                        label=label+'0'


                # elif stack_depth==0:
                #     label=label+'0'

        sentences.append(x)
        labels.append(label)
        lengths.append(len(x))
    return labels



# x, y, lengths = nest_sequences()
#
# print(len(x))
# print(len(y))
# print(len(lengths))
# # print(len(lengths)/2)
# for i in range(len(x)):
#     if lengths[i]!=2000:
#         print('length[',i,']!=2000, length = ',lengths[i])
#
print(generate_labels(['((()))','()()()', '(())()()((()))']))


def create_dataset():
    x, y, lengths = nest_sequences()

    print(len(x))
    print(len(y))
    print(len(lengths))
    # print(len(lengths)/2)
    for i in range(len(x)):
        if lengths[i] != 2000:
            print('length[', i, ']!=2000, length = ', lengths[i])

    with open('Dyck1_Dataset_Suzgun_2000tokens_nested.txt', 'a') as f:
        for i in range(len(x)):
            f.write(str(x[i]) + ',' + str(y[i]) + ',' + str(lengths[i]) + '\n')

    print('dataset written to document')

create_dataset()