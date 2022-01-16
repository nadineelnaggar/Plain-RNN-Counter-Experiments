import torch

X=[]
y=[]
data=[]


with open("Dyck1_Ternary_Dataset_1to5pairs_balanced.txt", 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        X.append(sentence)
        # y.append(label)
        # data.append((sentence, label))


n_classes=3
max_length=10
vocab=['(',')']
classes = ['valid', 'incomplete','invalid']

rep = torch.zeros(1,max_length,n_classes)
print(rep)

for i in range(2):
    rep = torch.zeros(1, max_length, n_classes)
    # print(rep)
    sentence = X[i]
    stack=[]
    depth = 0
    if len(sentence)<=max_length:
        for index, char in enumerate(sentence):
            if char=='(':
                stack.append(char)
                if depth>=0:
                    depth+=1
                elif depth<0:
                    depth=-1
            elif char==')':
                if len(stack)>0:
                    stack.pop()
                    depth-=1
                elif len(stack)<=0:
                    depth=-1

            if depth==0:
                pos=0

            elif depth>0:
                pos = 1
            elif depth==-1:
                pos=2

            # pos = vocab.index(char)
            rep[0][index+(max_length-len(sentence))][pos] = 1
    # else:
    #
    #     for index, char in enumerate(sentence):
    #         if char=='(':
    #             stack.append(char)
    #             if depth>=0:
    #                 depth+=1
    #         elif char==')':
    #             if stack!=[]:
    #                 stack.pop()
    #             depth-=1
    #
    #         if depth==0:
    #             pos=0
    #
    #         elif depth>0:
    #             pos = 1
    #         elif depth<0:
    #             pos=2
    #
    #         # pos = vocab.index(char)
    #         rep[0][index][pos] = 1
    #     for index, char in enumerate(sentence):
    #         pos = vocab.index(char)
    #         rep[index][0][pos]=1
    rep.requires_grad_(True)
    # return rep
    y.append(rep)
    print(sentence)
    print(rep)