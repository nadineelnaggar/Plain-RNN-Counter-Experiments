import numpy as np
import random


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
        lengths.append(len(x))


temp = list(zip(x, y, lengths))
random.shuffle(temp)
x, y, lengths = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
x, y, lengths = list(x), list(y), list(lengths)


with open('Dyck1_Dataset_Suzgun_train_Shuffle.txt', 'a') as f:
  for i in range(len(x)):
    f.write(str(x[i])+','+str(y[i])+','+str(lengths[i])+'\n')