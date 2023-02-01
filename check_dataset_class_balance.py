import Dyck1_Datasets
from Dyck1_Datasets import NextTokenPredictionTrainDataset, NextTokenPredictionValidationDataset, NextTokenPredictionShortTestDataset, NextTokenPredictionLongTestDataset





train_0s = 0
train_1s = 0

val_0s = 0
val_1s = 0

long_0s = 0
long_1s = 0


train_dataset = NextTokenPredictionTrainDataset()
val_dataset = NextTokenPredictionValidationDataset()
long_dataset = NextTokenPredictionLongTestDataset()

print(len(train_dataset))
for i in range(len(train_dataset)):
    data_entry = train_dataset[i]
    # print(data_entry['y'])
    output_label = data_entry['y']
    for j in range(len(output_label)):
        if output_label[j]=='0':
            train_0s+=1
        elif output_label[j]=='1':
            train_1s+=1


for i in range(len(val_dataset)):
    data_entry = val_dataset[i]
    # print(data_entry['y'])
    output_label = data_entry['y']
    for j in range(len(output_label)):
        if output_label[j]=='0':
            val_0s+=1
        elif output_label[j]=='1':
            val_1s+=1


for i in range(len(long_dataset)):
    data_entry = long_dataset[i]
    # print(data_entry['y'])
    output_label = data_entry['y']
    for j in range(len(output_label)):
        if output_label[j]=='0':
            long_0s+=1
        elif output_label[j]=='1':
            long_1s+=1

print('number of 0 labels in training set = ',train_0s)
print('number of 1 labels in training set = ',train_1s)
print('percentage of 0 labels = ',train_0s/(train_0s+train_1s)*100)

print('number of 0 labels in validation set = ',val_0s)
print('number of 1 labels in validation set = ',val_1s)
print('percentage of 0 labels = ',val_0s/(val_0s+val_1s)*100)

print('number of 0 labels in long test set = ',long_0s)
print('number of 1 labels in long test set = ',long_1s)
print('percentage of 0 labels = ',long_0s/(long_0s+long_1s)*100)