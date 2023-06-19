from Dyck1_Datasets import NextTokenPredictionTrainDataset, NextTokenPredictionValidationDataset, NextTokenPredictionLongTestDataset, NextTokenPredictionDataset2000tokens_zigzag, NextTokenPredictionDataset1000tokens
from Semi_Dyck1_Datasets import SemiDyck1TrainDataset, SemiDyck1ValidationDataset, SemiDyck1TestDataset, SemiDyck1Dataset2000tokens_zigzag, SemiDyck1Dataset1000tokens
import math
import numpy as np

Dyck1TrainDataset = NextTokenPredictionTrainDataset()
Dyck1ValidationDataset = NextTokenPredictionValidationDataset()
Dyck1LongTestDataset = NextTokenPredictionLongTestDataset()
Dyck11000TokenDataset = NextTokenPredictionDataset1000tokens()
Dyck1ZigzagDataset = NextTokenPredictionDataset2000tokens_zigzag()

SemiD1TrainDataset = SemiDyck1TrainDataset()
SemiD1ValidationDataset = SemiDyck1ValidationDataset()
SemiD1LongTestDataset = SemiDyck1TestDataset()
SemiD11000TokenDataset = SemiDyck1Dataset1000tokens()
SemiD1ZigzagDataset = SemiDyck1Dataset2000tokens_zigzag()


# dyck1_train_max_counts = []
# dyck1_train_avg_counts_sequences = []
# dyck_1_train_avg_max_counts_dataset = []
# dyck1_train_num_0_labels=0
# dyck1_train_num_1_labels = 0
# dyck1_train_avg_length = 0
#
# for i in range(len(Dyck1TrainDataset)):
#     elem = Dyck1TrainDataset[i]
#     # print(Dyck1TrainDataset[i]['x'])
#     seq = elem['x']
#     label = elem['y']
#     length = elem['length']
#
#     dyck1_train_num_0_labels = label.count('0')
#     dyck1_train_num_1_labels = label.count('1')
#     # print('count_0_labels = ',count_0_labels)
#     # print('count_1_labels = ',count_1_labels)
#
#     max_count = 0
#     count = 0
#     total_timestep_counts = []
#
#     for j in range(length):
#         if seq[j]=='(':
#             count+=1
#         elif seq[j]==')':
#             if count>0:
#                 count-=1
#         total_timestep_counts.append(count)
#         if count>max_count:
#             max_count=count
#
#     dyck1_train_max_counts.append(max_count)


def getCountLabels():
    print('*********************************')
    print('Dyck-1 Train Dataset')
    dyck1_train_num_0_labels=0
    dyck1_train_num_1_labels=0
    for i in range(len(Dyck1TrainDataset)):
        elem = Dyck1TrainDataset[i]
        # print(Dyck1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        dyck1_train_num_0_labels += label.count('0')
        dyck1_train_num_1_labels += label.count('1')
    
    print('num_0_labels = ',dyck1_train_num_0_labels)
    print('num_1_labels = ',dyck1_train_num_1_labels)
    print('percentage 0 labels = ',dyck1_train_num_0_labels/(dyck1_train_num_0_labels+dyck1_train_num_1_labels)*100,'%')
    
    print('**************************************')

    print('*********************************')
    print('Dyck-1 Validation Dataset')
    dyck1_val_num_0_labels = 0
    dyck1_val_num_1_labels = 0
    for i in range(len(Dyck1ValidationDataset)):
        elem = Dyck1ValidationDataset[i]
        # print(Dyck1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        dyck1_val_num_0_labels += label.count('0')
        dyck1_val_num_1_labels += label.count('1')

    print('num_0_labels = ', dyck1_val_num_0_labels)
    print('num_1_labels = ', dyck1_val_num_1_labels)
    print('percentage 0 labels = ',
          dyck1_val_num_0_labels / (dyck1_val_num_0_labels + dyck1_val_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Dyck-1 Long Test Dataset')
    dyck1_longtest_num_0_labels = 0
    dyck1_longtest_num_1_labels = 0
    for i in range(len(Dyck1LongTestDataset)):
        elem = Dyck1LongTestDataset[i]
        # print(Dyck1longtestDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        dyck1_longtest_num_0_labels += label.count('0')
        dyck1_longtest_num_1_labels += label.count('1')

    print('num_0_labels = ', dyck1_longtest_num_0_labels)
    print('num_1_labels = ', dyck1_longtest_num_1_labels)
    print('percentage 0 labels = ',
          dyck1_longtest_num_0_labels / (dyck1_longtest_num_0_labels + dyck1_longtest_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Dyck-1 1000token Dataset')
    dyck1_1000token_num_0_labels = 0
    dyck1_1000token_num_1_labels = 0
    for i in range(len(Dyck11000TokenDataset)):
        elem = Dyck11000TokenDataset[i]
        # print(Dyck11000tokenDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        dyck1_1000token_num_0_labels += label.count('0')
        dyck1_1000token_num_1_labels += label.count('1')

    print('num_0_labels = ', dyck1_1000token_num_0_labels)
    print('num_1_labels = ', dyck1_1000token_num_1_labels)
    print('percentage 0 labels = ',
          dyck1_1000token_num_0_labels / (dyck1_1000token_num_0_labels + dyck1_1000token_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Dyck-1 Zigzag Dataset')
    dyck1_zigzag_num_0_labels = 0
    dyck1_zigzag_num_1_labels = 0
    for i in range(len(Dyck1ZigzagDataset)):
        elem = Dyck1ZigzagDataset[i]
        # print(Dyck1zigzagDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        dyck1_zigzag_num_0_labels += label.count('0')
        dyck1_zigzag_num_1_labels += label.count('1')

    print('num_0_labels = ', dyck1_zigzag_num_0_labels)
    print('num_1_labels = ', dyck1_zigzag_num_1_labels)
    print('percentage 0 labels = ',
          dyck1_zigzag_num_0_labels / (dyck1_zigzag_num_0_labels + dyck1_zigzag_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 Train Dataset')
    semidyck1_train_num_0_labels = 0
    semidyck1_train_num_1_labels = 0
    for i in range(len(SemiD1TrainDataset)):
        elem = SemiD1TrainDataset[i]
        # print(semidyck1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        semidyck1_train_num_0_labels += label.count('0')
        semidyck1_train_num_1_labels += label.count('1')

    print('num_0_labels = ', semidyck1_train_num_0_labels)
    print('num_1_labels = ', semidyck1_train_num_1_labels)
    print('percentage 0 labels = ',
          semidyck1_train_num_0_labels / (semidyck1_train_num_0_labels + semidyck1_train_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 Validation Dataset')
    semidyck1_val_num_0_labels = 0
    semidyck1_val_num_1_labels = 0
    for i in range(len(SemiD1ValidationDataset)):
        elem = SemiD1ValidationDataset[i]
        # print(semidyck1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        semidyck1_val_num_0_labels += label.count('0')
        semidyck1_val_num_1_labels += label.count('1')

    print('num_0_labels = ', semidyck1_val_num_0_labels)
    print('num_1_labels = ', semidyck1_val_num_1_labels)
    print('percentage 0 labels = ',
          semidyck1_val_num_0_labels / (semidyck1_val_num_0_labels + semidyck1_val_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 Long Test Dataset')
    semidyck1_longtest_num_0_labels = 0
    semidyck1_longtest_num_1_labels = 0
    for i in range(len(SemiD1LongTestDataset)):
        elem = SemiD1LongTestDataset[i]
        # print(semidyck1longtestDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        semidyck1_longtest_num_0_labels += label.count('0')
        semidyck1_longtest_num_1_labels += label.count('1')

    print('num_0_labels = ', semidyck1_longtest_num_0_labels)
    print('num_1_labels = ', semidyck1_longtest_num_1_labels)
    print('percentage 0 labels = ',
          semidyck1_longtest_num_0_labels / (semidyck1_longtest_num_0_labels + semidyck1_longtest_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 1000token Dataset')
    semidyck1_1000token_num_0_labels = 0
    semidyck1_1000token_num_1_labels = 0
    for i in range(len(SemiD11000TokenDataset)):
        elem = SemiD11000TokenDataset[i]
        # print(semidyck11000tokenDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        semidyck1_1000token_num_0_labels += label.count('0')
        semidyck1_1000token_num_1_labels += label.count('1')

    print('num_0_labels = ', semidyck1_1000token_num_0_labels)
    print('num_1_labels = ', semidyck1_1000token_num_1_labels)
    print('percentage 0 labels = ',
          semidyck1_1000token_num_0_labels / (semidyck1_1000token_num_0_labels + semidyck1_1000token_num_1_labels) * 100, '%')

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 Zigzag Dataset')
    semidyck1_zigzag_num_0_labels = 0
    semidyck1_zigzag_num_1_labels = 0
    for i in range(len(SemiD1ZigzagDataset)):
        elem = SemiD1ZigzagDataset[i]
        # print(semidyck1zigzagDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']

        semidyck1_zigzag_num_0_labels += label.count('0')
        semidyck1_zigzag_num_1_labels += label.count('1')

    print('num_0_labels = ', semidyck1_zigzag_num_0_labels)
    print('num_1_labels = ', semidyck1_zigzag_num_1_labels)
    print('percentage 0 labels = ',
          semidyck1_zigzag_num_0_labels / (semidyck1_zigzag_num_0_labels + semidyck1_zigzag_num_1_labels) * 100, '%')

    print('**************************************')


def getCounts():
    print('*********************************')
    print('Dyck-1 Train Dataset')
    dyck1_train_max_counts = []
    dyck1_train_timestep_counts = []
    dyck1_train_avg_timestep_counts = []
    dyck1_train_avg_dataset_count = 0

    for i in range(len(Dyck1TrainDataset)):
        max_count=0
        count=0
        total_counts = 0
        timestep_counts = []
        elem = Dyck1TrainDataset[i]
        # print(Dyck1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(length):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts+=count
        dyck1_train_max_counts.append(max_count)
        dyck1_train_avg_timestep_counts.append(total_counts/length)
        # dyck1_train_avg_timestep_counts.append(np.average(dyck1_train_timestep_counts))
    dyck1_train_avg_dataset_count=np.average(dyck1_train_avg_timestep_counts)
    dyck1_train_avg_max_count = np.average(dyck1_train_max_counts)

    print('average dataset count = ',dyck1_train_avg_dataset_count)
    print('avg max count = ',dyck1_train_avg_max_count)



    print('**************************************')

    print('*********************************')
    print('Dyck-1 val Dataset')
    dyck1_val_max_counts = []
    dyck1_val_timestep_counts = []
    dyck1_val_avg_timestep_counts = []
    dyck1_val_avg_dataset_count = 0

    for i in range(len(Dyck1ValidationDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = Dyck1ValidationDataset[i]
        # print(Dyck1valDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        dyck1_val_max_counts.append(max_count)
        dyck1_val_avg_timestep_counts.append(total_counts / len(seq))
        # dyck1_val_avg_timestep_counts.append(np.average(dyck1_val_timestep_counts))
    dyck1_val_avg_dataset_count = np.average(dyck1_val_avg_timestep_counts)
    dyck1_val_avg_max_count = np.average(dyck1_val_max_counts)

    print('average dataset count = ', dyck1_val_avg_dataset_count)
    print('avg max count = ', dyck1_val_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Dyck-1 Long Test Dataset')
    dyck1_longtest_max_counts = []
    dyck1_longtest_timestep_counts = []
    dyck1_longtest_avg_timestep_counts = []
    dyck1_longtest_avg_dataset_count = 0

    for i in range(len(Dyck1LongTestDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = Dyck1LongTestDataset[i]
        # print(Dyck1longtestDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(length):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        dyck1_longtest_max_counts.append(max_count)
        dyck1_longtest_avg_timestep_counts.append(total_counts / length)
        # dyck1_longtest_avg_timestep_counts.append(np.average(dyck1_longtest_timestep_counts))
    dyck1_longtest_avg_dataset_count = np.average(dyck1_longtest_avg_timestep_counts)
    dyck1_longtest_avg_max_count = np.average(dyck1_longtest_max_counts)

    print('average dataset count = ', dyck1_longtest_avg_dataset_count)
    print('avg max count = ', dyck1_longtest_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Dyck-1 1000token Dataset')
    dyck1_1000token_max_counts = []
    dyck1_1000token_timestep_counts = []
    dyck1_1000token_avg_timestep_counts = []
    dyck1_1000token_avg_dataset_count = 0

    for i in range(len(Dyck11000TokenDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = Dyck11000TokenDataset[i]
        # print(Dyck11000tokenDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(length):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        dyck1_1000token_max_counts.append(max_count)
        dyck1_1000token_avg_timestep_counts.append(total_counts / length)
        # dyck1_1000token_avg_timestep_counts.append(np.average(dyck1_1000token_timestep_counts))
    dyck1_1000token_avg_dataset_count = np.average(dyck1_1000token_avg_timestep_counts)
    dyck1_1000token_avg_max_count = np.average(dyck1_1000token_max_counts)

    print('average dataset count = ', dyck1_1000token_avg_dataset_count)
    print('avg max count = ', dyck1_1000token_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Dyck-1 zigzag Dataset')
    dyck1_zigzag_max_counts = []
    dyck1_zigzag_timestep_counts = []
    dyck1_zigzag_avg_timestep_counts = []
    dyck1_zigzag_avg_dataset_count = 0

    for i in range(len(Dyck1ZigzagDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = Dyck1ZigzagDataset[i]
        # print(Dyck1zigzagDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        dyck1_zigzag_max_counts.append(max_count)
        dyck1_zigzag_avg_timestep_counts.append(total_counts / len(seq))
        # dyck1_zigzag_avg_timestep_counts.append(np.average(dyck1_zigzag_timestep_counts))
    dyck1_zigzag_avg_dataset_count = np.average(dyck1_zigzag_avg_timestep_counts)
    dyck1_zigzag_avg_max_count = np.average(dyck1_zigzag_max_counts)

    print('average dataset count = ', dyck1_zigzag_avg_dataset_count)
    print('avg max count = ', dyck1_zigzag_avg_max_count)

    print('**************************************')
    
    
    print('*********************************')
    print('Semi Dyck-1 Train Dataset')
    semid1_train_max_counts = []
    semid1_train_timestep_counts = []
    semid1_train_avg_timestep_counts = []
    semid1_train_avg_dataset_count = 0

    for i in range(len(SemiD1TrainDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = SemiD1TrainDataset[i]
        # print(semid1TrainDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        semid1_train_max_counts.append(max_count)
        semid1_train_avg_timestep_counts.append(total_counts / len(seq))
        # semid1_train_avg_timestep_counts.append(np.average(semid1_train_timestep_counts))
    semid1_train_avg_dataset_count = np.average(semid1_train_avg_timestep_counts)
    semid1_train_avg_max_count = np.average(semid1_train_max_counts)

    print('average dataset count = ', semid1_train_avg_dataset_count)
    print('avg max count = ', semid1_train_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 val Dataset')
    semid1_val_max_counts = []
    semid1_val_timestep_counts = []
    semid1_val_avg_timestep_counts = []
    semid1_val_avg_dataset_count = 0

    for i in range(len(SemiD1ValidationDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = SemiD1ValidationDataset[i]
        # print(semid1valDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        semid1_val_max_counts.append(max_count)
        semid1_val_avg_timestep_counts.append(total_counts / len(seq))
        # semid1_val_avg_timestep_counts.append(np.average(semid1_val_timestep_counts))
    semid1_val_avg_dataset_count = np.average(semid1_val_avg_timestep_counts)
    semid1_val_avg_max_count = np.average(semid1_val_max_counts)

    print('average dataset count = ', semid1_val_avg_dataset_count)
    print('avg max count = ', semid1_val_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 Long Test Dataset')
    semid1_longtest_max_counts = []
    semid1_longtest_timestep_counts = []
    semid1_longtest_avg_timestep_counts = []
    semid1_longtest_avg_dataset_count = 0

    for i in range(len(SemiD1LongTestDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = SemiD1LongTestDataset[i]
        # print(semid1longtestDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        semid1_longtest_max_counts.append(max_count)
        semid1_longtest_avg_timestep_counts.append(total_counts / len(seq))
        # semid1_longtest_avg_timestep_counts.append(np.average(semid1_longtest_timestep_counts))
    semid1_longtest_avg_dataset_count = np.average(semid1_longtest_avg_timestep_counts)
    semid1_longtest_avg_max_count = np.average(semid1_longtest_max_counts)

    print('average dataset count = ', semid1_longtest_avg_dataset_count)
    print('avg max count = ', semid1_longtest_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 1000token Dataset')
    semid1_1000token_max_counts = []
    semid1_1000token_timestep_counts = []
    semid1_1000token_avg_timestep_counts = []
    semid1_1000token_avg_dataset_count = 0

    for i in range(len(SemiD11000TokenDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = SemiD11000TokenDataset[i]
        # print(semid11000tokenDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        semid1_1000token_max_counts.append(max_count)
        semid1_1000token_avg_timestep_counts.append(total_counts / len(seq))
        # semid1_1000token_avg_timestep_counts.append(np.average(semid1_1000token_timestep_counts))
    semid1_1000token_avg_dataset_count = np.average(semid1_1000token_avg_timestep_counts)
    semid1_1000token_avg_max_count = np.average(semid1_1000token_max_counts)

    print('average dataset count = ', semid1_1000token_avg_dataset_count)
    print('avg max count = ', semid1_1000token_avg_max_count)

    print('**************************************')

    print('*********************************')
    print('Semi Dyck-1 zigzag Dataset')
    semid1_zigzag_max_counts = []
    semid1_zigzag_timestep_counts = []
    semid1_zigzag_avg_timestep_counts = []
    semid1_zigzag_avg_dataset_count = 0

    for i in range(len(SemiD1ZigzagDataset)):
        max_count = 0
        count = 0
        total_counts = 0
        timestep_counts = []
        elem = SemiD1ZigzagDataset[i]
        # print(semid1zigzagDataset[i]['x'])
        seq = elem['x']
        label = elem['y']
        length = elem['length']
        for j in range(len(seq)):
            if seq[j] == '(':
                count += 1
            elif seq[j] == ')':
                if count > 0:
                    count -= 1
            timestep_counts.append(count)
            if count > max_count:
                max_count = count
            total_counts += count
        semid1_zigzag_max_counts.append(max_count)
        semid1_zigzag_avg_timestep_counts.append(total_counts / len(seq))
        # semid1_zigzag_avg_timestep_counts.append(np.average(semid1_zigzag_timestep_counts))
    semid1_zigzag_avg_dataset_count = np.average(semid1_zigzag_avg_timestep_counts)
    semid1_zigzag_avg_max_count = np.average(semid1_zigzag_max_counts)

    print('average dataset count = ', semid1_zigzag_avg_dataset_count)
    print('avg max count = ', semid1_zigzag_avg_max_count)

    print('**************************************')
    

getCountLabels()
getCounts()