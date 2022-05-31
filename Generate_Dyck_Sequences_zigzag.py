

"""
values to use: 1000, 500, 200, 100, 50

"""

# vals = [1000, 500, 250, 200, 125, 100, 50, 25, 20, 10]
# len1= 2000
# vals2 = [10, 5, 2, 1]
# len2 = 20
# print(len(vals))


def generate_sequences(values, length):
    sequences = []
    labels = []
    lenghts = []
    for i in range(len(values)):
        # char_count = 0
        sequence = ''
        label = ''
        num_increments = int(length/values[i])
        # num_increments = int(20/values[i])
        # print(num_increments)
        for inc in range(num_increments):
            if inc%2==0:
                for j in range(int(values[i])):
                    sequence=sequence+'('
                    label=label+'1'
            elif inc%2!=0:
                for j in range(int(values[i])):
                    sequence=sequence+')'
                    if j<(int(values[i])-1):
                        label+='1'
                    elif j==(int(values[i])-1):
                        label+='0'
        sequences.append(sequence)
        labels.append(label)
        lenghts.append(len(sequence))

    return sequences, labels, lenghts

# seqs, labs = generate_sequences(vals, len1)
# print(len(seqs))
# print(len(labs))
#
# for i in range(len(seqs)):
#     if len(seqs[i])!=2000:
#         print(len(seqs[i]))
#
# # print(seqs)
# # print(labs)
#
# seqs2, labs2 = generate_sequences(vals2, len2)
# print(seqs2)
# print(labs2)


def create_dataset(values, length):
    seqs, labs, lengths = generate_sequences(values, length)

    with open('Dyck1_Dataset_Suzgun_2000tokens_zigzag.txt', 'a') as f:
        for i in range(len(seqs)):
            f.write(str(seqs[i]) + ',' + str(labs[i]) + ',' + str(lengths[i]) + '\n')

    print('dataset written to document')


vals = [1000, 500, 250, 200, 125, 100, 50, 25, 20, 10]
len1 = 2000

create_dataset(vals, len1)

