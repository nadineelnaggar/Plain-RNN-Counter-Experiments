x = []
y = []
lengths = []


with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        x.append(sentence)
        # y.append(label)
        lengths.append(len(x))




def getTimestepDepths():
    for j in range(len(x)):
        elem = x[j]
        depth = 0
        for i in range(len(elem)):
            pass