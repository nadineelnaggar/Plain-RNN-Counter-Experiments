x = []
y = []
lengths = []
timestep_depths = []


with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        x.append(sentence)
        # y.append(label)
        lengths.append(len(x))




def getTimestepDepths():
    # for j in range(len(x)):
    for j in range(3):
        elem = x[j]
        depth = 0
        depth_timestep = []
        print(elem)
        for i in range(len(elem)):
            print(elem[i])
            if elem[i] == '(':
                depth+=1
            elif elem[i]==')':
                depth-=1
            print(depth)
            depth_timestep.append(depth)

        print(depth_timestep)
        timestep_depths.append(depth_timestep)
        print(timestep_depths[j])
        # break
    print(timestep_depths)

getTimestepDepths()