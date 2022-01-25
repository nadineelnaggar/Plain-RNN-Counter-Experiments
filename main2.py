import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from models import VanillaRNN, VanillaLSTM, VanillaGRU

# SUZUGUN EXPERIMENT RUN HERE


model_name = 'LSTM'
# model_name='RNN'
# model_name='GRU'

# hidden_size=1
hidden_size=2
# hidden_size=3
# hidden_size=4
# hidden_size=5
# hidden_size=6

# task = 'TernaryClassification'
# task='BinaryClassification'
task='NextTokenPrediction'

# feedback='EndofSequence'
feedback='EveryTimeStep'

# num_bracket_pairs = 2
short_num_bracket_pairs_start = 1
num_bracket_pairs=25

length_bracket_pairs=50

use_optimiser='Adam'

learning_rate=0.001

num_layers=1
num_epochs=20
num_runs=10

output_activation='Softmax'

if task=='NextTokenPrediction':
    feedback='EveryTimeStep'
    output_activation='Sigmoid'


file_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'.txt'
excel_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'.xlsx'
modelname = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_MODEL.pth'
optimname = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_OPTIMISER.pth'
train_log= 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_TRAIN_LOG.txt'
test_log = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_TEST_LOG.txt'
long_test_log = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_LONG_TEST_LOG.txt'
plot_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_PLOT.png'



num_classes=2

if task=='TernaryClassification':
    num_classes=3
    labels=['valid', 'incomplete', 'invalid']
elif task=='BinaryClassification':
    num_classes=2
    labels=['valid','invalid']


vocab = ['(',')']
n_letters = len(vocab)
input_size=n_letters

X=[]
y=[]
data=[]

X_long=[]
y_long = []
data_long = []

# if task=='TernaryClassification':
#     if num_bracket_pairs==5:
#         with open("datasets/Dyck1_Ternary_Dataset_1to5pairs_balanced.txt", 'r') as f:
#             for line in f:
#                 line = line.split(",")
#                 sentence = line[0].strip()
#                 label = line[1].strip()
#                 X.append(sentence)
#                 y.append(label)
#                 data.append((sentence, label))
#     elif num_bracket_pairs==25:
#         with open("datasets/Dyck1_TernaryDataset_1to25pairs_12000elements_balanced.txt", 'r') as f:
#             for line in f:
#                 line = line.split(",")
#                 sentence = line[0].strip()
#                 label = line[1].strip()
#                 X.append(sentence)
#                 y.append(label)
#                 data.append((sentence, label))
#         with open("datasets/Dyck1_TernaryDataset_26to50pairs_6000elements_balanced.txt", 'r') as f:
#             for line in f:
#                 line = line.split(",")
#                 sentence = line[0].strip()
#                 label = line[1].strip()
#                 X_long.append(sentence)
#                 y_long.append(label)
#                 data_long.append((sentence, label))


if task=='TernaryClassification':
    with open('Dyck1_Dataset_Ternary_train__.txt', 'r') as f:
        for line in f:
            print('hello')

elif task=='NextTokenPrediction':
    with open('Dyck1_Dataset_Suzgun_train.txt', 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            X.append(sentence)
            y.append(label)
    with open('Dyck1_Dataset_Suzgun_test.txt', 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            X_long.append(sentence)
            y_long.append(label)



def encode_sentence(sentence, dataset='short'):
    # max_length=1
    # if dataset=='short' and model_name!='FFStack' and task=='BinaryClassification':
    if dataset == 'short':
        max_length=2*num_bracket_pairs
    elif dataset=='long':
        max_length=2*length_bracket_pairs
    rep = torch.zeros(max_length,1,n_letters)
    if len(sentence)<max_length:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index+(max_length-len(sentence))][0][pos] = 1
    else:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index][0][pos]=1
    rep.requires_grad_(True)
    return rep

def encode_labels(label, dataset='short'):

    # if output_activation=='Sigmoid' or output_activation=='Clipping':
    #     # return torch.tensor([labels.index(label)], dtype=torch.float32)
    #     if model_name=='VanillaRNN' or model_name=='VanillaLSTM' or model_name=='VanillaGRU':
    #         return torch.tensor([labels.index(label)],dtype=torch.float32)
    #     else:
    #         return torch.tensor(labels.index(label), dtype=torch.float32)
    # elif output_activation=='Softmax' and task=='TernaryClassification':
    if output_activation=='Softmax' and task=='TernaryClassification' and feedback=='EndofSequence':
        if label == 'valid':
            return torch.tensor([1,0,0],dtype=torch.float32)
        elif label=='incomplete':
            return torch.tensor([0,1,0],dtype=torch.float32)
        elif label == 'invalid':
            return torch.tensor([0,0,1],dtype=torch.float32)
    elif task=='NextTokenPrediction':
        if dataset == 'short':
            max_length = 2 * num_bracket_pairs
        elif dataset == 'long':
            max_length = 2 * length_bracket_pairs
        rep = torch.zeros(max_length, 1, n_letters)

        # output_vals = torch.zeros(1, 1, max_length)
        # for index, char in enumerate(label):
        #     if char == '1':
        #         output_vals[0][0][index] = 1
        #     elif char == '0':
        #         output_vals[0][0][index] = 0
        # return output_vals
        # output_vals = torch.zeros(1, max_length, num_classes)
        # for index, char in enumerate(label):
        #     if char == '1':
        #         output_vals[0][index][1] = 1
        #     elif char == '0':
        #         output_vals[0][index][0] = 1
        # output_vals = torch.zeros(1, max_length)
        # for index, char in enumerate(sentence):
        #     if char == '1':
        #         # output_vals[0][index] = 1
        #         output_vals[index] = 1
        #     elif char == '0':
        #         # output_vals[0][index] = 0
        #         output_vals[index]=0
        output_vals = torch.zeros(1, max_length, n_letters)
        # output_vals = torch.zeros(1, max_length,1)
        for index, char in enumerate(sentence):
            if char == '1':
                # output_vals[0][index] = 1
                output_vals[0][index][1] = 1
            elif char == '0':
                # output_vals[0][index] = 0
                output_vals[0][index][0] = 0
                output_vals[0][index] = torch.tensor(0,dtype=torch.float32)
    # elif output_activation == 'Softmax' and task == 'TernaryClassification' and feedback == 'EveryTimeStep':



def encode_dataset(sentences,labels, dataset='short'):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence, dataset))
        if feedback=='EveryTimeStep':
            max_length = 2*num_bracket_pairs
            rep = torch.zeros(1, max_length, num_classes)
            # print(rep)
            # sentence = X[i]
            stack = []
            depth = 0
            if len(sentence) <= max_length:
                for index, char in enumerate(sentence):
                    if char == '(':
                        stack.append(char)
                        if depth >= 0:
                            depth += 1
                        elif depth < 0:
                            depth = -1
                    elif char == ')':
                        if len(stack) > 0:
                            stack.pop()
                            depth -= 1
                        elif len(stack) <= 0:
                            depth = -1

                    if depth == 0:
                        pos = 0

                    elif depth > 0:
                        pos = 1
                    elif depth == -1:
                        if task=='TernaryClassification':
                            pos = 2
                        elif task=='BinaryClassification':
                            pos=1

                    # pos = vocab.index(char)
                    rep[0][index + (max_length - len(sentence))][pos] = 1
                encoded_labels.append(rep)
    if feedback=='EndofSequence':
        for label in labels:
            encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels

def classFromOutput(output):
    # if output.item() > 0.5:
    #     category_i = 1
    # else:
    #     category_i = 0
    # return labels[category_i], category_i
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0]
    return labels[category_i], category_i

X_train = []
X_test = []
y_train = []
y_test = []
X_long = []
y_long = []

if task!='NextTokenPrediction':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
elif task=='NextTokenPrediction':
    X_train = X[:int(0.7*len(X))]
    y_train = y[:int(0.7*len(X))]
    X_test = X[int(0.7*len(X)):]
    y_test = y[int(0.7*len(X)):]

print('length of training set = ',len(X_train))
print('length of test set = ',len(X_test))

X_notencoded = X
y_notencoded = y
X_train_notencoded = X_train
y_train_notencoded = y_train
X_test_notencoded = X_test
y_test_notencoded = y_test
X_train, y_train = encode_dataset(X_train,y_train)
X_test, y_test = encode_dataset(X_test,y_test)
X_encoded, y_encoded = encode_dataset(X,y)


X_long_notencoded = X_long
y_long_notencoded = y
X_long, y_long=encode_dataset(X_long,y_long,'long')



train_accuracies = []
train_dataframes = []
test_accuracies = []
long_test_accuracies = []

def train_model(model, task='NextTokenPrediction'):
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = []
    df1 = pd.DataFrame()

    weights_input = []
    biases_input = []
    weights_output = []
    biases_output = []
    gradients_input = []
    gradients_output = []
    confusion_matrices = []
    all_losses = []
    current_loss = 0
    all_epoch_incorrect_guesses = []
    accuracies = []
    print_flag = False

    for epoch in range(num_epochs):
        epochs.append(epoch)

        if epoch == (num_epochs - 1):
            print_flag = True
        if print_flag == True:
            with open(train_log, 'a') as f:
                f.write('\nEPOCH ' + str(epoch) + '\n')
        # confusion = torch.zeros(num_classes, num_classes)
        num_correct = 0
        num_samples = len(X_train)
        current_loss = 0
        epoch_incorrect_guesses = []
        predicted_classes = []
        expected_classes = []

        for i in range(len(X_train)):
            input_tensor = X_train[i]
            class_tensor = y_train[i]
            input_sentence = X_train_notencoded[i]
            class_category = y_train_notencoded[i]

            optimiser.zero_grad()

            if feedback=='EveryTimeStep':
                max_length = 2 * num_bracket_pairs
                # output_vals = torch.zeros(1, max_length, num_classes)
                output_vals = torch.zeros(1, max_length,num_classes)

            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('////////////////////////////////////////\n')
                    f.write('input sentence = ' + input_sentence + '\n')

                # print('////////////////////////////////////////')
                # print('input sentence = ', input_sentence)


            if model.model_name == 'VanillaLSTM':
                hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
            elif model.model_name == 'VanillaRNN' or model.model_name == 'VanillaGRU':
                hidden = torch.zeros(1, 1, hidden_size)


            for j in range(input_tensor.size()[0]):

                if model.model_name == 'VanillaLSTM' or model.model_name == 'VanillaRNN' or \
                        model.model_name == 'VanillaGRU':
                    # output, hidden = model(input_tensor[j].squeeze(), hidden)
                    output, hidden = model(input_tensor[j], hidden)
                    if feedback=='EveryTimeStep':
                        # output_vals[0][j]=output
                        output_vals[0][j]=output

            if feedback=='EndofSequence':
                loss = criterion(output, class_tensor)
            elif feedback=='EveryTimeStep':
                loss = criterion(output_vals,class_tensor)
            # if print_flag == True:
            #     print('Loss = ', loss)

            loss.backward()
            optimiser.step()
            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('output in train function = ' + str(output) + '\n')
                #     print('//////////////////////////////////////////\n')
                # print('output in train function = ', output)

            # guess, guess_i = classFromOutput(output)
            # class_i = labels.index(class_category)
            # confusion[class_i][guess_i] += 1
            # current_loss += loss
            # expected_classes.append(class_i)
            # predicted_classes.append(guess_i)

            output_vals_np = output_vals.detach().numpy()

            for index, elem in enumerate(output_vals_np):
                # if elem <= 0.5:
                #     elem = 0
                #     output_vals[index] = 0
                # elif elem > 0.5:
                #     elem = 1
                #     output_vals[index] = 1
                for idx, val in enumerate(elem):
                    for id, v in enumerate(val):
                        if v<=0.5:
                            v=0
                        elif v>0.5:
                            v=1
                        val[id] = v
                    # if val <= 0.5:
                    #     val = 0
                    #
                    # elif val > 0.5:
                    #     val = 1
                    # elem[idx] = val

            # if output_vals_np == y_train[i].detach().numpy():
            if np.array_equal(output_vals_np, y_train[i].detach().numpy()):
                num_correct+=1
                guess='correct'
            # elif output_vals!=y_train[i].detach().numpy():
            elif np.array_equal(output_vals_np, y_train[i].detach().numpy())==False:
                guess = 'incorrect'
                epoch_incorrect_guesses.append(input_sentence)
            # if guess == class_category:
            #     num_correct += 1

            # else:
            #     epoch_incorrect_guesses.append(input_sentence)
            if print_flag == True:
                with open(train_log, 'a') as f:
                    # f.write('predicted class = ' + guess + '\n')
                    # f.write('actual class = ' + class_category + '\n')
                    f.write('sentence = '+str(X_train[i].detach().numpy())+'\n')
                    f.write('predicted output = '+str(output_vals.detach().numpy())+'\n')
                    f.write('binarised predicted output = '+str(output_vals_np)+'\n')
                    f.write('actual output = '+str(y_train[i].detach().numpy())+'\n')
                    f.write(guess+'\n')


        accuracy = num_correct / len(X_train) * 100

        # if (epoch + 1) % 50 == 0:
        #     print('input weights = ', model.fc1.weight)
        #     print('input bias = ', model.fc1.bias)
        #     print('output weights  = ', model.fc2.weight)
        #     print('output bias = ', model.fc2.bias)
        #     print('input tensor gradient = ', input_tensor.grad)

        print('Accuracy for epoch ', epoch, '=', accuracy, '%')
        all_losses.append(current_loss / len(X_train))
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)

        accuracies.append(accuracy)

        # weights_input.append(model.fc1.weight.clone().detach().numpy())
        #
        #
        # gradients_input.append(model.fc1.weight.grad.clone().detach().numpy())
        #
        # biases_input.append(model.fc1.bias.clone().detach().numpy())
        #
        # weights_output.append(model.fc2.weight.clone().detach().numpy())
        #
        # gradients_output.append(model.fc2.weight.grad.clone().detach().numpy())
        #
        #
        # biases_output.append(model.fc2.bias.clone().detach().numpy())
        #
        # confusion_matrices.append(confusion)

        if epoch == num_epochs - 1:
            print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            # print('confusion matrix for training set\n', confusion)
            print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')

            if i == len(X_train) - 1:
                print('input tensor = ', input_tensor)

                print('final input sentence = ', input_sentence)


    df1['epoch'] = epochs
    # df1['input weights'] = weights_input
    # df1['input weight gradients'] = gradients_input
    # df1['input biases'] = biases_input
    # df1['output weights'] = weights_output
    # df1['output weight gradients'] = gradients_output
    # df1['output biases'] = biases_output
    df1['accuracies'] = accuracies
    df1['average epoch losses'] = all_losses
    # df1['confusion matrices'] = confusion_matrices
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses

    torch.save(model.state_dict(), modelname)
    torch.save(optimiser.state_dict(), optimname)

    print('all incorrect guesses in training across all epochs = \n', all_epoch_incorrect_guesses)
    return accuracy, df1

# model = VanillaLSTM(input_size,hidden_size,num_layers,num_classes)

# train_accuracy, _ = train_model(model,'TernaryClassification')
# print(train_accuracy)


def test_model(model, dataset='short'):

    model.eval()
    num_correct = 0
    if dataset=='short':
        num_samples = len(X_test)
        filename = test_log
    elif dataset=='long':
        num_samples = len(X_long)
        filename = long_test_log
    # num_samples = len(X_encoded)
    # confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    print_flag=True
    print('////////////////////////////////////////')
    print('TEST')
    with open(filename,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')
    with torch.no_grad():

        for i in range(num_samples):
            if dataset=='short':
                class_category = y_test_notencoded[i]
                class_tensor = y_test[i]
                input_sentence = X_test_notencoded[i]
                input_tensor = X_test[i]
            elif dataset=='long':
                class_category = y_long_notencoded[i]
                class_tensor=y_long[i]
                input_sentence = X_long_notencoded[i]
                input_tensor = X_long[i]



            # print('////////////////////////////////////////////')
            # print('Test sample = ',input_sentence)


            if feedback=='EveryTimeStep':
                max_length = 2 * num_bracket_pairs
                output_vals = torch.zeros(1, max_length, num_classes)

            with open(filename,'a') as f:
                f.write('/////////////////////////////////////////////\n')


            if model.model_name=='VanillaLSTM':
                hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
            elif model.model_name=='VanillaRNN' or model.model_name=='VanillaGRU':
                hidden = torch.zeros(1,1,hidden_size)


            for j in range(input_tensor.size()[0]):

                # print('input tensor[j][0] = ',input_tensor[j][0])
                with open(filename,'a') as f:
                    f.write('input tensor[j][0] = '+str(input_tensor[j][0])+'\n')




                output, hidden = model(input_tensor[j], hidden)

                if feedback == 'EveryTimeStep':
                    output_vals[0][j] = output
            # guess, guess_i = classFromOutput(output)
            # class_i = labels.index(class_category)

            output_vals_np = output_vals.detach().numpy()


            for index, elem in enumerate(output_vals_np):
                # if elem <= 0.5:
                #     elem = 0
                #     output_vals[index] = 0
                # elif elem > 0.5:
                #     elem = 1
                #     output_vals[index] = 1
                for idx, val in enumerate(elem):
                    for id, v in enumerate(val):
                        if v<=0.5:
                            v=0
                        elif v>0.5:
                            v=1
                        val[id] = v
                    # if val <= 0.5:
                    #     val = 0
                    #
                    # elif val > 0.5:
                    #     val = 1
                    # elem[idx] = val

            if dataset=='short':
                # if output_vals_np == y_test[i].detach.numpy():
                if np.array_equal(output_vals_np,y_test[i].detach().numpy()):
                    num_correct += 1
                    guess = 'correct'
                # elif output_vals_np != y_test[i].detach().numpy():
                elif np.array_equal(output_vals_np, y_test[i].detach().numpy())==False:
                    guess = 'incorrect'
                    # epoch_incorrect_guesses.append(input_sentence)
            elif dataset=='long':
                # if output_vals_np == y_long[i].detach.numpy():
                if np.array_equal(output_vals_np, y_long[i].detach.numpy()):
                    num_correct += 1
                    guess = 'correct'
                # elif output_vals_np != y_long[i].detach().numpy():
                elif np.array_equal(output_vals_np, y_long[i].detach().numpy()):
                    guess = 'incorrect'
                    # epoch_incorrect_guesses.append(input_sentence)


            # print('predicted class = ',guess)
            # print('actual class = ',class_category)
            with open(filename,'a') as f:
                # f.write('predicted class = '+guess+'\n')
                # f.write('actual class = '+class_category+'\n')
                if dataset=='short':
                    f.write('sentence = ' + str(X_test[i].detach().numpy()) + '\n')
                    f.write('predicted output = ' + str(output_vals.detach().numpy()) + '\n')
                    f.write('binarised predicted output = ' + str(output_vals_np) + '\n')
                    f.write('actual output = ' + str(y_test[i].detach().numpy()) + '\n')
                    f.write(guess + '\n')
                elif dataset=='long':
                    f.write('sentence = ' + str(X_long[i].detach().numpy()) + '\n')
                    f.write('predicted output = ' + str(output_vals.detach().numpy()) + '\n')
                    f.write('binarised predicted output = ' + str(output_vals_np) + '\n')
                    f.write('actual output = ' + str(y_long[i].detach().numpy()) + '\n')
                    f.write(guess + '\n')

            # confusion[class_i][guess_i] += 1
            # predicted_classes.append(guess_i)
            # expected_classes.append(class_i)
            # if guess == class_category:
            if guess=='correct':
                num_correct+=1
                correct_guesses.append(input_sentence)
            else:
                incorrect_guesses.append(input_sentence)


    accuracy = num_correct/num_samples*100
    # print('confusion matrix for test set \n',confusion)
    # conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
    print('correct guesses in testing = ', correct_guesses)
    print('incorrect guesses in testing = ', incorrect_guesses)

    with open(filename,'a') as f:
        f.write('test accuracy = '+str(accuracy)+'%\n')
        # f.write('confusion matrix for test set = \n'+str(confusion)+'\n')
        f.write('correct guesses in testing = '+str(correct_guesses)+'\n')
        f.write('incorrect guesses in testing = '+str(incorrect_guesses)+'\n')
    return accuracy



with open(file_name,'a') as f:
    f.write('Output activation = '+output_activation+'\n')
    f.write('Optimiser used = '+use_optimiser+'\n')
    f.write('Learning rate = '+str(learning_rate)+'\n')
    f.write('Number of runs = '+str(num_runs)+'\n')
    f.write('Number of epochs in each run = '+str(num_epochs)+'\n')
    f.write('Saved model name = '+modelname+'\n')
    f.write('Saved optimiser name = '+optimname+'\n')
    f.write('Excel name = '+excel_name+'\n')
    f.write('Train log name = '+train_log+'\n')
    f.write('Test log name = ' +test_log + '\n')
    f.write('Long test log name = ' + long_test_log + '\n')
    f.write('///////////////////////////////////////////////////////////////\n')
    f.write('\n')

for i in range(num_runs):
    if model_name=='LSTM':
        model = VanillaLSTM(input_size, hidden_size, num_layers, num_classes,output_activation)
    elif model_name=='GRU':
        model = VanillaGRU(input_size, hidden_size, num_layers, num_classes, output_activation)
    elif model_name=='RNN':
        model = VanillaRNN(input_size, hidden_size, num_layers, num_classes,output_activation)

    # train_accuracy = train_model(model)
    train_accuracy, df = train_model(model)
    train_accuracies.append(train_accuracy)
    train_dataframes.append(df)
    test_accuracy = test_model(model)
    test_accuracies.append(test_accuracy)

    long_test_accuracy = test_model(model, 'long')
    long_test_accuracies.append(long_test_accuracy)


    with open(file_name, "a") as f:
        f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
        f.write('test accuracy for run ' + str(i) + ' = ' + str(test_accuracy) + '%\n')

runs = []
for i in range(len(train_dataframes)):
    runs.append('run' + str(i))

dfs = dict(zip(runs, train_dataframes))
writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

for sheet_name in dfs.keys():
    dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

writer.save()

max_train_accuracy = max(train_accuracies)
min_train_accuracy = min(train_accuracies)
avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
std_train_accuracy = np.std(train_accuracies)

max_test_accuracy = max(test_accuracies)
min_test_accuracy = min(test_accuracies)
avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
std_test_accuracy = np.std(test_accuracies)


max_long_test_accuracy = max(long_test_accuracies)
min_long_test_accuracy = min(long_test_accuracies)
avg_long_test_accuracy = sum(long_test_accuracies) / len(test_accuracies)
std_long_test_accuracy = np.std(long_test_accuracies)

with open(file_name, "a") as f:
    f.write('/////////////////////////////////////////////////////////////////\n')
    f.write('Maximum train accuracy = ' + str(max_train_accuracy) + '%\n')
    f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
    f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
    f.write('Standard Deviation for train accuracy = ' + str(std_train_accuracy) + '\n')
    f.write('/////////////////////////////////////////////////////////////////\n')
    f.write('Maximum test accuracy = ' + str(max_test_accuracy) + '%\n')
    f.write('Minimum test accuracy = ' + str(min_test_accuracy) + '%\n')
    f.write('Average test accuracy = ' + str(avg_test_accuracy) + '%\n')
    f.write('Standard Deviation for test accuracy = ' + str(std_test_accuracy) + '\n')

    f.write('/////////////////////////////////////////////////////////////////\n')
    f.write('Maximum long test accuracy = ' + str(max_long_test_accuracy) + '%\n')
    f.write('Minimum long test accuracy = ' + str(min_long_test_accuracy) + '%\n')
    f.write('Average long test accuracy = ' + str(avg_long_test_accuracy) + '%\n')
    f.write('Standard Deviation for long test accuracy = ' + str(std_long_test_accuracy) + '\n')


"""
FIX THESE PLOTS AFTER ADDING LONG TEST SET TO THE MIX
"""

# width = 0.3
# plt.bar(np.arange(len(train_accuracies)), train_accuracies, width=width, label='Train Accuracy')
# plt.bar(np.arange(len(test_accuracies)) + width, test_accuracies, width=width, label='Test Accuracy')
# plt.bar(np.arange(len(long_test_accuracies)) + (2 * width), long_test_accuracies, width=width,
#                 label='Long Test Accuracy')
#         # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#         #            ncol=3, mode="expand", borderaxespad=0.)
# plt.xticks(np.arange(len(train_accuracies)))
#
#
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                ncol=3, mode="expand", borderaxespad=0.)
# plt.xticks(np.arange(len(train_accuracies)))
# plt.yticks(np.arange(0, 101, step=10))
# plt.ylabel('Accuracy (%)')
# plt.xlabel('Run Number')
#
# plt.savefig(plot_name)
# plt.show()









#######################################################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import sklearn
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
# from models import VanillaRNN, VanillaLSTM, VanillaGRU
#
#
#
#
# model_name = 'LSTM'
# # model_name='RNN'
# # model_name='GRU'
#
# # hidden_size=1
# hidden_size=2
# # hidden_size=3
# # hidden_size=4
# # hidden_size=5
# # hidden_size=6
#
# # task = 'TernaryClassification'
# # task='BinaryClassification'
# task='NextTokenPrediction'
#
# # feedback='EndofSequence'
# feedback='EveryTimeStep'
#
# # num_bracket_pairs = 2
# short_num_bracket_pairs_start = 1
# num_bracket_pairs=25
#
# length_bracket_pairs=50
#
# use_optimiser='Adam'
#
# learning_rate=0.001
#
# num_layers=1
# num_epochs=20
# num_runs=10
#
# output_activation='Softmax'
#
# if task=='NextTokenPrediction':
#     feedback='EveryTimeStep'
#     output_activation='Sigmoid'
#
#
# file_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'.txt'
# excel_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'.xlsx'
# modelname = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_MODEL.pth'
# optimname = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_OPTIMISER.pth'
# train_log= 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_TRAIN_LOG.txt'
# test_log = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_TEST_LOG.txt'
# long_test_log = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_LONG_TEST_LOG.txt'
# plot_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_Feedback_'+feedback+'_'+str(hidden_size)+'hidden_units_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'_PLOT.png'
#
#
#
# num_classes=2
#
# if task=='TernaryClassification':
#     num_classes=3
#     labels=['valid', 'incomplete', 'invalid']
# elif task=='BinaryClassification':
#     num_classes=2
#     labels=['valid','invalid']
#
#
# vocab = ['(',')']
# n_letters = len(vocab)
# input_size=n_letters
#
# X=[]
# y=[]
# data=[]
#
# X_long=[]
# y_long = []
# data_long = []
#
# # if task=='TernaryClassification':
# #     if num_bracket_pairs==5:
# #         with open("datasets/Dyck1_Ternary_Dataset_1to5pairs_balanced.txt", 'r') as f:
# #             for line in f:
# #                 line = line.split(",")
# #                 sentence = line[0].strip()
# #                 label = line[1].strip()
# #                 X.append(sentence)
# #                 y.append(label)
# #                 data.append((sentence, label))
# #     elif num_bracket_pairs==25:
# #         with open("datasets/Dyck1_TernaryDataset_1to25pairs_12000elements_balanced.txt", 'r') as f:
# #             for line in f:
# #                 line = line.split(",")
# #                 sentence = line[0].strip()
# #                 label = line[1].strip()
# #                 X.append(sentence)
# #                 y.append(label)
# #                 data.append((sentence, label))
# #         with open("datasets/Dyck1_TernaryDataset_26to50pairs_6000elements_balanced.txt", 'r') as f:
# #             for line in f:
# #                 line = line.split(",")
# #                 sentence = line[0].strip()
# #                 label = line[1].strip()
# #                 X_long.append(sentence)
# #                 y_long.append(label)
# #                 data_long.append((sentence, label))
#
#
# if task=='TernaryClassification':
#     with open('Dyck1_Dataset_Ternary_train__.txt', 'r') as f:
#         for line in f:
#             print('hello')
#
# elif task=='NextTokenPrediction':
#     with open('Dyck1_Dataset_Suzgun_train.txt', 'r') as f:
#         for line in f:
#             line = line.split(",")
#             sentence = line[0].strip()
#             label = line[1].strip()
#             X.append(sentence)
#             y.append(label)
#     with open('Dyck1_Dataset_Suzgun_test.txt', 'r') as f:
#         for line in f:
#             line = line.split(",")
#             sentence = line[0].strip()
#             label = line[1].strip()
#             X_long.append(sentence)
#             y_long.append(label)
#
#
#
# def encode_sentence(sentence, dataset='short'):
#     # max_length=1
#     # if dataset=='short' and model_name!='FFStack' and task=='BinaryClassification':
#     if dataset == 'short':
#         max_length=2*num_bracket_pairs
#     elif dataset=='long':
#         max_length=2*length_bracket_pairs
#     rep = torch.zeros(max_length,1,n_letters)
#     if len(sentence)<max_length:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[index+(max_length-len(sentence))][0][pos] = 1
#     else:
#         for index, char in enumerate(sentence):
#             pos = vocab.index(char)
#             rep[index][0][pos]=1
#     rep.requires_grad_(True)
#     return rep
#
# def encode_labels(label, dataset='short'):
#
#     # if output_activation=='Sigmoid' or output_activation=='Clipping':
#     #     # return torch.tensor([labels.index(label)], dtype=torch.float32)
#     #     if model_name=='VanillaRNN' or model_name=='VanillaLSTM' or model_name=='VanillaGRU':
#     #         return torch.tensor([labels.index(label)],dtype=torch.float32)
#     #     else:
#     #         return torch.tensor(labels.index(label), dtype=torch.float32)
#     # elif output_activation=='Softmax' and task=='TernaryClassification':
#     if output_activation=='Softmax' and task=='TernaryClassification' and feedback=='EndofSequence':
#         if label == 'valid':
#             return torch.tensor([1,0,0],dtype=torch.float32)
#         elif label=='incomplete':
#             return torch.tensor([0,1,0],dtype=torch.float32)
#         elif label == 'invalid':
#             return torch.tensor([0,0,1],dtype=torch.float32)
#     elif task=='NextTokenPrediction':
#         if dataset == 'short':
#             max_length = 2 * num_bracket_pairs
#         elif dataset == 'long':
#             max_length = 2 * length_bracket_pairs
#         rep = torch.zeros(max_length, 1, n_letters)
#
#         # output_vals = torch.zeros(1, 1, max_length)
#         # for index, char in enumerate(label):
#         #     if char == '1':
#         #         output_vals[0][0][index] = 1
#         #     elif char == '0':
#         #         output_vals[0][0][index] = 0
#         # return output_vals
#         # output_vals = torch.zeros(1, max_length, num_classes)
#         # for index, char in enumerate(label):
#         #     if char == '1':
#         #         output_vals[0][index][1] = 1
#         #     elif char == '0':
#         #         output_vals[0][index][0] = 1
#         output_vals = torch.zeros(1, max_length)
#         for index, char in enumerate(sentence):
#             if char == '1':
#                 output_vals[0][index] = 1
#             elif char == '0':
#                 output_vals[0][index] = 0
#     # elif output_activation == 'Softmax' and task == 'TernaryClassification' and feedback == 'EveryTimeStep':
#
#
#
# def encode_dataset(sentences,labels, dataset='short'):
#     encoded_sentences = []
#     encoded_labels = []
#     for sentence in sentences:
#         encoded_sentences.append(encode_sentence(sentence, dataset))
#         if feedback=='EveryTimeStep':
#             max_length = 2*num_bracket_pairs
#             rep = torch.zeros(1, max_length, num_classes)
#             # print(rep)
#             # sentence = X[i]
#             stack = []
#             depth = 0
#             if len(sentence) <= max_length:
#                 for index, char in enumerate(sentence):
#                     if char == '(':
#                         stack.append(char)
#                         if depth >= 0:
#                             depth += 1
#                         elif depth < 0:
#                             depth = -1
#                     elif char == ')':
#                         if len(stack) > 0:
#                             stack.pop()
#                             depth -= 1
#                         elif len(stack) <= 0:
#                             depth = -1
#
#                     if depth == 0:
#                         pos = 0
#
#                     elif depth > 0:
#                         pos = 1
#                     elif depth == -1:
#                         if task=='TernaryClassification':
#                             pos = 2
#                         elif task=='BinaryClassification':
#                             pos=1
#
#                     # pos = vocab.index(char)
#                     rep[0][index + (max_length - len(sentence))][pos] = 1
#                 encoded_labels.append(rep)
#     if feedback=='EndofSequence':
#         for label in labels:
#             encoded_labels.append(encode_labels(label))
#     return encoded_sentences, encoded_labels
#
# def classFromOutput(output):
#     # if output.item() > 0.5:
#     #     category_i = 1
#     # else:
#     #     category_i = 0
#     # return labels[category_i], category_i
#     top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
#     category_i = top_i[0]
#     return labels[category_i], category_i
#
# X_train = []
# X_test = []
# y_train = []
# y_test = []
# X_long = []
# y_long = []
#
# if task!='NextTokenPrediction':
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
# elif task=='NextTokenPrediction':
#     X_train = X[:int(0.7*len(X))]
#     y_train = y[:int(0.7*len(X))]
#     X_test = X[int(0.7*len(X)):]
#     y_test = y[int(0.7*len(X)):]
#
# print('length of training set = ',len(X_train))
# print('length of test set = ',len(X_test))
#
# X_notencoded = X
# y_notencoded = y
# X_train_notencoded = X_train
# y_train_notencoded = y_train
# X_test_notencoded = X_test
# y_test_notencoded = y_test
# X_train, y_train = encode_dataset(X_train,y_train)
# X_test, y_test = encode_dataset(X_test,y_test)
# X_encoded, y_encoded = encode_dataset(X,y)
#
#
# X_long_notencoded = X_long
# y_long_notencoded = y
# X_long, y_long=encode_dataset(X_long,y_long,'long')
# # X_long = []
# # y_long = []
# # data_long = []
#
# # print(X_train_notencoded)
# # print(X_train)
# # print(y_train_notencoded)
# # print(y_train)
#
#
# # with open("Dyck1_Ternary_Dataset_6to10pairs_balanced_length.txt", 'r') as f:
# #     for line in f:
# #         line = line.split(",")
# #         sentence = line[0].strip()
# #         label = line[1].strip()
# #         X_long.append(sentence)
# #         y_long.append(label)
# #         data.append((sentence, label))
# #
# # X_long_notencoded = X_long
# # y_long_notencoded = y_long
# # X_long, y_long = encode_dataset(X_long, y_long, dataset='long')
#
#
# train_accuracies = []
# train_dataframes = []
# test_accuracies = []
# long_test_accuracies = []
#
# def train_model(model, task='TernaryClassification'):
#     criterion = nn.MSELoss()
#     optimiser = optim.Adam(model.parameters(), lr=learning_rate)
#
#     epochs = []
#     df1 = pd.DataFrame()
#
#     weights_input = []
#     biases_input = []
#     weights_output = []
#     biases_output = []
#     gradients_input = []
#     gradients_output = []
#     confusion_matrices = []
#     all_losses = []
#     current_loss = 0
#     all_epoch_incorrect_guesses = []
#     accuracies = []
#     print_flag = False
#
#     for epoch in range(num_epochs):
#         epochs.append(epoch)
#
#         if epoch == (num_epochs - 1):
#             print_flag = True
#         if print_flag == True:
#             with open(train_log, 'a') as f:
#                 f.write('\nEPOCH ' + str(epoch) + '\n')
#         confusion = torch.zeros(num_classes, num_classes)
#         num_correct = 0
#         num_samples = len(X_train)
#         current_loss = 0
#         epoch_incorrect_guesses = []
#         predicted_classes = []
#         expected_classes = []
#
#         for i in range(len(X_train)):
#             input_tensor = X_train[i]
#             class_tensor = y_train[i]
#             input_sentence = X_train_notencoded[i]
#             class_category = y_train_notencoded[i]
#
#             optimiser.zero_grad()
#
#             if feedback=='EveryTimeStep':
#                 max_length = 2 * num_bracket_pairs
#                 # output_vals = torch.zeros(1, max_length, num_classes)
#                 output_vals = torch.zeros(1, max_length)
#
#             if print_flag == True:
#                 with open(train_log, 'a') as f:
#                     f.write('////////////////////////////////////////\n')
#                     f.write('input sentence = ' + input_sentence + '\n')
#
#                 # print('////////////////////////////////////////')
#                 # print('input sentence = ', input_sentence)
#
#
#             if model.model_name == 'VanillaLSTM':
#                 hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
#             elif model.model_name == 'VanillaRNN' or model.model_name == 'VanillaGRU':
#                 hidden = torch.zeros(1, 1, hidden_size)
#
#
#             for j in range(input_tensor.size()[0]):
#
#                 if model.model_name == 'VanillaLSTM' or model.model_name == 'VanillaRNN' or \
#                         model.model_name == 'VanillaGRU':
#                     # output, hidden = model(input_tensor[j].squeeze(), hidden)
#                     output, hidden = model(input_tensor[j], hidden)
#                     if feedback=='EveryTimeStep':
#                         # output_vals[0][j]=output
#                         output_vals[0][j] = output
#
#             if feedback=='EndofSequence':
#                 loss = criterion(output, class_tensor)
#             elif feedback=='EveryTimeStep':
#                 loss = criterion(output_vals,class_tensor)
#             # if print_flag == True:
#             #     print('Loss = ', loss)
#
#             loss.backward()
#             optimiser.step()
#             if print_flag == True:
#                 with open(train_log, 'a') as f:
#                     f.write('output in train function = ' + str(output) + '\n')
#                 #     print('//////////////////////////////////////////\n')
#                 # print('output in train function = ', output)
#
#             guess, guess_i = classFromOutput(output)
#             class_i = labels.index(class_category)
#             confusion[class_i][guess_i] += 1
#             current_loss += loss
#             expected_classes.append(class_i)
#             predicted_classes.append(guess_i)
#             if guess == class_category:
#                 num_correct += 1
#
#             else:
#                 epoch_incorrect_guesses.append(input_sentence)
#             if print_flag == True:
#                 with open(train_log, 'a') as f:
#                     f.write('predicted class = ' + guess + '\n')
#                     f.write('actual class = ' + class_category + '\n')
#
#         accuracy = num_correct / len(X_train) * 100
#
#         # if (epoch + 1) % 50 == 0:
#         #     print('input weights = ', model.fc1.weight)
#         #     print('input bias = ', model.fc1.bias)
#         #     print('output weights  = ', model.fc2.weight)
#         #     print('output bias = ', model.fc2.bias)
#         #     print('input tensor gradient = ', input_tensor.grad)
#
#         print('Accuracy for epoch ', epoch, '=', accuracy, '%')
#         all_losses.append(current_loss / len(X_train))
#         all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
#
#         accuracies.append(accuracy)
#
#         weights_input.append(model.fc1.weight.clone().detach().numpy())
#
#
#         gradients_input.append(model.fc1.weight.grad.clone().detach().numpy())
#
#         biases_input.append(model.fc1.bias.clone().detach().numpy())
#
#         weights_output.append(model.fc2.weight.clone().detach().numpy())
#
#         gradients_output.append(model.fc2.weight.grad.clone().detach().numpy())
#
#
#         biases_output.append(model.fc2.bias.clone().detach().numpy())
#
#         confusion_matrices.append(confusion)
#
#         if epoch == num_epochs - 1:
#             print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
#             print('confusion matrix for training set\n', confusion)
#             print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
#
#             if i == len(X_train) - 1:
#                 print('input tensor = ', input_tensor)
#
#                 print('final input sentence = ', input_sentence)
#
#
#     df1['epoch'] = epochs
#     df1['input weights'] = weights_input
#     df1['input weight gradients'] = gradients_input
#     df1['input biases'] = biases_input
#     df1['output weights'] = weights_output
#     df1['output weight gradients'] = gradients_output
#     df1['output biases'] = biases_output
#     df1['accuracies'] = accuracies
#     df1['average epoch losses'] = all_losses
#     df1['confusion matrices'] = confusion_matrices
#     df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses
#
#     torch.save(model.state_dict(), modelname)
#     torch.save(optimiser.state_dict(), optimname)
#
#     print('all incorrect guesses in training across all epochs = \n', all_epoch_incorrect_guesses)
#     return accuracy, df1
#
# # model = VanillaLSTM(input_size,hidden_size,num_layers,num_classes)
#
# # train_accuracy, _ = train_model(model,'TernaryClassification')
# # print(train_accuracy)
#
#
# def test_model(model, dataset='short'):
#
#     model.eval()
#     num_correct = 0
#     if dataset=='short':
#         num_samples = len(X_test)
#         filename = test_log
#     elif dataset=='long':
#         num_samples = len(X_long)
#         filename = long_test_log
#     # num_samples = len(X_encoded)
#     confusion = torch.zeros(num_classes, num_classes)
#     expected_classes = []
#     predicted_classes = []
#     correct_guesses = []
#     incorrect_guesses = []
#     print_flag=True
#     print('////////////////////////////////////////')
#     print('TEST')
#     with open(filename,'a') as f:
#         f.write('////////////////////////////////////////\n')
#         f.write('TEST '+dataset+'\n')
#     with torch.no_grad():
#
#         for i in range(num_samples):
#             if dataset=='short':
#                 class_category = y_test_notencoded[i]
#                 class_tensor = y_test[i]
#                 input_sentence = X_test_notencoded[i]
#                 input_tensor = X_test[i]
#             # elif dataset=='long':
#             #     class_category = y_long_notencoded[i]
#             #     class_tensor=y_long[i]
#             #     input_sentence = X_long_notencoded[i]
#             #     input_tensor = X_long[i]
#
#
#
#             # print('////////////////////////////////////////////')
#             # print('Test sample = ',input_sentence)
#
#
#             if feedback=='EveryTimeStep':
#                 max_length = 2 * num_bracket_pairs
#                 output_vals = torch.zeros(1, max_length, num_classes)
#
#             with open(filename,'a') as f:
#                 f.write('/////////////////////////////////////////////\n')
#
#
#             if model.model_name=='VanillaLSTM':
#                 hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
#             elif model.model_name=='VanillaRNN' or model.model_name=='VanillaGRU':
#                 hidden = torch.zeros(1,1,hidden_size)
#
#
#             for j in range(input_tensor.size()[0]):
#
#                 # print('input tensor[j][0] = ',input_tensor[j][0])
#                 with open(filename,'a') as f:
#                     f.write('input tensor[j][0] = '+str(input_tensor[j][0])+'\n')
#
#
#
#
#                 output, hidden = model(input_tensor[j], hidden)
#
#                 if feedback == 'EveryTimeStep':
#                     output_vals[0][j] = output
#             guess, guess_i = classFromOutput(output)
#             class_i = labels.index(class_category)
#             # print('predicted class = ',guess)
#             # print('actual class = ',class_category)
#             with open(filename,'a') as f:
#                 f.write('predicted class = '+guess+'\n')
#                 f.write('actual class = '+class_category+'\n')
#             confusion[class_i][guess_i] += 1
#             predicted_classes.append(guess_i)
#             expected_classes.append(class_i)
#             if guess == class_category:
#                 num_correct+=1
#                 correct_guesses.append(input_sentence)
#             else:
#                 incorrect_guesses.append(input_sentence)
#
#
#     accuracy = num_correct/num_samples*100
#     print('confusion matrix for test set \n',confusion)
#     conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
#     print('correct guesses in testing = ', correct_guesses)
#     print('incorrect guesses in testing = ', incorrect_guesses)
#
#     with open(filename,'a') as f:
#         f.write('test accuracy = '+str(accuracy)+'%\n')
#         f.write('confusion matrix for test set = \n'+str(confusion)+'\n')
#         f.write('correct guesses in testing = '+str(correct_guesses)+'\n')
#         f.write('incorrect guesses in testing = '+str(incorrect_guesses)+'\n')
#     return accuracy
#
#
#
# with open(file_name,'a') as f:
#     f.write('Output activation = '+output_activation+'\n')
#     f.write('Optimiser used = '+use_optimiser+'\n')
#     f.write('Learning rate = '+str(learning_rate)+'\n')
#     f.write('Number of runs = '+str(num_runs)+'\n')
#     f.write('Number of epochs in each run = '+str(num_epochs)+'\n')
#     f.write('Saved model name = '+modelname+'\n')
#     f.write('Saved optimiser name = '+optimname+'\n')
#     f.write('Excel name = '+excel_name+'\n')
#     f.write('Train log name = '+train_log+'\n')
#     f.write('Test log name = ' +test_log + '\n')
#     f.write('Long test log name = ' + long_test_log + '\n')
#     f.write('///////////////////////////////////////////////////////////////\n')
#     f.write('\n')
#
# for i in range(num_runs):
#     model = VanillaLSTM(input_size, hidden_size, num_layers, num_classes)
#
#     # train_accuracy = train_model(model)
#     train_accuracy, df = train_model(model)
#     train_accuracies.append(train_accuracy)
#     train_dataframes.append(df)
#     test_accuracy = test_model(model)
#     test_accuracies.append(test_accuracy)
#
#     long_test_accuracy = test_model(model, 'long')
#     long_test_accuracies.append(long_test_accuracy)
#
#
#     with open(file_name, "a") as f:
#         f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
#         f.write('test accuracy for run ' + str(i) + ' = ' + str(test_accuracy) + '%\n')
#
# runs = []
# for i in range(len(train_dataframes)):
#     runs.append('run' + str(i))
#
# dfs = dict(zip(runs, train_dataframes))
# writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
#
# for sheet_name in dfs.keys():
#     dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
#
# writer.save()
#
# max_train_accuracy = max(train_accuracies)
# min_train_accuracy = min(train_accuracies)
# avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
# std_train_accuracy = np.std(train_accuracies)
#
# max_test_accuracy = max(test_accuracies)
# min_test_accuracy = min(test_accuracies)
# avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
# std_test_accuracy = np.std(test_accuracies)
#
#
# max_long_test_accuracy = max(long_test_accuracies)
# min_long_test_accuracy = min(long_test_accuracies)
# avg_long_test_accuracy = sum(long_test_accuracies) / len(test_accuracies)
# std_long_test_accuracy = np.std(long_test_accuracies)
#
# with open(file_name, "a") as f:
#     f.write('/////////////////////////////////////////////////////////////////\n')
#     f.write('Maximum train accuracy = ' + str(max_train_accuracy) + '%\n')
#     f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
#     f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
#     f.write('Standard Deviation for train accuracy = ' + str(std_train_accuracy) + '\n')
#     f.write('/////////////////////////////////////////////////////////////////\n')
#     f.write('Maximum test accuracy = ' + str(max_test_accuracy) + '%\n')
#     f.write('Minimum test accuracy = ' + str(min_test_accuracy) + '%\n')
#     f.write('Average test accuracy = ' + str(avg_test_accuracy) + '%\n')
#     f.write('Standard Deviation for test accuracy = ' + str(std_test_accuracy) + '\n')
#
#     f.write('/////////////////////////////////////////////////////////////////\n')
#     f.write('Maximum long test accuracy = ' + str(max_long_test_accuracy) + '%\n')
#     f.write('Minimum long test accuracy = ' + str(min_long_test_accuracy) + '%\n')
#     f.write('Average long test accuracy = ' + str(avg_long_test_accuracy) + '%\n')
#     f.write('Standard Deviation for long test accuracy = ' + str(std_long_test_accuracy) + '\n')
#
#
# """
# FIX THESE PLOTS AFTER ADDING LONG TEST SET TO THE MIX
# """
#
# # width = 0.3
# # plt.bar(np.arange(len(train_accuracies)), train_accuracies, width=width, label='Train Accuracy')
# # plt.bar(np.arange(len(test_accuracies)) + width, test_accuracies, width=width, label='Test Accuracy')
# # plt.bar(np.arange(len(long_test_accuracies)) + (2 * width), long_test_accuracies, width=width,
# #                 label='Long Test Accuracy')
# #         # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
# #         #            ncol=3, mode="expand", borderaxespad=0.)
# # plt.xticks(np.arange(len(train_accuracies)))
# #
# #
# # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
# #                ncol=3, mode="expand", borderaxespad=0.)
# # plt.xticks(np.arange(len(train_accuracies)))
# # plt.yticks(np.arange(0, 101, step=10))
# # plt.ylabel('Accuracy (%)')
# # plt.xlabel('Run Number')
# #
# # plt.savefig(plot_name)
# # plt.show()