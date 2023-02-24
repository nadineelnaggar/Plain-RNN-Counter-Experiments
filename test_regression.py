import torch
import torch.nn as nn
import numpy as np
import scipy
import pandas as pd
import sklearn
import seaborn as sns
import statsmodels as sm
from Dyck1_Datasets import NextTokenPredictionValidationDataset, NextTokenPredictionDataset1000tokens


weights_a = [1]
weights_b = [-1]
weights_u = [1]
biases_input=biases_u=[0]


# weights_a.append(1)
# weights_b.append(-1)
# biases_input.append(0)
# biases_u.append(0)
# weights_u=1

i=0
a_value = weights_a[i] + biases_u[i] + biases_input[i]
b_value = weights_b[i] + biases_u[i] + biases_input[i]
ab_ratio = a_value / b_value
u_dev = abs(weights_u[i] - 1)

ab_dev = abs(ab_ratio - -1)

print(a_value)
print(b_value)
print(ab_ratio)
print(u_dev)
print(ab_dev)

