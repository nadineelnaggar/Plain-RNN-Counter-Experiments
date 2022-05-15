import torch
import torch.nn as nn

# x = 1000000000000000000000000000000000000

x = torch.tensor(1, dtype=torch.float32)
print(torch.sigmoid(x))


x = torch.tensor(1000000000, dtype=torch.float32)
print(torch.sigmoid(x))

x = torch.tensor(-1000000000, dtype=torch.float32)
print(torch.sigmoid(x))


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()