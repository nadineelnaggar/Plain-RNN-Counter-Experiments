import torch
import torch.nn as nn

# x = 1000000000000000000000000000000000000

x = torch.tensor(1, dtype=torch.float32)
print(torch.sigmoid(x))


x = torch.tensor(1000000000, dtype=torch.float32)
print(torch.sigmoid(x))

x = torch.tensor(-1000000000, dtype=torch.float32)
print(torch.sigmoid(x))

print(torch.sigmoid(torch.tensor(32768,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-32768,dtype=torch.float32)))

print(torch.tanh(torch.tensor(32768,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-32768,dtype=torch.float32)))


x = torch.tensor(65536,dtype=torch.float32)
print(torch.sigmoid(x))
print(torch.tanh(x))


print(torch.sigmoid(torch.tensor(16384,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-16384,dtype=torch.float32)))

print(torch.tanh(torch.tensor(16384,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-16384,dtype=torch.float32)))


print(torch.sigmoid(torch.tensor(8192,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-8192,dtype=torch.float32)))

print(torch.tanh(torch.tensor(8192,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-8192,dtype=torch.float32)))

print(torch.sigmoid(torch.tensor(2048,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-2048,dtype=torch.float32)))

print(torch.tanh(torch.tensor(2048,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-2048,dtype=torch.float32)))


print(torch.sigmoid(torch.tensor(1024,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-1024,dtype=torch.float32)))

print(torch.tanh(torch.tensor(1024,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-1024,dtype=torch.float32)))

print('64')
print(torch.sigmoid(torch.tensor(64,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-64,dtype=torch.float32)))

print(torch.tanh(torch.tensor(64,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-64,dtype=torch.float32)))

print('128')

print(torch.sigmoid(torch.tensor(128,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-128,dtype=torch.float32)))

print(torch.tanh(torch.tensor(128,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-128,dtype=torch.float32)))

print('32')

print(torch.sigmoid(torch.tensor(32,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-32,dtype=torch.float32)))

print(torch.tanh(torch.tensor(32,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-32,dtype=torch.float32)))


print('8')

print('sigmoid(8) = ',torch.sigmoid(torch.tensor(8,dtype=torch.float32)))
print('sigmoid(-8) = ',torch.sigmoid(torch.tensor(-8,dtype=torch.float32)))

print('tanh(8) = ',torch.tanh(torch.tensor(8,dtype=torch.float32)))
print('tanh(-8) = ',torch.tanh(torch.tensor(-8,dtype=torch.float32)))


print('4')

print(torch.sigmoid(torch.tensor(4,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-4,dtype=torch.float32)))

print(torch.tanh(torch.tensor(4,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-4,dtype=torch.float32)))

print('16')

print(torch.sigmoid(torch.tensor(16,dtype=torch.float32)))
print(torch.sigmoid(torch.tensor(-16,dtype=torch.float32)))

print(torch.tanh(torch.tensor(16,dtype=torch.float32)))
print(torch.tanh(torch.tensor(-16,dtype=torch.float32)))

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()