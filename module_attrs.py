import torch
import torch.nn as nn


conv = nn.Conv2d(3, 16, kernel_size=2, padding=1, stride=1)
bn = nn.BatchNorm2d(16)

print(conv.__dict__)
print(conv.parameters())
print(conv.named_parameters())
print(conv.weight.data.shape)
print(conv.bias.data.shape)

print(bn.weight.data)
print(bn.bias.data)
print(bn.running_mean)
print(bn.running_var)