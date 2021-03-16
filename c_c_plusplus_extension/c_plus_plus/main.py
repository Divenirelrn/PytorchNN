import torch
from torch.autograd import Variable

from test import Test

x = Variable(torch.Tensor([1,2,3]), requires_grad=True)
y = Variable(torch.Tensor([4,5,6]), requires_grad=True)
test = Test()
z = test(x, y)
z.sum().backward()
print('x: ', x)
print('y: ', y)
print('z: ', z)
print('x.grad: ', x.grad)
print('y.grad: ', y.grad)