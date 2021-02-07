"""
    hook一共有3个：
    1.  torch.autograd.Variable.register_hook (Python method, in Automatic differentiation package
    2.  torch.nn.Module.register_backward_hook (Python method, in torch.nn)
    3.  torch.nn.Module.register_forward_hook

"""

#x->R2, y=x+2，z=(y1**2 + y2**2)/2 ,通过梯度下降法求最小值
# import torch
#
# x = torch.randn(2,1,requires_grad=True)
# y = x + 2
# z = torch.mean(torch.pow(y,2))
# z.backward()
# lr = 1e-3
# x.data -= lr * x.grad.data

#但问题是，如果想要求中间变量y的梯度，系统会返回错误。
# print(y.grad)
#因此可以用hook, register_hook的作用是，当反传时，除了完成原有的反传，额外多完成一些任务。
# 你可以定义一个中间变量的hook，将它的grad值打印出来，当然你也可以定义一个全局列表，将每次的grad值添加到里面去。

grad_list = list()

def print_grad(grad):
    print(grad)
    grad_list.append(grad)

import torch

x = torch.randn(2,1,requires_grad=True)
y = x + 2
y.register_hook(print_grad)
z = torch.mean(torch.pow(y,2))
z.backward()
lr = 1e-3
x.data -= lr * x.grad.data

print(grad_list)

#register_forward_hook和register_backward_hook的用法和这个大同小异。
# 只不过对象从Variable改成了你自己定义的nn.Module。
# 当你训练一个网络，想要提取中间层的参数、或者特征图的时候，使用hook就能派上用场了。




