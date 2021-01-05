import torch


torch.Tensor.detach()
"""
 假设有modelA与modelB， 需要将modelA的output传给modelB， 
 我们只需要训练modelB，
 input_B = output_A.detach()
 可以使两张计算图的连接断开
"""