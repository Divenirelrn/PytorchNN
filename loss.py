import torch
import torch.nn as nn
import torch.nn.functional as F

"""
交叉熵损失：
    一般形式：-1/n sum(sum(-ylnf(x))) #本质上是one-hot标签与softmax计算的概率值计算内积
    二分类： -1/n sum(sum( -(plnq + (1-p)ln(1-q) )) #本质上是one-hot标签与sigmoid计算的概率值计算内积
    
    交叉熵损失函数由最大似然估计得来，最大化似然函数相当于最小化损失函数
    概率：已知参数求预测值为某类的概率（推理）；
    似然：已知预测值为某类的概率，求参数（训练）
"""


nn.MSELoss() #F.mse_loss()
nn.NLLLoss() #F.nll_loss()
nn.BCELoss()
nn.SmoothL1Loss() #F.smooth_l1_loss()
nn.CrossEntropyLoss()  #F.binary_cross_entropy()  F.cross_entropy()
                       #pytorch的CrossEntropyLoss()实现是log-softmax加NLLLOSS