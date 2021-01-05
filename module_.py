import torch
import torch.nn as nn
import torch.nn.functional as F


#卷积层
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1) #F.conv2d()

#BN层
bn = nn.BatchNorm2d(16) #F.batch_norm

#激活层
activae = nn.ReLU() #F.relu()
active = nn.LeakyReLU(0.1) #F.leaky_relu()
softmax = nn.Softmax(dim=1) #F.softmax()

#全连接层
fc = nn.Linear(2048, 1000) #F.linear()

#pooling层
pool = nn.MaxPool2d(3, stride=2) #F.max_pool2d()
pool = nn.AvgPool2d(3, stride=2) #F.avg_pool2d()

#上采样
x = F.upsample(x, scale_factor=2, mode="nearest")
x = F.interpolate(x, scale_factor=2, mode="nearest")

#dropout层
dropout = nn.Dropout2d(p=0.2) #p为丢弃概率 #F.dropout()
