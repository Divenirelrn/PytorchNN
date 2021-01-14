import torch.nn as nn


#nn.Modulelist, 类似于一个列表，支持extend、append等操作,但不能直接用于forward
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)])
        self.model.extend([nn.BatchNorm2d(16)])
        self.model.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for module in self.model:
            x = module(x)


#nn.ModuleDict, 类似于一个字典, 但不能直接用于forward
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.ModuleDict({
            "conv": nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            "bn": nn.BatchNorm2d(16),
            "relu": nn.ReLU()
        })

    def forward(self, x):
        for name, module in self.model.items():
            x = module(x)


# nn.Sequential, 将所有模块拼接成一个子模块，,可以直接用于forward
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("conv", nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1))
        self.model.add_module("bn", nn.BatchNorm2d(16))

    def forward(self, x):
        x = self.model(x)


