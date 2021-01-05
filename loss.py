import torch
import torch.nn as nn
import torch.nn.functional as F


nn.MSELoss() #F.mse_loss()
nn.NLLLoss() #F.nll_loss()
nn.BCELoss()
nn.SmoothL1Loss() #F.smooth_l1_loss()
nn.CrossEntropyLoss()  #F.binary_cross_entropy()  F.cross_entropy()