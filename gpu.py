import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

#并行计算
torch.nn.DataParallel()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

