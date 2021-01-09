import torch
import os, time

#############GPU与CPU对比###############
device = "cuda" if torch.cuda.is_available() else "cpu"

#cpu
a = torch.randn(1, 13 * 13 * 3, 85)
b = torch.randn(1, 13 * 13 * 3, 85)

#gpu
a = a.to(device)
b = b.to(device)

s1 = time.time()
c = a * b
print("Run time: %f" % (time.time() - s1))

#############多GPU设置###############
#方法一
CUDA_VISIBLE_DEVICES = 1,2,3

#方法一
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

#方法三
torch.cuda.set_device(1,2,3)

##############并行计算设置###############
model = torch.nn.DataParallel(model, device_ids=[1,2,3])


