import torch.optim as optim
import torch

optimizer = optim.SGD()
optimizer = optim.Adam()

optimizer.step()

optimizer.zero_grad()

with torch.no_grad():
    pass

