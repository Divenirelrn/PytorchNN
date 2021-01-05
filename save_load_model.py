import torch

torch.save(net.state_dict(), "ckpt.mdl")
net.load_state_dict(torch.load("ckpt.mdl"))