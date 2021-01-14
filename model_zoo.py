from torch.utils import model_zoo


state_dict = model_zoo.load_url("https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth")