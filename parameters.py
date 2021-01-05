from torchvision import models

resnet18 = models.resnet18(pretrained=True)

for name, param in resnet18.named_parameters():
    writer.add_histogram(name, param.clone().cpu().item().numpy(), n_iter)

resnet18.parameters()
resnet18.named_parameters()