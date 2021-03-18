#Facebook宣布推出PyTorch Hub,一行代码调用所有模型：torch.hub
#PyTorch Hub的使用简单到不能再简单，不需要下载模型，只用了一个torch.hub.load()就完成了对图像分类模型AlexNet的调用。
import torch
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
model.eval()

'''
PyTorch Hub允许用户对已发布的模型执行以下操作：
1、查询可用的模型;
2、加载模型;
3、查询模型中可用的方法。
'''
#1、查询可用的模型
torch.hub.list('pytorch/vision')
#2、加载模型
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
#至于如何获得此模型的详细帮助信息，可以使用下面的API：
print(torch.hub.help('pytorch/vision', 'deeplabv3_resnet101'))
#如果模型的发布者后续加入错误修复和性能改进，用户也可以非常简单地获取更新，确保自己用到的是最新版本:
model = torch.hub.load(..., force_reload=True)
#对于另外一部分用户来说，稳定性更加重要，他们有时候需要调用特定分支的代码。例如pytorch_GAN_zoo的hub分支:
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True)
#3.查看模型可用方法
dir(model)
#如果你对forward方法感兴趣，使用help(model.forward) 了解运行运行该方法所需的参数:
help(model.forward)