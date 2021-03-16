#pip install thop

#查看resnet50
from torchvision.models import resnet50
from thop import profile
model = resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

#查看自己的模型
class YourModule(nn.Module):
    # your definition
def count_your_model(model, x, y):
    # your rule here

input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ),
                        custom_ops={YourModule: count_your_model})

#提升输出结果的可读性
from thop import clever_format
flops, params = clever_format([flops, params], "%.3f")