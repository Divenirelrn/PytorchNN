#TorchScript是PyTorch模型（nn.Module的子类）的中间表示，可以在高性能环境（例如C ++）中运行。
#共有两种方法将pytorch模型转成torch script ，一种是trace，另一种是script。
# 一般在模型内部没有控制流存在的话（if，for循环），直接用trace方法就可以了。
# 如果模型内部存在控制流，那就需要用到script方法了。

#trace
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)

    def forward(self,x):
    	x = self.conv1(x)
        return x

model = MyModule()  # 实例化模型
trace_module = torch.jit.trace(model,torch.rand(1,1,224,224))
print(trace_module.code)  # 查看模型结构
output = trace_module (torch.ones(1, 3, 224, 224)) # 测试
print(output)
trace_modult('model.pt') # 模型保存

#script
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(2,3,3)

    def forward(self,x):
        b,c,h,w = x.shape
        if c ==1:
            x = self.conv1(x)
        else:
            x = self.conv2(x)
        return x

model = MyModule()

# 这样写会报错，因为有控制流
# trace_module = torch.jit.trace(model,torch.rand(1,1,224,224))

# 此时应该用script方法
script_module = torch.jit.script(model)
print(script_module.code)
output = script_module(torch.rand(1,1,224,224))