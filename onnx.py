#End-to-end AlexNet from PyTorch to Caffe2

from torch.autograd import Variable
import torch.onnx
import torchvision

dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
model = torchvision.models.alexnet(pretrained=True).cuda()
torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)
#关键参数verbose=True使exporter可以打印出一种人类可读的网络表示

#验证
#conda install -c conda-forge onnx
import onnx

# Load the ONNX model
model = onnx.load("alexnet.proto")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

#pip install onnx-caffe2
# ...continuing from above
import onnx_caffe2.backend as backend
import numpy as np
# or "CPU"
rep = backend.prepare(model, device="CUDA:0")
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class onnx_caffe2.backend.Workspace)
outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print(outputs[0])