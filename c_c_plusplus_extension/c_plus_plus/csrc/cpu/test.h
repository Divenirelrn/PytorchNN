#include <torch/extension.h>
#include <vector>

// 前向传播
torch::Tensor Test_forward_cpu(const torch::Tensor& inputA,
                            const torch::Tensor& inputB);
// 反向传播
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput);

/*
注意，这里引用的 <torch/extension.h> 头文件至关重要，它主要包括三个重要模块：
pybind11，用于 C++ 和 python 交互；
ATen，包含 Tensor 等重要的函数和类；
一些辅助的头文件，用于实现 ATen 和 pybind11 之间的交互。
*/