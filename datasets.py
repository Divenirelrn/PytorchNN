from torch.utils.data import dataset, dataloader
from torchvision import models, datasets, transforms

"""
PyTorch的数据加载模块，一共涉及到Dataset，Sampler，Dataloader三个类

Dataset负责对raw data source封装，将其封装成Python可识别的数据结构，其必须提供提取数据个体的接口。
Dataset共有Map-style datasets和Iterable-style datasets两种：
    map-style dataset：实现了__getitem__和__len__接口，表示一个从索引/key到样本数据的map。比如：datasets[10]，就表示第10个样本。
    iterable-style dataset：实现了__iter__接口，表示在data samples上的一个Iterable（可迭代对象），这种形式的dataset非常不适合
    随机存取（代价太高），但非常适合处理流数据。比如：iter(datasets)获得迭代器，然后不断使用next迭代从而实现遍历。
Sampler负责提供一种遍历数据集所有元素索引的方式。
Dataloader负责加载数据，同时支持map-style和iterable-style Dataset，支持单进程/多进程，还可以设置loading order, batch size, 
pin memory等加载参数。

具体实践中，我们需要使用Dataset的子类，自己实现的或者现成的。
PyTorch为我们提供的现成的Dataset子类：
    TensorDataset
    IterableDataset
    ConcatDataset
    ChainDataset
    Subset
常用的为TensorDataset和IterableDataset.

PyTorch为我们提供了几种现成的Sampler子类：
SequentialSampler
RandomSampler
SubsetRandomSampler
WeightedRandomSampler
BatchSampler
DistributedSampler

SequentialSampler指定总是按照相同的次序，顺序地采样元素
RandomSampler提供了随机采样元素的方式。
BatchSampler包装另一个sampler（输入参数），用来产生一个mini-batch大小的索引，相当于是为dataloader提供了提取dataset的
1个mini-batch样本的索引。

dataloader:
    for data, label in train_loader:
    data, label = data.to(device), label.to(device).squeeze()
    opt.zero_grad()
    logits = model(data)
    loss = criterion(logits, label)
    
实战建议：
    Sampler我们一般不用管，直接使用DataLoader默认指定的就行
        如果是iterable-style dataset，默认使用_InfiniteConstantSampler：
        如果是map-style dataset，有shuffle则默认使用RandomSampler；没有shuffle则默认使用SequentialSampler
"""

datasets.ImageFolder()

len(testloader.dataset)

tud.random_split()