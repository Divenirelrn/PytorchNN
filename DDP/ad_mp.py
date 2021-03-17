"""
我们需要一个脚本，用来启动一个进程的每一个GPU。每个进程需要知道使用哪个GPU，
以及它在所有正在运行的进程中的阶序（rank）。而且，我们需要在每个节点上运行脚本
"""

def main():
    """
    args.nodes 是我们使用的结点数
    args.gpus 是每个结点的GPU数.
    args.nr 是当前结点的阶序rank，这个值的取值范围是 0 到 args.nodes - 1.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes        #基于结点数以及每个结点的GPU数，
                                                    # 我们可以计算 world_size 或者需要运行的总进程数，这和总GPU数相等。
    os.environ['MASTER_ADDR'] = '10.57.23.164'      #告诉Multiprocessing模块去哪个IP地址找process 0以确保初始同步所有进程
    os.environ['MASTER_PORT'] = '8888'              #告诉Multiprocessing模块去哪个端口找process 0以确保初始同步所有进程
    mp.spawn(train, nprocs=args.gpus, args=(args,)) #现在，我们需要生成 args.gpus 个进程, 每个进程都运行 train(i, args),
                                                    # 其中 i 从 0 到 args.gpus - 1。注意, main() 在每个结点上都运行,
                                                    # 因此总共就有 args.nodes * args.gpus = args.world_size 个进程.
    #还可以：export MASTER_ADDR=10.57.23.164 和 export MASTER_PORT=8888
    #########################################################


def train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu #这里是该进程在所有进程中的全局rank（一个进程对应一个GPU）
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    #初始化进程并加入其他进程。这就叫做“blocking”，也就是说只有当所有进程都加入了,单个进程才会运行。
    # 这里使用了 nccl 后端，因为Pytorch文档说它是跑得最快的。 init_method 让进程组知道去哪里找到它需要的设置。
    # 在这里，它就在寻找名为 MASTER_ADDR 以及 MASTER_PORT 的环境变量，这些环境变量在 main 函数中设置过。
    # 当然，本来可以把world_size 设置成一个全局变量，不过本脚本选择把它作为一个关键字参量
    # （和当前进程的全局阶序global rank一样）
    ############################################################

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    #将模型封装为一个 DistributedDataParallel 模型。这将把模型复制到GPU上进行处理。
    ###############################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    # nn.utils.data.DistributedSampler 确保每个进程拿到的都是不同的训练数据切片。
    ################################################################

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        ##############################
        shuffle=False,  #
        ##############################
        num_workers=0,
        pin_memory=True,
        #############################
        sampler=train_sampler)  #
    #############################
    ...

"""
要在4个节点上运行它(每个节点上有8个gpu)，我们需要4个终端(每个节点上有一个)。在节点0上(由 main 中的第13行设置)：

python src/mnist-distributed.py -n 4 -g 8 -nr 0

而在其他的节点上：

python src/mnist-distributed.py -n 4 -g 8 -nr i

其中 i∈1,2,3. 换句话说，我们要把这个脚本在每个结点上运行脚本，让脚本运行 args.gpus 个进程以在训练开始之前同步每个进程。

注意，脚本中的batchsize设置的是每个GPU的batchsize，因此实际的batchsize要乘上总共的GPU数目（worldsize）。
"""