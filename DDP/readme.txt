nn.DataParallel 使用起来更加简单（通常只要封装模型然后跑训练代码就ok了）。但是在每个训练批次（batch）中，因为模型的权重都是在
一个进程上先算出来 然后再把他们分发到每个GPU上，所以网络通信就成为了一个瓶颈，而GPU使用率也通常很低。
除此之外，nn.DataParallel 需要所有的GPU都在一个节点（一台机器）上，且并不支持 Apex 的 混合精度训练.

使用 nn.DistributedDataParallel 进行Multiprocessing可以在多个gpu之间复制该模型，每个gpu由一个进程控制。
这些GPU可以位于同一个节点上，也可以分布在多个节点上。每个进程都执行相同的任务，并且每个进程与所有其他进程通信。
只有梯度会在进程/GPU之间传播，这样网络通信就不至于成为一个瓶颈了。

上述的步骤要求需要多个进程，甚至可能是不同结点上的多个进程同步和通信。而Pytorch通过它的 distributed.init_process_group 函数实现。
这个函数需要知道如何找到进程0（process 0），一边所有的进程都可以同步，也知道了一共要同步多少进程。
每个独立的进程也要知道总共的进程数，以及自己在所有进程中的阶序（rank）,当然也要知道自己要用那张GPU。
总进程数称之为 world size。最后，每个进程都需要知道要处理的数据的哪一部分，这样批处理就不会重叠。
而Pytorch通过 nn.utils.data.DistributedSampler 来实现这种效果。

