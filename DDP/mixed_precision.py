rank = args.nr * args.gpus + gpu
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=args.world_size,
    rank=rank)

torch.manual_seed(0)
model = ConvNet()
torch.cuda.set_device(gpu)
model.cuda(gpu)
batch_size = 100
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(gpu)
optimizer = torch.optim.SGD(model.parameters(), 1e-4)
# Wrap the model
##############################################################
model, optimizer = amp.initialize(model, optimizer,
                                  opt_level='O2')
# amp.initialize 将模型和优化器为了进行后续混合精度训练而进行封装。
# 注意，在调用 amp.initialize 之前，模型模型必须已经部署在GPU上。
# opt_level 从 O0 （全部使用浮点数）一直到 O3 （全部使用半精度浮点数）。
# 而 O1 和 O2 属于不同的混合精度程度，具体可以参阅APEX的官方文档。注意之前数字前面的是大写字母O。
model = DDP(model)
#apex.parallel.DistributedDataParallel 是一个 nn.DistributedDataParallel 的替换版本。
# 我们不需要指定GPU，因为Apex在一个进程中只允许用一个GPU。
# 且它也假设程序在把模型搬到GPU之前已经调用了 torch.cuda.set_device(local_rank)(line 10)
##############################################################
# Data loading code
...
start = datetime.now()
total_step = len(train_loader)
for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        ##############################################################
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            #混合精度训练需要缩放损失函数以阻止梯度出现下溢。不过Apex会自动进行这些工作。
        ##############################################################
        optimizer.step()
...