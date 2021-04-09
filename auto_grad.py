import torch.autograd

"""
计算图:
    计算图通常包含两种元素，一个是 tensor，另一个是 Function。张量 tensor 不必多说，但是大家可能对 Function 比较陌生。
    这里 Function 指的是在计算图中某个节点（node）所进行的运算，比如加减乘除卷积等等之类的，Function 内部有 forward() 
    和 backward() 两个方法，分别应用于正向、反向传播。
    
    a = torch.tensor(2.0, requires_grad=True)
    b = a.exp()
    print(b)
    # tensor(7.3891, grad_fn=<ExpBackward>)
    在我们做正向传播的过程中，除了执行 forward() 操作之外，还会同时会为反向传播做一些准备，为反向计算图添加 Function 节点。
    在上边这个例子中，变量 b 在反向传播中所需要进行的操作是 <ExpBackward> 。
    
具体例子：
    input = torch.ones([2, 2], requires_grad=False)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(3.0, requires_grad=True)
    w3 = torch.tensor(4.0, requires_grad=True)
    
    l1 = input * w1
    l2 = l1 + w2
    l3 = l1 * w3
    l4 = l2 * l3
    loss = l4.mean()
    
    
    print(w1.data, w1.grad, w1.grad_fn)
    # tensor(2.) None None
    
    print(l1.data, l1.grad, l1.grad_fn)
    # tensor([[2., 2.],
    #         [2., 2.]]) None <MulBackward0 object at 0x000001EBE79E6AC8>
    
    print(loss.data, loss.grad, loss.grad_fn)
    # tensor(40.) None <MeanBackward0 object at 0x000001EBE79D8208>
    
    input = [1.0, 1.0, 1.0, 1.0]
    w1 = [2.0, 2.0, 2.0, 2.0]
    w2 = [3.0, 3.0, 3.0, 3.0]
    w3 = [4.0, 4.0, 4.0, 4.0]
    
    l1 = input x w1 = [2.0, 2.0, 2.0, 2.0]
    l2 = l1 + w2 = [5.0, 5.0, 5.0, 5.0]
    l3 = l1 x w3 = [8.0, 8.0, 8.0, 8.0] 
    l4 = l2 x l3 = [40.0, 40.0, 40.0, 40.0] 
    loss = mean(l4) = 40.0
    
    loss.backward()

    print(w1.grad, w2.grad, w3.grad)
    # tensor(28.) tensor(8.) tensor(10.)
    print(l1.grad, l2.grad, l3.grad, l4.grad, loss.grad)
    # None None None None None
    
叶子张量：
    对于任意一个张量来说，我们可以用 tensor.is_leaf 来判断它是否是叶子张量（leaf tensor）。在反向传播过程中，
    只有 is_leaf=True 的时候，需要求导的张量的导数结果才会被最后保留下来。

    对于 requires_grad=False 的 tensor 来说，我们约定俗成地把它们归为叶子张量。但其实无论如何划分都没有影响，
    因为张量的 is_leaf 属性只有在需要求导的时候才有意义。

    我们真正需要注意的是当 requires_grad=True 的时候，如何判断是否是叶子张量：当这个 tensor 是用户创建的时候，
    它是一个叶子节点，当这个 tensor 是由其他运算操作产生的时候，它就不是一个叶子节点。我们来看个例子：
    
    a = torch.ones([2, 2], requires_grad=True)
    print(a.is_leaf)
    # True
    
    b = a + 2
    print(b.is_leaf)
    # False
    # 因为 b 不是用户创建的，是通过计算生成的
    
非叶节点的梯度：
    对于叶子节点来说，它们的 grad_fn 属性都为空；而对于非叶子结点来说，因为它们是通过一些操作生成的，所以它们的 grad_fn 不为空。
    我们有办法保留中间变量的导数吗？当然有，通过使用 tensor.retain_grad() 就可以：
    
    # 和前边一样
    # ...
    loss = l4.mean()
    
    l1.retain_grad()
    l4.retain_grad()
    loss.retain_grad()
    
    loss.backward()
    
    print(loss.grad)
    # tensor(1.)
    print(l4.grad)
    # tensor([[0.2500, 0.2500],
    #         [0.2500, 0.2500]])
    print(l1.grad)
    # tensor([[7., 7.],
    #         [7., 7.]])
    
    如果我们只是想进行 debug，只需要输出中间变量的导数信息，而不需要保存它们，我们还可以使用 tensor.register_hook，例子如下：
    
    # 和前边一样
    # ...
    loss = l4.mean()
    
    l1.register_hook(lambda grad: print('l1 grad: ', grad))
    l4.register_hook(lambda grad: print('l4 grad: ', grad))
    loss.register_hook(lambda grad: print('loss grad: ', grad))
    
    loss.backward()
    
    # loss grad:  tensor(1.)
    # l4 grad:  tensor([[0.2500, 0.2500],
    #         [0.2500, 0.2500]])
    # l1 grad:  tensor([[7., 7.],
    #         [7., 7.]])
    
    print(loss.grad)
    # None
    # loss 的 grad 在 print 完之后就被清除掉了
    
inplace 操作：
    现在我们来看一下本篇的重点，inplace operation。可以说，我们求导时候大部分的 bug，都出在使用了 inplace 操作上。
    现在我们以 PyTorch 不同的报错信息作为驱动，来讲一讲 inplace 操作吧。第一个报错信息：
    RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: balabala...

    我们来看两种情况，大家觉得这两种情况哪个是 inplace 操作，哪个不是？或者两个都是 inplace？

    # 情景 1
    a = a.exp()
    
    # 情景 2
    a[0] = 10
    答案是：情景1不是 inplace，类似 Python 中的 i=i+1, 而情景2是 inplace 操作，类似 i+=1。
    
    我们来实际测试一下：
    # 我们要用到 id() 这个函数，其返回值是对象的内存地址
    # 情景 1
    a = torch.tensor([3.0, 1.0])
    print(id(a)) # 2112716404344
    a = a.exp()
    print(id(a)) # 2112715008904
    # 在这个过程中 a.exp() 生成了一个新的对象，然后再让 a
    # 指向它的地址，所以这不是个 inplace 操作
    
    # 情景 2
    a = torch.tensor([3.0, 1.0])
    print(id(a)) # 2112716403840
    a[0] = 10
    print(id(a), a) # 2112716403840 tensor([10.,  1.])
    # inplace 操作，内存地址没变
    
    PyTorch 是怎么检测 tensor 发生了 inplace 操作呢？答案是通过 tensor._version 来检测的。我们还是来看个例子：
    
    a = torch.tensor([1.0, 3.0], requires_grad=True)
    b = a + 2
    print(b._version) # 0
    
    loss = (b * b).mean()
    b[0] = 1000.0
    print(b._version) # 1
    
    loss.backward()
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation ...
    
    每次 tensor 在进行 inplace 操作时，变量 _version 就会加1，其初始值为0。在正向传播过程中，求导系统记录的 b 
    的 version 是0，但是在进行反向传播的过程中，求导系统发现 b 的 version 变成1了，所以就会报错了。但是还有一
    种特殊情况不会报错，就是反向传播求导的时候如果没用到 b 的值（比如 y=x+1， y 关于 x 的导数是1，和 x 无关），
    自然就不会去对比 b 前后的 version 了，所以不会报错。
    
    上边我们所说的情况是针对非叶子节点的，对于 requires_grad=True 的叶子节点来说，要求更加严格了，甚至在叶子节点被使用
    之前修改它的值都不行。我们来看一个报错信息：

    RuntimeError: leaf variable has been moved into the graph interior
    这个意思通俗一点说就是你的一顿 inplace 操作把一个叶子节点变成了非叶子节点了。我们知道，非叶子节点的导数在默认情况
    下是不会被保存的，这样就会出问题了。举个小例子：
    
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf)
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True
    
    a[:] = 0
    print(a, a.is_leaf)
    # tensor([0., 0., 0., 0.], grad_fn=<CopySlices>) False
    
    loss = (a*a).mean()
    loss.backward()
    # RuntimeError: leaf variable has been moved into the graph interior
            
    我们看到，在进行对 a 的重新 inplace 赋值之后，表示了 a 是通过 copy operation 生成的，grad_fn 都有了，所以自然而然不
    是叶子节点了。本来是该有导数值保留的变量，现在成了导数会被自动释放的中间变量了，所以 PyTorch 就给你报错了。还有另外一种情况：
    
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    a.add_(10.) # 或者 a += 10.
    # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    
    这个更厉害了，不等到你调用 backward，只要你对需要求导的叶子张量使用了这些操作，马上就会报错。那是不是需要求导的叶子
    节点一旦被初始化赋值之后，就不能修改它们的值了呢？我们如果在某种情况下需要重新对叶子变量赋值该怎么办呢？有办法！
    
    # 方法一
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf, id(a))
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True 2501274822696
    
    a.data.fill_(10.)
    # 或者 a.detach().fill_(10.)
    print(a, a.is_leaf, id(a))
    # tensor([10., 10., 10., 10.], requires_grad=True) True 2501274822696
    
    loss = (a*a).mean()
    loss.backward()
    print(a.grad)
    # tensor([5., 5., 5., 5.])
    
    # 方法二
    a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
    print(a, a.is_leaf)
    # tensor([10.,  5.,  2.,  3.], requires_grad=True) True
    
    with torch.no_grad():
        a[:] = 10.
    print(a, a.is_leaf)
    # tensor([10., 10., 10., 10.], requires_grad=True) True
    
    loss = (a*a).mean()
    loss.backward()
    print(a.grad)
    # tensor([5., 5., 5., 5.])
    
    总之，我们在实际写代码的过程中，没有必须要用 inplace operation 的情况，而且支持它会带来很大的性能上的牺牲，所以 
    PyTorch 不推荐使用 inplace 操作，当求导过程中发现有 inplace 操作影响求导正确性的时候，会采用报错的方式提醒。但
    这句话反过来说就是，因为只要有 inplace 操作不当就会报错，所以如果我们在程序中使用了 inplace 操作却没报错，那么
    说明我们最后求导的结果是正确的，没问题的。这就是我们常听见的没报错就没有问题。

静态图与动态图：
    除了动态图之外，PyTorch 还有一个特性，叫 eager execution。意思就是当遇到 tensor 计算的时候，马上就回去执行计算，
    也就是，实际上 PyTorch 根本不会去构建正向计算图，而是遇到操作就执行。真正意义上的正向计算图是把所有的操作都添加
    完，构建好了之后，再运行神经网络的正向传播。
    
    正是因为 PyTorch 的两大特性：动态图和 eager execution，所以它用起来才这么顺手，简直就和写 Python 程序一样舒服，
    debug 也非常方便。除此之外，我们从之前的描述也可以看出，PyTorch 十分注重占用内存（或显存）大小，没有用的空间释
    放很及时，可以很有效地利用有限的内存。
"""

torch.autograd.grad()
loss.backward()