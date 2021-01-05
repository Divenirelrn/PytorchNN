from torchvision import transforms


#一般随机裁剪 + 旋转用的比较多
#torch无加噪声接口
#Data augmentation helps, but not much(特征都是一类，方差小)
transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.Scale([32,32]),
    transforms.RandomCrop([28,28]),
    transforms.RandomHorizontalFlip(), #随机水平翻转
    transforms.RandomVerticalFlip(), #随机垂直翻转
    transforms.RandomRotation(15), #随机旋转(-15度 < x < 15度)
    transforms.RandomRotation([0, 90, 180, 270]) #随机旋转
])