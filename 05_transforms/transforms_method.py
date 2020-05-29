# -*- coding: utf-8 -*-
"""
# @file name  : transforms_method.py
# @author     : DarrenZhang
# @date       : 2020年5月7日08:56:07
# @brief      : transforms方法(一)
"""
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.my_dataset import RMBDataset
from matplotlib import pyplot as plt
from tools.common_tools import set_seed, transform_invert


set_seed(seed=1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


# ============================ step 1/5 数据 ============================
train_dir = "H:/PyTorch_From_Zero_To_One/data/rmb_split/train"
valid_dir = "H:/PyTorch_From_Zero_To_One/data/rmb_split/valid"
print(train_dir)

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    transforms.CenterCrop(196),     # 512会进行填充0像素

    # 2 RandomCrop
    # 05_transforms.RandomCrop(224, padding=16),
    # 05_transforms.RandomCrop(224, padding=(16, 64)),
    # 05_transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # 05_transforms.RandomCrop(512, pad_if_needed=True),  # 当size大于图片尺寸的时候，设置pad_if_needed=True
    # 05_transforms.RandomCrop(224, padding=64, padding_mode='edge'),  # 边缘像素填充
    # 05_transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # 05_transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # 05_transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # 05_transforms.FiveCrop(112),  # 得到的是tuple类型
    # 05_transforms.Lambda(lambda crops: torch.stack([(05_transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # 05_transforms.TenCrop(112, vertical_flip=False),
    # 05_transforms.Lambda(lambda crops: torch.stack([(05_transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # 05_transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # 05_transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # 05_transforms.RandomRotation(90),  # [-90, 90]
    # 05_transforms.RandomRotation((90), expand=True),
    # 05_transforms.RandomRotation(30, center=(0, 0)),
    # 05_transforms.RandomRotation(30, center=(0, 0), expand=True),  # expand only for center rotation，旋转丢失的信息无法找回

    transforms.ToTensor(),  # FiveCrop返回的是tensor 就不需要totensor了
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.savefig("result.png")
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # FiveCrop 得到的是5维的tensor，可视化使用下面代码
        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)






