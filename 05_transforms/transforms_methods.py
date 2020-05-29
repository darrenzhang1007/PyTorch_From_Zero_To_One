# -*- coding: utf-8 -*-
"""
# @file name  : transforms_method.py
# @author     : DarrenZhang
# @date       : 2020年5月7日09:53:53
# @brief      : transforms方法二
"""
import os
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.my_dataset import RMBDataset
from tools.common_tools import transform_invert, set_seed


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

    # 1 Pad
    # 05_transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # 05_transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # 05_transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # 05_transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter
    transforms.ColorJitter(brightness=0.5),
    # 05_transforms.ColorJitter(contrast=0.5),
    # 05_transforms.ColorJitter(saturation=0.5),
    # 05_transforms.ColorJitter(hue=0.3),

    # 3 Grayscale
    # 05_transforms.Grayscale(num_output_channels=3),  # 概率值为1的GrayScale
    transforms.RandomGrayscale(num_output_channels=3, p=0.1),

    # 4 Affine
    # 05_transforms.RandomAffine(degrees=30),
    # 05_transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),  # 平移
    # 05_transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # 05_transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # 05_transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing 随机遮挡。接受的是tensor，需要先转化成tensor后面转化为tensor的代码要注释掉
    # 05_transforms.ToTensor(),
    # 05_transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # 05_transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),  # 随机填充色彩

    # 1 RandomChoice
    # 05_transforms.RandomChoice([05_transforms.RandomVerticalFlip(p=1), 05_transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # 05_transforms.RandomApply([05_transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         05_transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    # 05_transforms.RandomOrder([05_transforms.RandomRotation(15),
    #                         05_transforms.Pad(padding=32),
    #                         05_transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    transforms.ToTensor(),
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
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()





