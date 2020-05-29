# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/11 11:42
# @Author  : DarrenZhang
# @FileName: 02_tensorboard_methods.py
# @Software: PyCharm
# @Blog    ：https://www.yuque.com/darrenzhang
# @Brief   : tensorboard方法使用
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tools.my_dataset import RMBDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tools.common_tools import set_seed
from model.lenet import LeNet


set_seed(1)  # 设置随机种子

# ----------------------------------- 0 SummaryWriter -----------------------------------
flag = 0
# flag = 1
if flag:
    log_dir = "./train_log/test_log_dir"
    writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")
    # writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")

    for x in range(100):
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.close()


# ----------------------------------- 1 scalar and scalars -----------------------------------
flag = 0
# flag = 1
if flag:

    max_epoch = 100

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(max_epoch):

        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                 "xcosx": x * np.cos(x)}, x)

    writer.close()

# ----------------------------------- 2 histogram -----------------------------------
flag = 0
# flag = 1
if flag:

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(2):

        np.random.seed(x)

        data_union = np.arange(100)
        data_normal = np.random.normal(size=1000)

        writer.add_histogram('distribution union', data_union, x)
        writer.add_histogram('distribution normal', data_normal, x)

        plt.subplot(121).hist(data_union, label="union")
        plt.subplot(122).hist(data_normal, label="normal")
        plt.legend()
        plt.show()

    writer.close()


# ----------------------------------- 3 image -----------------------------------
flag = 0
# flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # img 1     random
    fake_img = torch.randn(3, 512, 512)
    writer.add_image("fake_img", fake_img, 1)
    time.sleep(1)

    # img 2     ones
    fake_img = torch.ones(3, 512, 512)  # 255
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 2)

    # img 3     1.1
    fake_img = torch.ones(3, 512, 512) * 1.1
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 3)

    # img 4     HW
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")

    # img 5     HWC
    fake_img = torch.rand(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")

    writer.close()


# ----------------------------------- 4 make_grid -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    train_dir = "H:/PyTorch_From_Zero_To_One/data/rmb_split/train"

    transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
    train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    data_batch, label_batch = next(iter(train_loader))  # BCHW

    img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
    # img_grid = vutils.make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
    writer.add_image("input img", img_grid, 0)

    writer.close()


# ----------------------------------- 5 add_graph -----------------------------------
## 1.2版本对于此方法有Bug
flag = 0
# flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # 模型
    fake_img = torch.randn(1, 3, 32, 32)

    lenet = LeNet(classes=2)

    writer.add_graph(lenet, fake_img)

    writer.close()

    from torchsummary import summary
    print(summary(lenet, (3, 32, 32), device="cpu"))