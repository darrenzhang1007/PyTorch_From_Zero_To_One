# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 9:47
# @Author  : DarrenZhang
# @FileName: nn_conv_layers.py
# @Software: PyCharm
# @Blog    ：https://www.yuque.com/darrenzhang
# @Brief   : 学习卷积层
"""
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed

set_seed(3)  # 设置随机种子

# ================================= load img ==================================
# print(os.path.dirname(os.path.abspath(__file__)))  # 当前文件路径
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to 02_tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W 加上batch维度

# ================================= create convolution layer ==================================

# ================ 2d
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================ transposed
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)


# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img1 = img_tensor[0, 0:2, ...]
print(img1.shape)
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()