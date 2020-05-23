# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : DarrenZhang
# @date       : 2020年5月10日11:07:07
# @brief      : 通用函数
"""
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image


def transform_invert(img_, transform_train):
    """
    将输入的tensor数据进行反transfrom操作，对输入的数据进行可视化展示
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])  # 乘方差 加均值

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        # img_ = img_.detach()
        img_ = np.array(img_) * 255

    # 将numpy_array转化为PIL
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)







