# -*- coding:utf-8 -*-
"""
@file name  : tensor_create.py
@author     : DarrenZhang
@date       : 2020年4月30日16:00:02
@brief      : 张量的创建
"""
import torch
import numpy as np
torch.manual_seed(1)

# ============== example 1 ============== #
# create tensor by `torch.tensor`
flag = False
# flag = True
if flag:
    arr = np.ones((3, 3))
    print("ndarray的数据类型：{}".format(arr.dtype))
    t = torch.tensor(arr, device='cpu')
    # t = torch.tensor(arr, device='cuda')  # cuda not gpu
    print(t)


# ============== example 2 ============== #
# create tensor by `torch.from_numpy`
flag = True
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("numpy arr is {}".format(arr))
    print("tensor from numpy t is {}".format(t))

    print("\n修改arr")
    arr[0, 0] = 0
    print("modified numpy arr is {}".format(arr))
    print("modified tensor from numpy t is {}".format(t))

    print("\n修改tensor")
    t[0, 0] = -1
    print("numpy arr is {}".format(arr))
    print("modified tensor from numpy t is {}".format(t))
