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
# create 02_tensor by `torch.02_tensor`
flag = False
# flag = True
if flag:
    arr = np.ones((3, 3))
    print("ndarray的数据类型：{}".format(arr.dtype))
    t = torch.tensor(arr, device='cpu')
    # t = torch.02_tensor(arr, device='cuda')  # cuda not 14_gpu_use
    print(t)

# ============== example 2 ============== #
# create 02_tensor by `torch.from_numpy`
# flag = True
flag = False
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("numpy arr is {}".format(arr))
    print("02_tensor from numpy t is {}".format(t))

    print("\n修改arr")
    arr[0, 0] = 0
    print("modified numpy arr is {}".format(arr))
    print("modified 02_tensor from numpy t is {}".format(t))

    print("\n修改tensor")
    t[0, 0] = -1
    print("numpy arr is {}".format(arr))
    print("modified 02_tensor from numpy t is {}".format(t))

# ============== example 3 ============== #
# create 02_tensor by `torch.zeros`
# flag = True
flag = False
if flag:
    out_t = torch.tensor([1])
    print(out_t)
    t = torch.zeros((3, 3), out=out_t)
    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))  # print memory address

# ============== example 4 ============== #
# create 02_tensor by `torch.zeros_like`
flag = False
# flag = True
if flag:
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out_t = torch.zeros_like(t)
    print("t is {}".format(t))
    print("out t is {}".format(out_t))

# ============== example 5 ============== #
# create 02_tensor by `torch.full`
flag = False
# flag = True
if flag:
    t = torch.full((3, 3), 1)
    print("t is {}".format(t))

# ============== example 6 ============== #
# create 02_tensor by `torch.full_like`
flag = False
# flag = True
if flag:
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out_t = torch.full_like(t, 1)
    print("t is {}".format(t))
    print("out t is {}".format(out_t))

# ============== example 7 ============== #
# create 02_tensor by `torch.arange()`
flag = False
# flag = True
if flag:
    t = torch.arange(2, 10, 2)
    print("t is {}".format(t))

# ============== example 8 ============== #
# create 02_tensor by `torch.linspace()`
# flag = True
flag = False
if flag:
    # t = torch.linspace(2, 10, 5)
    t = torch.linspace(2, 10, 6)
    print("t is {}".format(t))

# ============== example 9 ============== #
# create 02_tensor by `torch.logspace()`
# flag = True
flag = False
if flag:
    t = torch.logspace(2, 10, 6)
    print("t is {}".format(t))

# ============== example 10 ============== #
# create 02_tensor by `torch.eye()`
# flag = True
flag = False
if flag:
    t = torch.eye(4, 5)
    print("t is {}".format(t))

# ============== example 11 ============== #
# create 02_tensor by `torch.normal()`
# flag = True
flag = False
if flag:
    # mean：张量 std: 张量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{} \n std:{}".format(mean, std))
    print(t_normal)

    # mean：标量 std: 标量
    t_normal = torch.normal(0., 1., size=(4,))
    print(t_normal)

    # mean：张量 std: 标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)

    # mean：标量 std: 张量
    std = torch.arange(1, 5, dtype=torch.float)
    mean = 1
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)


# ============== example 12 ============== #
# create 02_tensor by `torch.randn()`
# flag = True
flag = False
if flag:
    t = torch.randn(3)
    print(t)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    out_t = torch.randn_like(t)
    print(out_t)
    t = torch.rand(3)
    print(t)


# ============== example 13 ============== #
# create 02_tensor by `torch.randint()`
flag = True
# flag = False
if flag:
    t = torch.randint(3, 10, (5, 5))
    print(t)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out_t = torch.randint_like(input=t, low=3, high=10)
    print(out_t)


# ============== example 14 ============== #
# create 02_tensor by `torch.randperm()`
# flag = True
flag = False
if flag:
    t = torch.randperm(10)
    print(t)


# ============== example 15 ============== #
# create 02_tensor by `torch.bernoulli()`
# flag = True
flag = False
if flag:
    a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    t = torch.bernoulli(a)
    b = torch.from_numpy(np.array([[0.2, 0.5, 0.7], [0.2, 0.5, 0.7], [0.2, 0.5, 0.7]]))
    print(t)
    print(b)