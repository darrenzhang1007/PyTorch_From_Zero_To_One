# -*- coding: utf-8 -*-
# @Time    : 2020/5/1 17:14
# @Author  : DarrenZhang
# @FileName: computational_graph.py
# @Software: PyCharm
# @Blog    ：https://www.yuque.com/darrenzhang
# @Brief   : computational graph

import torch

w = torch.tensor(1, dtype=float, requires_grad=True)
x = torch.tensor(2, dtype=float, requires_grad=True)

a = torch.add(w, x)
a.retain_grad()  # 在反向传播之后，仍然保留非叶子节点的梯度
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)  # 反向传播之后，会将非叶子节点释放掉，节省内存开销

# 查看 grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)  # 记录创建该张量时所用的方法

