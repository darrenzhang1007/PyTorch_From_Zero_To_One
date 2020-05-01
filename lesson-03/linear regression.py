# -*- coding: utf-8 -*-
# @Time    : 2020/5/1 16:32
# @Author  : DarrenZhang
# @FileName: linear regression.py
# @Software: PyCharm
# @Blog    ：https://www.yuque.com/darrenzhang
# @Brief   : Linear Regression Model
import torch
import matplotlib.pyplot as plt
torch.manual_seed(100)

lr = 0.001  # learning rate

# create training data
x = torch.randn(20, 1) * 10
print(x.shape)
print(x.data)
y = 2*x + (5 + torch.randn(20, 1))  # y = wx + b + noise

# build linear regression model
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

# beginning to iterate
for iteration in range(10000):
    # forward
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # compute MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # backward
    loss.backward()

    # update parameters
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # clean grad to zero
    w.grad.zero_()
    b.grad.zero_()

    # draw the process
    if iteration % 100 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
