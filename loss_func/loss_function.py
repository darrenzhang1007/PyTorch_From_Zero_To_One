# -*- coding: utf-8 -*-
# @Time    : 2020年5月10日15:13:35
# @Author  : DarrenZhang
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://www.yuque.com/darrenzhang
# @brief   : 1. nn.CrossEntropyLoss
#            2. nn.NLLLoss
#            3. BCELoss
#            4. BCEWithLogitsLoss
#            5. nn.L1Loss
#            6. nn.MSELoss
#            7. nn.SmoothL1Loss
#            8. nn.PoissonNLLLoss
#            9. nn.KLDivLoss
#            10. nn.MarginRankingLoss
#            11. nn.MultiLabelMarginLoss
#            12. nn.SoftMarginLoss
#            13. nn.MultiLabelSoftMarginLoss
#            14. nn.MultiMarginLoss
#            15. nn.TripletMarginLoss
#            16. nn.HingeEmbeddingLoss
#            17. nn.CosineEmbeddingLoss
#            18. nn.CTCLoss
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- CrossEntropy loss: reduction -----------------------------------
flag = 0
# flag = 1
if flag:
    # def loss function
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("Cross Entropy Loss:\n ", loss_none, loss_sum, loss_mean)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    idx = 0

    input_1 = inputs.detach().numpy()[idx]      # [1, 2]
    target_1 = target.numpy()[idx]              # [0]

    # 第一项
    x_class = input_1[target_1]

    # 第二项
    sigma_exp_x = np.sum(list(map(np.exp, input_1)))
    log_sigma_exp_x = np.log(sigma_exp_x)

    # 输出loss
    loss_1 = -x_class + log_sigma_exp_x

    print("第一个样本loss为: ", loss_1)


# ----------------------------------- weight -----------------------------------
# flag = 0
flag = 1
if flag:
    # def loss function
    weights = torch.tensor([1, 2], dtype=torch.float)
    # weights = torch.tensor([0.7, 0.3], dtype=torch.float)

    loss_f_none_w = nn.CrossEntropyLoss(weight=weights, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)


# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:
    weights = torch.tensor([1, 2], dtype=torch.float)
    weights_all = np.sum(list(map(lambda x: weights.numpy()[x], target.numpy())))  # [0, 1, 1]  # [1 2 2]

    mean = 0
    loss_sep = loss_none.detach().numpy()
    for i in range(target.shape[0]):

        x_class = target.numpy()[i]
        tmp = loss_sep[i] * (weights.numpy()[x_class] / weights_all)
        mean += tmp

    print(mean)


# ----------------------------------- 2 NLLLoss -----------------------------------
flag = 0
# flag = 1
if flag:

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.NLLLoss(weight=weights, reduction='none')
    loss_f_sum = nn.NLLLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.NLLLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print("NLL Loss", loss_none_w, loss_sum, loss_mean)


# ----------------------------------- 3 BCE Loss -----------------------------------
flag = 0
# flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCELoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCELoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCELoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print("BCE Loss", loss_none_w, loss_sum, loss_mean)


# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    idx = 0

    x_i = inputs.detach().numpy()[idx, idx]
    y_i = target.numpy()[idx, idx]              #

    # loss
    # l_i = -[ y_i * np.log(x_i) + (1-y_i) * np.log(1-y_i) ]      # np.log(0) = nan
    l_i = -y_i * np.log(x_i) if y_i else -(1-y_i) * np.log(1-x_i)

    # 输出loss
    print("BCE inputs: ", inputs)
    print("第一个loss为: ", l_i)


# ----------------------------------- 4 BCE with Logis Loss -----------------------------------
flag = 0
# flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)


# --------------------------------- pos weight

flag = 0
# flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1], dtype=torch.float)
    pos_w = torch.tensor([3], dtype=torch.float)        # 3

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none', pos_weight=pos_w)
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum', pos_weight=pos_w)
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean', pos_weight=pos_w)

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\npos_weights: ", pos_w)
    print(loss_none_w, loss_sum, loss_mean)


# ------------------------------------------------- 5 L1 loss ----------------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.ones((2, 2))
    target = torch.ones((2, 2)) * 3

    loss_f = nn.L1Loss(reduction='none')
    loss = loss_f(inputs, target)

    print("input:{}\ntarget:{}\nL1 loss:{}".format(inputs, target, loss))

# ------------------------------------------------- 6 MSE loss ----------------------------------------------

    loss_f_mse = nn.MSELoss(reduction='none')
    loss_mse = loss_f_mse(inputs, target)

    print("MSE loss:{}".format(loss_mse))

# ------------------------------------------------- 7 Smooth L1 loss ----------------------------------------------
flag = 0
# flag = 1
if flag:
    inputs = torch.linspace(-3, 3, steps=500)
    target = torch.zeros_like(inputs)

    loss_f = nn.SmoothL1Loss(reduction='none')

    loss_smooth = loss_f(inputs, target)

    loss_l1 = np.abs(inputs.numpy())

    plt.plot(inputs.numpy(), loss_smooth.numpy(), label='Smooth L1 Loss')
    plt.plot(inputs.numpy(), loss_l1, label='L1 loss')
    plt.xlabel('x_i - y_i')
    plt.ylabel('loss value')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------------------------- 8 Poisson NLL Loss ----------------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.randn((2, 2))
    target = torch.randn((2, 2))

    loss_f = nn.PoissonNLLLoss(log_input=True, full=False, reduction='none')
    loss = loss_f(inputs, target)
    print("input:{}\ntarget:{}\nPoisson NLL loss:{}".format(inputs, target, loss))

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    idx = 0

    loss_1 = torch.exp(inputs[idx, idx]) - target[idx, idx]*inputs[idx, idx]

    print("第一个元素loss:", loss_1)


# ------------------------------------------------- 9 KL Divergence Loss ----------------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    inputs_log = torch.log(inputs)
    target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2]], dtype=torch.float)

    loss_f_none = nn.KLDivLoss(reduction='none')
    loss_f_mean = nn.KLDivLoss(reduction='mean')
    loss_f_bs_mean = nn.KLDivLoss(reduction='batchmean')

    loss_none = loss_f_none(inputs, target)
    loss_mean = loss_f_mean(inputs, target)
    loss_bs_mean = loss_f_bs_mean(inputs, target)

    print("loss_none:\n{}\nloss_mean:\n{}\nloss_bs_mean:\n{}".format(loss_none, loss_mean, loss_bs_mean))

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    idx = 0

    loss_1 = target[idx, idx] * (torch.log(target[idx, idx]) - inputs[idx, idx])

    print("第一个元素loss:", loss_1)


# ---------------------------------------------- 10 Margin Ranking Loss --------------------------------------------
flag = 0
# flag = 1
if flag:

    x1 = torch.tensor([[1], [2], [3]], dtype=torch.float)
    x2 = torch.tensor([[2], [2], [2]], dtype=torch.float)

    target = torch.tensor([1, 1, -1], dtype=torch.float)

    loss_f_none = nn.MarginRankingLoss(margin=0, reduction='none')

    loss = loss_f_none(x1, x2, target)

    print(loss)

# ---------------------------------------------- 11 Multi Label Margin Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
    y = torch.tensor([[0, 3, -1, -1]], dtype=torch.long)

    loss_f = nn.MultiLabelMarginLoss(reduction='none')

    loss = loss_f(x, y)

    print(loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    x = x[0]
    item_1 = (1-(x[0] - x[1])) + (1 - (x[0] - x[2]))    # [0]
    item_2 = (1-(x[3] - x[1])) + (1 - (x[3] - x[2]))    # [3]

    loss_h = (item_1 + item_2) / x.shape[0]

    print(loss_h)

# ---------------------------------------------- 12 SoftMargin Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
    target = torch.tensor([[-1, 1], [1, -1]], dtype=torch.float)

    loss_f = nn.SoftMarginLoss(reduction='none')

    loss = loss_f(inputs, target)

    print("SoftMargin: ", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    idx = 0

    inputs_i = inputs[idx, idx]
    target_i = target[idx, idx]

    loss_h = np.log(1 + np.exp(-target_i * inputs_i))

    print(loss_h)


# ---------------------------------------------- 13 MultiLabel SoftMargin Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.tensor([[0.3, 0.7, 0.8]])
    target = torch.tensor([[0, 1, 1]], dtype=torch.float)

    loss_f = nn.MultiLabelSoftMarginLoss(reduction='none')

    loss = loss_f(inputs, target)

    print("MultiLabel SoftMargin: ", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    i_0 = torch.log(torch.exp(-inputs[0, 0]) / (1 + torch.exp(-inputs[0, 0])))

    i_1 = torch.log(1 / (1 + torch.exp(-inputs[0, 1])))
    i_2 = torch.log(1 / (1 + torch.exp(-inputs[0, 2])))

    loss_h = (i_0 + i_1 + i_2) / -3

    print(loss_h)

# ---------------------------------------------- 14 Multi Margin Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    x = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3]])
    y = torch.tensor([1, 2], dtype=torch.long)

    loss_f = nn.MultiMarginLoss(reduction='none')

    loss = loss_f(x, y)

    print("Multi Margin Loss: ", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    x = x[0]
    margin = 1

    i_0 = margin - (x[1] - x[0])
    # i_1 = margin - (x[1] - x[1])
    i_2 = margin - (x[1] - x[2])

    loss_h = (i_0 + i_2) / x.shape[0]

    print(loss_h)

# ---------------------------------------------- 15 Triplet Margin Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    anchor = torch.tensor([[1.]])
    pos = torch.tensor([[2.]])
    neg = torch.tensor([[0.5]])

    loss_f = nn.TripletMarginLoss(margin=1.0, p=1)

    loss = loss_f(anchor, pos, neg)

    print("Triplet Margin Loss", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:

    margin = 1
    a, p, n = anchor[0], pos[0], neg[0]

    d_ap = torch.abs(a-p)
    d_an = torch.abs(a-n)

    loss = d_ap - d_an + margin

    print(loss)

# ---------------------------------------------- 16 Hinge Embedding Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    inputs = torch.tensor([[1., 0.8, 0.5]])
    target = torch.tensor([[1, 1, -1]])

    loss_f = nn.HingeEmbeddingLoss(margin=1, reduction='none')

    loss = loss_f(inputs, target)

    print("Hinge Embedding Loss", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:
    margin = 1.
    loss = max(0, margin - inputs.numpy()[0, 2])

    print(loss)


# ---------------------------------------------- 17 Cosine Embedding Loss -----------------------------------------
flag = 0
# flag = 1
if flag:

    x1 = torch.tensor([[0.3, 0.5, 0.7], [0.3, 0.5, 0.7]])
    x2 = torch.tensor([[0.1, 0.3, 0.5], [0.1, 0.3, 0.5]])

    target = torch.tensor([[1, -1]], dtype=torch.float)

    loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='none')

    loss = loss_f(x1, x2, target)

    print("Cosine Embedding Loss", loss)

# --------------------------------- compute by hand
flag = 0
# flag = 1
if flag:
    margin = 0.

    def cosine(a, b):
        numerator = torch.dot(a, b)
        denominator = torch.norm(a, 2) * torch.norm(b, 2)
        return float(numerator/denominator)

    l_1 = 1 - (cosine(x1[0], x2[0]))

    l_2 = max(0, cosine(x1[0], x2[0]))

    print(l_1, l_2)


# ---------------------------------------------- 18 CTC Loss -----------------------------------------
# flag = 0
flag = 1
if flag:
    T = 50      # Input sequence length
    C = 20      # Number of classes (including blank)
    N = 16      # Batch size
    S = 30      # Target sequence length of longest target in batch
    S_min = 10  # Minimum target length, for demonstration purposes

    # Initialize random batch of input vectors, for *size = (T,N,C)
    inputs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(inputs, target, input_lengths, target_lengths)

    print("CTC loss: ", loss)
