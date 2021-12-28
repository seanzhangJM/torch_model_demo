#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:04
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : run_task.py
# @Software: PyCharm

import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torch_model_demo.util.log_config import get_logger
from torch import nn
from d2l import torch as d2l
from torch_model_demo.config.ttparas.traintestconfigs import TRAIN_CONFIG
from torch_model_demo.util.train_and_test import train

logger = get_logger()


def mnist_demo():
    BATCH_SIZE = TRAIN_CONFIG['BATCH_SIZE']
    LEARNING_RATE = TRAIN_CONFIG['LEARNING_RATE']
    NUM_EPOCHS = TRAIN_CONFIG['NUM_EPOCHS']
    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)
    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    train(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer)


if __name__ == '__main__':
    pass
