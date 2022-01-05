#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:04
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : run_task.py
# @Software: PyCharm

import torch
import os

import torchvision
from d2l.torch import get_dataloader_workers
from torch.utils import data
from torch_model_demo.util.log_config import get_logger
from torch import nn
from d2l import torch as d2l
from torch_model_demo.config.ttparas.traintestconfigs import TRAIN_CONFIG
from torch_model_demo.util.train_and_test import train, try_gpu
from torch_model_demo.model.cnnNet import LeNet, GoogleNet
from torchvision import transforms

logger = get_logger()


def train_mnist_demo():
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
    train(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer, try_gpu())


def train_fashion_demo():
    def load_data_fashion_mnist(batch_size, resize=None):
        """Download the Fashion-MNIST dataset and then load it into memory.

        Defined in :numref:`sec_fashion_mnist`"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./torch_model_demo/data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="./torch_model_demo/data", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))

    BATCH_SIZE = TRAIN_CONFIG['BATCH_SIZE']
    LEARNING_RATE = TRAIN_CONFIG['LEARNING_RATE']
    NUM_EPOCHS = TRAIN_CONFIG['NUM_EPOCHS']
    train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)
    train(LeNet, train_iter, test_iter, NUM_EPOCHS, LEARNING_RATE, try_gpu())


def train_fashion_onGoogleNet():
    def load_data_fashion_mnist(batch_size, resize=None):
        """Download the Fashion-MNIST dataset and then load it into memory.

        Defined in :numref:`sec_fashion_mnist`"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./torch_model_demo/data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="./torch_model_demo/data", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train(GoogleNet, train_iter, test_iter, num_epochs, lr, try_gpu())


if __name__ == '__main__':
    train_fashion_demo()
