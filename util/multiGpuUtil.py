#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/2/12 12:53
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : multiGpuUtil.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config.ttparas.traintestconfigs import TRAIN_CONFIG
from d2l.torch import get_dataloader_workers
from torch.utils import data
from torchvision import transforms
from util.bean import Timer, Animator, Accumulator
from util.train_and_test import try_gpu, accuracy


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


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_onMultiGpu(net, num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)
    devices = [try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices) #这边设置*
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = Timer(), 10
    animator = Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0]) #导致这边自动分配数据多多个GPU上，torch框架自动实现了的*
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
