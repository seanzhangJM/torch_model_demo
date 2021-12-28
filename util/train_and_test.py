#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 11:25
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : train_and_test.py
# @Software: PyCharm

import torch
from torch import nn
from torch.utils import data
from torch_model_demo.util.bean import Accumulator, Animator
from torch_model_demo.util.log_config import get_logger

logger = get_logger()


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    logger.info("验证{}数据集准确率成功".format(str(data_iter)))
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    :param net: model
    :param train_iter: data iterator
    :param loss: user defined loss or comming from the pytorch trainig framework
    :param updater: learning function,it could be the user-defined or comming from the pytoch built-in frame work
    :return:
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            #crossentropy 默认是mean的话就要乘以样本数目进行恢复
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    :param net:
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epochs:
    :param updater:
    :return:
    """
    global train_metrics, test_acc
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        logger.info("第{}轮：训练完成".format(str(epoch)))
        test_acc = evaluate_accuracy(net, test_iter)
        logger.info(
            "train loss:" + str(train_metrics[0]) + "train acc:" + str(train_metrics[1]) + "test acc:" + str(test_acc))
        # animator要在jupyter里面才有用
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 根据不同任务这里灵活调节
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc
