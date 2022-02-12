#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/2/8 14:57
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : selfFefinedLayerWithPara.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    r"""
    定义具有参数的层， 这些参数可以通过训练进行调整。 我们可以使用内置函数来创建参数，这些函数提供一些基本的管理功能。
    比如管理访问、初始化、共享、保存和加载模型参数。 这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。
    """

    # nn.Linear(100,10),这个Module需要100*10的weight和10*1的bias，bias可以忽略，如果我想对这一层的参数做特殊处理，譬如初始化，保存，我要从dict中去找，在module里事先定义，可以省去很多麻烦
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
