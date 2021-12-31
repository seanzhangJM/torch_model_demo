#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2021/12/27 14:04
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : run_task.py
# @Software: PyCharm

import sys

sys.path.extend(["."])

from torch_model_demo.task.run_task import train_fashion_demo

if __name__ == '__main__':
    train_fashion_demo()
