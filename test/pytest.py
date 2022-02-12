#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/2/8 10:04
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : pytest.py
# @Software: PyCharm

import os
import sys


# class Person:
#     def __init__(self, age, name):
#         self.__age = age
#         self.__name = name


if __name__ == '__main__':
    cache_dir = os.path.join('..', 'data.csv')

    print(os.path.abspath(cache_dir))
    print(os.path.abspath(os.path.dirname(cache_dir)))
    print(os.path.splitext(os.path.abspath(cache_dir)))
    # p = Person(10,"sean")
    a  = (1,1)+(4,5)
    print(a)
    device = None
    if not device:
        print("hahah")

