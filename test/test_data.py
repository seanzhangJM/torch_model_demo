#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/2/8 9:46
# @Author  : zhangjianming
# @Email   : YYDSPanda@163.com
# @File    : test_data.py
# @Software: PyCharm

import random
import tarfile
import zipfile

import torch
from torch.utils import data
from d2l import torch as d2l
import requests
import os
import hashlib

# DATA_HUB存放的是一个字典，里面保存的key:数据文件名,内容：tuple2(下载地址,校验sha-1密匙)
DATA_HUB = dict()
# 这个是下载地址，可以根据实际网址配，这里给出的是d2l的下载地址
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 生成样本 y=Xw+b+ϵ，其中epsilon是高斯随机误差
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 数据遍历方式的参考，备注这种方式需要把数据全部加载到内存，并不高效，建议使用torch的数据加载方式
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# torch加载数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    # url 下载地址,sha1密匙
    url, sha1_hash = DATA_HUB[name]
    # 缓存目录不存在就创建，存在就忽略，备注:..在pycharm和linux下不一样
    # ex:.. 在torch_model_demo这个项目的test package下，在pycharm里面../data只的就是torch_model_demo/data这个相对路径
    # 而在linux系统中，现在假设我的项目torch_model_demo放在/root/process下面，cd 到/root/process/torch_model_demol 下，把/root/process/torch_model_demol加入 sys.path.extend(["."])系统路径，
    # 这时候../data不管在torch_model_demo的哪个package下面，../data，指的都是/root/process/torch_model_demol的上级目录下的data目录，即/root/process/data/这个目录
    # 结论，在具体的运行环境里面，如果是win pycharm里面就用.. ，linux cd到项目目录，就用
    os.makedirs(cache_dir, exist_ok=True)
    # 路径+文件名，
    fname = os.path.join(cache_dir, url.split('/')[-1])
    # 接下来进行sha1摘要函数验证
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存，返回文件名
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    #获得文件所在目录
    base_dir = os.path.dirname(fname)
    #分离文件扩展名
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)
