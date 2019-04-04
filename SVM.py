# -*- coding:utf-8 -*-
import numpy as np
"""
@project: PyCharmProject
@author: KunJ
@file: SVM.py
@ide: Pycharm
@time: 2019-04-04 19:53:23
@month: 四月
"""
import csv
from sklearn import svm


def dataSet(filename):
    """获取西瓜3.0alpha数据集"""
    X = []
    Y = []
    with open(filename) as f:
        reader = csv.reader(f)  # 返回一个reader的迭代器
        head_row = next(reader)  # next获取标题行
        print(head_row)
        print(reader)
        for line in reader:
            print(line)
            X.append(line[7:9])  # 密度和含糖率
            Y.append(line[10])  # 类别标签
    return X, Y


def linearKernel(X, Y):
    """
    线性核函数
    :param X:数据集
    :param Y:标签
    :return:
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    print("线性核函数的支持向量为：\n", clf.support_vectors_)
    print("每个类别的支持向量的个数：\n", clf.n_support_)


def gassiankernel(X, Y):
    """
    高斯核函数
    :param X:数据集
    :param Y:标签
    :return:
    """
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, Y)
    print("高斯核函数的支持向量为：\n", clf.support_vectors_)
    print("每个类别的支持向量的个数：\n", clf.n_support_)

if __name__ == '__main__':
    filename = 'C:\\Users\\Kun_J\\Desktop\\西瓜3.0.csv'
    X, Y = dataSet(filename)
    linearKernel(X, Y)
    gassiankernel(X, Y)



