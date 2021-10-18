# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/8/2021 10:05 AM
# @File:showimg
'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description :
LastEditTime: 2020-11-27 03:42:46
'''
import copy
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image


def mask_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    color_array = np.array([[0, 0, 0],  # other
                            [177, 191, 122],  # farm_land
                            [0, 128, 0],  # forest
                            [128, 168, 93],  # grass
                            [62, 51, 0],  # road
                            [128, 128, 0],  # urban_area
                            [128, 128, 128],  # countryside
                            [192, 128, 0],  # industrial_land
                            [0, 128, 128],  # construction
                            [132, 200, 173],  # water
                            [128, 64, 0]],  # bareland
                           dtype='uint8')
    anno_vis = np.zeros([label.shape[0],label.shape[1], 3], dtype='uint8')
    for npmax_index in range(np.max(label)):
        # masktmp = copy.deepcopy(mask)
        if npmax_index in label:
            anno_vis[np.where(label == npmax_index)]=color_array[npmax_index]

    # r = np.where(label == 1, 255, 0)
    # g = np.where(label == 2, 255, 0)
    # b = np.where(label == 3, 255, 0)
    # anno_vis2 = np.dstack((b, g, r)).astype(np.uint8)
    # # 黄色分量(红255, 绿255, 蓝0)
    # for npmax_index in range(np.max(label) - 3):
    #     anno_vis2[:, :, 0] = anno_vis2[:, :, 0] + np.where(label == 3 + 1 + npmax_index, 255, 0)
    #     anno_vis2[:, :, 1] = anno_vis2[:, :, 1] + np.where(label == 3 + 1 + npmax_index, 255, 0)
    #     anno_vis2[:, :, 2] = anno_vis2[:, :, 2] + np.where(label == 3 + 1 + npmax_index, 255, 0)

    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1-alpha, 0)
        return overlapping


def draw(image, mask):
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask_vis(mask, image))
    plt.show()

def mask_replace_num(mask,originNum,newNum):
    masktmp = copy.deepcopy(mask)
    if originNum in mask:
        masktmp[np.where(mask == originNum)] = newNum
    return masktmp


if __name__ == '__main__':
    filename="000013"
    image = cv.imread(os.path.join(r"/tcdata/suichang_round1_test_partB_210120/",filename + ".tif"))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask = cv.imread(os.path.join(r"/user_data/tmp_data/testcresults/",filename + ".png"), cv.IMREAD_GRAYSCALE)
    draw(image, mask)

    # mask = cv.imread(os.path.join(r"C:\Users\zengxh\Documents\workspace\PyCharm-workspace\tianchi\tianchi-logic-object\data\model1-4class",filename + ".png"), cv.IMREAD_GRAYSCALE)
    # draw(image, mask)
