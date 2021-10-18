# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/8/2021 10:12 AM
# @File:cvkmean.py
# !/usr/bin/env python
import os

import cv2
import numpy as np
import copy

import torch
from skimage import morphology

from test_transforms import draw, mask_vis
from tqdm import tqdm


def mask_replace_num(mask, originNum, newNum):
    masktmp = copy.deepcopy(mask)
    if originNum in mask:
        masktmp[np.where(mask == originNum)] = newNum
    return masktmp


class Segment:
    def __init__(self, segments, bestLabels=None):
        # define number of segments, with default 5
        self.segments = segments
        self.bestLabels = bestLabels

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        reshaped_labels = self.bestLabels.reshape(-1, 1)
        reshaped_labels = reshaped_labels.astype(np.int32)
        ret, label, center = cv2.kmeans(vectorized, self.segments, reshaped_labels, criteria, 10,
                                        cv2.KMEANS_USE_INITIAL_LABELS)
        # ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # data, K, bestLabels, criteria, attempts, flags, centers=None
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        return component


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return torch.from_numpy(categorical)


if __name__ == "__main__":
    import argparse
    import sys

    source_dir = r"C:\Users\zengxh\Documents\workspace\PyCharm-workspace\tianchi\tianchi-logic-object\data\suichang_round1_test_partB_210120"
    mask_dir = r"C:\Users\zengxh\Desktop\results"
    kmeans_dir = r"C:\Users\zengxh\Desktop\results-kmean"

    kernel = np.ones((3, 3), np.uint8)

    for mask_name in tqdm(os.listdir(mask_dir)):
        image = cv2.imread(os.path.join(source_dir, mask_name.replace(".png", ".tif")))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        maskset = set(mask.flatten().tolist())
        kmeans_img = os.path.join(kmeans_dir, mask_name)
        if not os.path.exists(kmeans_img):
            maskkmeans = copy.deepcopy(mask)
            for index, classnum in enumerate(maskset):
                maskkmeans = mask_replace_num(maskkmeans, classnum, index)

            seg = Segment(len(maskset), maskkmeans)
            label, result = seg.kmeans(image)
            cv2.imwrite(kmeans_img, label)
        else:
            label = cv2.imread(kmeans_img, cv2.IMREAD_GRAYSCALE)

        masktmp = copy.deepcopy(mask)
        for classnum in maskset:
            if classnum in [3, 4, 8, 10]:
                masktmp2 = copy.deepcopy(mask)
                # masktmp2[np.where(masktmp2 == classnum)] = 255
                masktmp2[np.where(masktmp2 != classnum)] = 0
                # cv2.imshow('masktmp2', masktmp2)
                dilate = cv2.dilate(masktmp2, kernel, iterations=1)
                dilatepos = np.where(dilate == classnum)
                # cv2.imshow('dilate', dilate)

                maskpos = np.where(mask == classnum)
                counts = np.bincount(label[maskpos])
                label_classnum = np.argmax(counts)  # 众数为classnum对应label中的分类

                labelttmp = copy.deepcopy(label)
                label_classnum_pos = np.where(labelttmp == label_classnum)  # labelclassnum的位置
                labelttmp[label_classnum_pos] = classnum

                labelttmp[dilatepos] = labelttmp[dilatepos] + 11

                label_classnum_pos2 = np.where(labelttmp == (classnum + 11))

                masktmp[label_classnum_pos2] = classnum

        cv2.imwrite(os.path.join(r"C:\Users\zengxh\Desktop\results", mask_name), masktmp)
