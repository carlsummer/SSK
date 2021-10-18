# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/23/2021 12:42 PM
# @File:mycode
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
# pip install scikit-image
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
import copy
import os
# mmseg/core/evaluation/class_names.py 0-18 19个分类
from mmseg.core.evaluation.class_names import cityscapes_palette


def readimg(filepath):
    # filepath = data["img_metas"][0].data[0][0]["filename"]
    originimg = cv2.imread(filepath)
    cv2.imwrite('sdf.jpg', originimg)


def remove_small_objects_and_holes(label, min_size, area_threshold):
    # kernel = cv.getStructuringElement(cv.MORPH_RECT,(500,500))
    # label = cv.dilate(label,kernel)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
    # label = cv.erode(label,kernel)
    label = remove_small_objects0(label, min_size)
    label = remove_small_holes0(label, area_threshold)
    return label


def remove_small_objects0(img, min_size=50):
    return remove_small_objects(img, min_size=min_size, connectivity=1, in_place=True)  # 移除小的区域


def remove_small_holes0(img, area_threshold=50):
    return remove_small_holes(img, area_threshold=area_threshold, connectivity=1, in_place=True)  # 移除小的孔洞

def to_categorical(y, num_classes=None, dtype='int32'):
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
    return categorical

def mask_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    color_array = np.array(cityscapes_palette(),  # bareland
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
        overlapping = cv2.addWeighted(img, alpha, anno_vis, 1-alpha, 0)
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

class Segment:
    def __init__(self, segments,bestLabels=None):
        # define number of segments, with default 5
        self.segments = segments
        self.bestLabels = bestLabels

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        reshaped_labels = self.bestLabels.reshape(-1,1)
        reshaped_labels=reshaped_labels.astype(np.int32)
        ret, label, center = cv2.kmeans(vectorized, self.segments, reshaped_labels, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS)
        # ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                                        #data, K, bestLabels, criteria, attempts, flags, centers=None
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        return component

kmeans_dir = "/home/zengxh/datasets/results-kmean"

def kmmoursfunc(mask,filepath,kernel=np.ones((12, 24), np.uint8),classlist=[4,5,13,6],lenmaskset = 0):
    mask = mask.astype('uint8')
    mask = mask + 1
    image = cv2.imread(filepath)

    # :return: 0.jpg /home/deploy/datasets/creepageDistance/lr/test .jpg 0
    path_obj = Path(filepath)
    mask_name, filedir, filesuffix, filenamestem = path_obj.name,str(path_obj.parent),path_obj.suffix,path_obj.stem

    maskset = set(mask.flatten().tolist())
    kmeans_img = os.path.join(kmeans_dir, mask_name)
    if not os.path.exists(kmeans_img):
        maskkmeans = copy.deepcopy(mask)
        for index, classnum in enumerate(maskset):
            maskkmeans = mask_replace_num(maskkmeans, classnum, index)

        seg = Segment(len(maskset) if lenmaskset == 0 else lenmaskset , maskkmeans)
        label, result = seg.kmeans(image)
        cv2.imwrite(kmeans_img, label)
    else:
        label = cv2.imread(kmeans_img, cv2.IMREAD_GRAYSCALE)

    masktmp = copy.deepcopy(mask)
    for classnum in maskset:
        if classnum in classlist:
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

            labelttmp[dilatepos] = labelttmp[dilatepos] + 20

            label_classnum_pos2 = np.where(labelttmp == (classnum + 20))

            masktmp[label_classnum_pos2] = classnum
    return masktmp - 1

def kmmours(mask,filepath):
    # image = cv2.imread(filepath)
    # draw(image,mask)
    #
    # gtlabel = cv2.imread("data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png", 0)
    # draw(image, gtlabel)

    kernel = np.ones((3, 3), np.uint8)
    classlist = [4,5,10,15,17,18]
    lenmaskset = 0
    result = kmmoursfunc(mask,filepath,kernel,classlist,lenmaskset)
    # draw(image,result)

    for _ in range(10):
        predict_list = to_categorical(mask + 1, num_classes=20) + to_categorical(result + 1, num_classes=20)
        mask = np.argmax(predict_list, axis=-1)  # 256*256
        mask = mask - 1

        result = kmmoursfunc(mask, filepath, kernel, classlist, lenmaskset)
    return result