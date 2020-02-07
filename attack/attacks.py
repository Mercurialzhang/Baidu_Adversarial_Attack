# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import paddle.fluid as fluid
from scipy import signal
import os
import xml
import glob

from .utils import gkern, linf_img_tenosr

kernel = gkern(7, 4).astype(np.float32)

std = [0.229, 0.224, 0.225]


def MIFGSM(adv_program,
           o, input_layer, step_size=16.0/256,
           epsilon=16.0/256, iteration=20,
           gt_label=0, use_gpu=False, gradients=None, imgname=None):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    adv = o.copy()

    mask = np.zeros([3, 224, 224], dtype=np.float32)

    filepath = glob.glob(os.path.join('./annotation', imgname.split('_')[0] + '*', imgname))[0]
    e = xml.etree.ElementTree.parse(filepath).getroot()
    for objs in e.iter('size'):
        height = int(objs.find('height').text)
        width = int(objs.find('width').text)

    for objs in e.iter('object'):
        x_min = np.floor(int(objs.find('bndbox').find('xmin').text) / width * 224).astype('int32')
        y_min = np.floor(int(objs.find('bndbox').find('ymin').text) / height * 224).astype('int32')
        x_max = np.ceil(int(objs.find('bndbox').find('xmax').text) / width * 224).astype('int32')
        y_max = np.ceil(int(objs.find('bndbox').find('ymax').text) / height * 224).astype('int32')

        mask[:,y_min:(y_max+1),x_min:(x_max+1)] = 1.

    mask_square = np.zeros([3, 224, 224], dtype=np.float32)
    mask_square[:, 36:188, 36:188] = 1.

    mask = mask * mask_square

    gt_label = np.array([gt_label]).astype('int64')
    gt_label = np.expand_dims(gt_label, axis=0)

    grad = np.zeros([3, 224, 224], dtype=np.float32)
    img_std = np.array(std).reshape((3, 1, 1)).astype('float32')
    step = np.array([step_size, step_size, step_size]).reshape((3, 1, 1)).astype('float32')
    step /= img_std
    alpha = step
    img_size = 256

    for _ in range(iteration):
        aug_all = []
        rotate_all = []
        pad_all = []
        random_size_all = []
        flip_all = []
        weight_all = []
        noise_all = []
        for i in range(6):
            aug = np.random.normal(0, 0.05, [1, 2, 3]).astype('float32')
            aug += np.array([1., 0., 0., 0., 1., 0.]).astype('float32').reshape([1, 2, 3])
            rotate_degree = np.random.normal(0, 0.15) / 3.14 * 180
            rotate = cv2.getRotationMatrix2D((0, 0), rotate_degree, 1)
            rotate = np.expand_dims(rotate, 0).astype('float32')

            random_size = np.random.randint(224, img_size)
            pad_top = np.random.randint(0, img_size - random_size)
            pad_bottom = img_size - random_size - pad_top
            pad_left = np.random.randint(0, img_size - random_size)
            pad_right = img_size - random_size - pad_left

            random_size = np.array([random_size]).astype('int32')
            pad = np.array([pad_top, pad_bottom, pad_left, pad_right]).astype('int32')

            aug_all.append(aug)
            random_size_all.append(random_size)
            rotate_all.append(rotate)
            pad_all.append(pad)

            if np.random.uniform(0, 1) > 0.5:
                flip_all.append(np.array([[[ -1.,  0.000000e+00,  0.000000e+00],
                                        [0.000000e+00,  1.,  0.000000e+00]]]).astype(np.float32))
            else:
                flip_all.append(np.array([[[1., 0.000000e+00, 0.000000e+00],
                                           [0.000000e+00, 1., 0.000000e+00]]]).astype(np.float32))

            if np.random.uniform(0, 1) > 0.5:
                weight_all.append(np.ones([1, 1, 121]).astype(np.float32))
            else:
                weight_all.append(np.zeros([1, 1, 121]).astype(np.float32))

            noise_all.append(np.random.uniform(0.8, 1.25, [1, 3]).astype('float32'))

        weight_all = np.concatenate(weight_all, axis=1)
        if np.sum(weight_all) == 0:
            weight_all = np.ones([1, 6, 121]).astype(np.float32)
        aug_all = np.concatenate(aug_all, axis=1)
        rotate_all = np.concatenate(rotate_all, axis=1)
        random_size_all = np.concatenate(random_size_all, axis=0)
        pad_all = np.concatenate(pad_all, axis=0)
        flip_all = np.concatenate(flip_all, axis=1)
        noise_all = np.concatenate(noise_all, axis=1)

        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': gt_label,
                          'weight': weight_all,
                          'size': random_size_all,
                          'pad': pad_all,
                          'aug': aug_all,
                          'rotate': rotate_all,
                          'flip': flip_all,
                          'noise': noise_all})
        g = g[0][0]

        g[0] = signal.convolve2d(g[0], kernel, mode='same')
        g[1] = signal.convolve2d(g[1], kernel, mode='same')
        g[2] = signal.convolve2d(g[2], kernel, mode='same')

        g = g / (np.sqrt(np.sum((g ** 2), axis=0)) + 1e-20)
        g = 0.7 * grad + g * 0.3

        adv = adv + alpha * (g / np.max(np.abs(g))) * mask
        adv = linf_img_tenosr(o, adv, epsilon)
        grad = g

    return adv
