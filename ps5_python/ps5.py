#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 5 13:40:08 2020

@author: sanskar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
# User Modules
import lk
import pyramid

if __name__ == '__main__':
    # Problem 1
    shift0 = cv2.imread('input/TestSeq/Shift0.png', 0)
    shiftr2 = cv2.imread('input/TestSeq/ShiftR2.png', 0)
    shiftr5u5 = cv2.imread('input/TestSeq/ShiftR5U5.png', 0)
    shiftr10 = cv2.imread('input/TestSeq/ShiftR10.png', 0)
    shiftr20 = cv2.imread('input/TestSeq/ShiftR20.png', 0)
    shiftr40 = cv2.imread('input/TestSeq/ShiftR40.png', 0)

    flow1 = lk.lucas_kanade(shift0, shiftr2, w_size=(21, 21))
    flow2 = lk.lucas_kanade(shift0, shiftr5u5, w_size=(21, 21))
    flow3 = lk.lucas_kanade(shift0, shiftr10, w_size=(31, 31))
    flow4 = lk.lucas_kanade(shift0, shiftr20, w_size=(31, 31))
    flow5 = lk.lucas_kanade(shift0, shiftr40, w_size=(31, 31))

    lk.draw_flow_quiver(flow1, shift0, 'output/ps5-1-a-1.png')
    lk.draw_flow_quiver(flow2, shift0, 'output/ps5-1-a-2.png')
    lk.draw_flow_quiver(flow3, shift0, 'output/ps5-1-b-1.png')
    lk.draw_flow_quiver(flow4, shift0, 'output/ps5-1-b-2.png')
    lk.draw_flow_quiver(flow5, shift0, 'output/ps5-1-b-3.png')

    # Problem 2
    seq_img1 = cv2.imread('input/DataSeq1/yos_img_01.jpg', 0)
    seq_img1_lev1 = pyramid.reduce(seq_img1)
    seq_img1_lev2 = pyramid.reduce(seq_img1_lev1)
    seq_img1_lev3 = pyramid.reduce(seq_img1_lev2)
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Gaussian Pyramids')
    ax[0, 0].imshow(seq_img1, cmap='gray', interpolation='bicubic')
    ax[0, 1].imshow(seq_img1_lev1, cmap='gray', interpolation='bicubic')
    ax[1, 0].imshow(seq_img1_lev2, cmap='gray', interpolation='bicubic')
    ax[1, 1].imshow(seq_img1_lev3, cmap='gray', interpolation='bicubic')
    fig.tight_layout()
    fig.savefig('output/ps5-2-a-1.png')

    seq_img1_levm1 = pyramid.expand(seq_img1_lev1)
    seq_img1_levm2 = pyramid.expand(seq_img1_lev2)
    seq_img1_levm3 = pyramid.expand(seq_img1_lev3)
    lap_lev1 = pyramid.find_diff(seq_img1.astype(np.float32), seq_img1_levm1)
    lap_lev2 = pyramid.find_diff(seq_img1_lev1, seq_img1_levm2)
    lap_lev3 = pyramid.find_diff(seq_img1_lev2, seq_img1_levm3)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Laplacian Pyramids')
    ax[0, 0].imshow(lap_lev1, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
    ax[0, 1].imshow(lap_lev2, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
    ax[1, 0].imshow(lap_lev3, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
    ax[1, 1].imshow(seq_img1_lev3, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
    fig.tight_layout()
    fig.savefig('output/ps5-2-b-1.png')

    # Problem 3
