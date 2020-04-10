#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:28:15 2020

@author: sanskar
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import stereo

if __name__ == '__main__':
    # Problem 1
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
    D_L = stereo.disparity_ssd(L, R)
    D_R = stereo.disparity_ssd(R, L)

    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(os.path.join('output', 'ps2-1-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps2-1-a-2.png'), D_R)

    # Problem 2
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    D_L = np.abs(stereo.disparity_ssd(L, R, search_range=201))
    D_R = np.abs(stereo.disparity_ssd(R, L, search_range=201))
    
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(os.path.join('output', 'ps2-2-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps2-2-a-2.png'), D_R)

    # Problem 3
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    
    noise = np.zeros(L.shape, dtype=np.float64)
    noise = cv2.randn(noise, 0, 0.05)
    L_n = cv2.add(L, noise)
    L_c = cv2.multiply(L, 1.1)
    
    D_L_n = np.abs(stereo.disparity_ssd(L_n, R, search_range=201))
    D_R_n = np.abs(stereo.disparity_ssd(R, L_n, search_range=201))
    D_L_c = np.abs(stereo.disparity_ssd(L_c, R, search_range=201))
    D_R_c = np.abs(stereo.disparity_ssd(R, L_c, search_range=201))
                   
    D_L_n = cv2.normalize(D_L_n, D_L_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_n = cv2.normalize(D_R_n, D_R_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_L_c = cv2.normalize(D_L_c, D_L_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_c = cv2.normalize(D_R_c, D_R_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(os.path.join('output', 'ps2-3-a-1.png'), D_L_n)
    cv2.imwrite(os.path.join('output', 'ps2-3-a-2.png'), D_R_n)
    cv2.imwrite(os.path.join('output', 'ps2-3-b-1.png'), D_L_c)
    cv2.imwrite(os.path.join('output', 'ps2-3-b-2.png'), D_R_c)
    
    # Problem 4
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
    
    L = L.astype(np.float32)
    R = R.astype(np.float32)
    noise = np.zeros(L.shape, dtype=np.float32)
    noise = cv2.randn(noise, 0, 0.05)
    L_n = cv2.add(L, noise)
    L_c = cv2.multiply(L, 1.1)
    
    D_L = np.abs(stereo.disparity_ncorr(L, R, search_range=201))
    D_R = np.abs(stereo.disparity_ncorr(R, L, search_range=201))
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(os.path.join('output', 'ps2-4-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps2-4-a-2.png'), D_R)
    
    D_L_n = np.abs(stereo.disparity_ncorr(L_n, R, search_range=201))
    D_R_n = np.abs(stereo.disparity_ncorr(R, L_n, search_range=201))
    D_L_c = np.abs(stereo.disparity_ncorr(L_c, R, search_range=201))
    D_R_c = np.abs(stereo.disparity_ncorr(R, L_c, search_range=201))
                   
    D_L_n = cv2.normalize(D_L_n, D_L_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_n = cv2.normalize(D_R_n, D_R_n, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_L_c = cv2.normalize(D_L_c, D_L_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_c = cv2.normalize(D_R_c, D_R_c, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(os.path.join('output', 'ps2-4-b-1.png'), D_L_n)
    cv2.imwrite(os.path.join('output', 'ps2-4-b-2.png'), D_R_n)
    cv2.imwrite(os.path.join('output', 'ps2-4-b-3.png'), D_L_c)
    cv2.imwrite(os.path.join('output', 'ps2-4-b-4.png'), D_R_c)
    
    # Problem 5
    L = cv2.imread(os.path.join('input', 'pair2-L.png'), 0) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair2-R.png'), 0) * (1.0 / 255.0)
    L = L.astype(np.float32)
    R = R.astype(np.float32)
    
    D_L = np.abs(stereo.disparity_ncorr(L, R, search_range=201))
    D_R = np.abs(stereo.disparity_ncorr(R, L, search_range=201))
    
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(os.path.join('output', 'ps2-5-a-1.png'), D_L)
    cv2.imwrite(os.path.join('output', 'ps2-5-a-2.png'), D_R)
