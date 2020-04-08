#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:54:44 2020

@author: sanskar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import hough

if __name__ == '__main__':
    img = cv2.imread(os.path.join('input', 'ps1-input0.png'), 0)
    # Problem 1: Canny edge
    img_clean = cv2.GaussianBlur(img,(5,5),1)
    
    img_sobelx = cv2.Sobel(img_clean,cv2.CV_64F,1,0,ksize=3)
    img_sobely = cv2.Sobel(img_clean,cv2.CV_64F,0,1,ksize=3)
    img_grad = cv2.Laplacian(img_clean,cv2.CV_64F)
    img_dir = np.arctan2(img_sobely,img_sobelx)
    
    plt.imshow(img_sobelx, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    img_edges = cv2.Canny(img, 100, 200)
    cv2.imwrite(os.path.join('output', 'ps1-1-a-1.png'), img_edges)

    # Problem 2: Hough Line
    H, theta, rho = hough.hough_lines_acc(img_edges)
    H_norm = (H/np.amax(H)) * 255
    cv2.imwrite(os.path.join('output', 'ps1-2-a-1.png'), H_norm)

    H1 = H_norm.copy()
    peaks = hough.hough_peaks(H1, 8, nHoodSize = (50,50))
    for x,y in peaks:
        cv2.circle(H1, (int(x), int(y)), 10, (255,0,0), 1)
    cv2.imwrite(os.path.join('output', 'ps1-2-b-1.png'), H1)
    
    hough.hough_lines_draw(img, 'ps1-2-c-1.png', peaks, rho, theta)
    
    # Problem 3
    img = cv2.imread(os.path.join('input', 'ps1-input0-noise.png'), 0)
    img_clean = cv2.GaussianBlur(img,(15,15),1)
    cv2.imwrite(os.path.join('output', 'ps1-3-a-1.png'), img_clean)
    
    img_edges1 = cv2.Canny(img, 200, 400)
    cv2.imwrite(os.path.join('output', 'ps1-3-b-1.png'), img_edges1)
    
    img_edges2 = cv2.Canny(img_clean, 200, 400)
    cv2.imwrite(os.path.join('output', 'ps1-3-b-2.png'), img_edges2)

    H, theta, rho = hough.hough_lines_acc(img_edges2)
    H_norm = (H/np.amax(H)) * 255

    H1 = H_norm.copy()
    peaks = hough.hough_peaks(H1, 8, nHoodSize = (50,50))
    for x,y in peaks:
        cv2.circle(H1, (int(x), int(y)), 10, (255,0,0), 1)
    cv2.imwrite(os.path.join('output', 'ps1-3-c-1.png'), H1)

    hough.hough_lines_draw(img, 'ps1-3-c-2.png', peaks, rho, theta)
    
    # Problem 4
    img = cv2.imread(os.path.join('input', 'ps1-input1.png'), 0)
    img_clean = cv2.GaussianBlur(img,(15,15),1)
    cv2.imwrite(os.path.join('output', 'ps1-4-a-1.png'), img_clean)

    img_edges = cv2.Canny(img_clean, 200, 400)
    cv2.imwrite(os.path.join('output', 'ps1-4-b-1.png'), img_edges)
    H, theta, rho = hough.hough_lines_acc(img_edges)
    H_norm = (H/np.amax(H)) * 255

    H1 = H_norm.copy()
    peaks = hough.hough_peaks(H1, 10, nHoodSize = (21,21))
    for x,y in peaks:
        cv2.circle(H1, (int(x), int(y)), 10, (255,0,0), 1)
    cv2.imwrite(os.path.join('output', 'ps1-4-c-1.png'), H1)

    hough.hough_lines_draw(img, 'ps1-4-c-2.png', peaks, rho, theta)
