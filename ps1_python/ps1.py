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
import math
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

    # Problem 5
    img = cv2.imread(os.path.join('input', 'ps1-input1.png'), 0)
    img_clean = cv2.GaussianBlur(img,(11,11),5)
    cv2.imwrite(os.path.join('output', 'ps1-5-a-1.png'), img_clean)
    
    img_edges = cv2.Canny(img_clean, 50, 100)
    cv2.imwrite(os.path.join('output', 'ps1-5-a-2.png'), img_edges)
    
    H = hough.hough_circles_acc(img_edges, 20)
    centers = hough.hough_peaks(H, 10, 0.75*np.amax(H))
    
    H1 = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    
    for x, y in centers:
        cv2.drawMarker(H1, (int(y), int(x)), (0,0,255), cv2.MARKER_DIAMOND)
        cv2.circle(H1, (int(y), int(x)), 20, (0,255,0), 2)
    cv2.imwrite(os.path.join('output', 'ps1-5-a-3.png'), H1)
    
    H2 = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    centers, radii = hough.find_circles(img_edges, np.arange(20,50,2))
    for point, r in zip(centers, radii):
        x, y = point[0], point[1]
        cv2.drawMarker(H2, (int(y), int(x)), (0,0,255), cv2.MARKER_DIAMOND)
        cv2.circle(H2, (int(y), int(x)), r, (0,255,0), 2)
    cv2.imwrite(os.path.join('output', 'ps1-5-b-1.png'), H2)
    
    # Problem 6
    img = cv2.imread(os.path.join('input', 'ps1-input2.png'), 0)
    img_clean = cv2.GaussianBlur(img,(15,15),1)

    img_edges = cv2.Canny(img_clean, 100, 110)
    H, theta, rho = hough.hough_lines_acc(img_edges)
    H_norm = (H/np.amax(H)) * 255

    H1 = H_norm.copy()
    peaks = hough.hough_peaks(H1, 10, threshold=120, nHoodSize = (51, 51))
    hough.hough_lines_draw(img, 'ps1-6-a-1.png', peaks, rho, theta)
    
    filterPeaks = hough.filter_lines(peaks)
    hough.hough_lines_draw(img, 'ps1-6-c-1.png', filterPeaks, rho, theta)

    # Problem 7
    img = cv2.imread(os.path.join('input', 'ps1-input2.png'), 0)
    img_clean = cv2.GaussianBlur(img,(15,15),1)
    img_edges = cv2.Canny(img_clean, 100, 200)
    
    H = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    centers, radii = hough.find_circles(img_edges, np.arange(20,30,2))
    for point, r in zip(centers, radii):
        x, y = point[0], point[1]
        cv2.drawMarker(H, (int(y), int(x)), (0,0,255), cv2.MARKER_DIAMOND)
        cv2.circle(H, (int(y), int(x)), r, (0,255,0), 2)
    cv2.imwrite(os.path.join('output', 'ps1-7-a-1.png'), H)
    
    # Problem 8
    img = cv2.imread(os.path.join('input', 'ps1-input3.png'), 0)
    img_clean = cv2.GaussianBlur(img,(15,15),1)
    img_edges = cv2.Canny(img_clean, 100, 200)
    
    H, theta, rho = hough.hough_lines_acc(img_edges)
    H_norm = (H/np.amax(H)) * 255
    H1 = H_norm.copy()
    peaks = hough.hough_peaks(H1, 10, threshold=120, nHoodSize = (51, 51))    
    filterPeaks = hough.filter_lines(peaks)
    
    H2 = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    centers, radii = hough.find_circles(img_edges, np.arange(20,24,2))
    for point, r in zip(centers, radii):
        x, y = point[0], point[1]
        cv2.drawMarker(H2, (int(y), int(x)), (255,0,0), cv2.MARKER_DIAMOND)
        cv2.circle(H2, (int(y), int(x)), r, (0,255,0), 2)
    
    for theta_idx , rho_idx in peaks:
        theta_idx = int(theta_idx)
        rho_idx = int(rho_idx)
        a = np.cos(theta[theta_idx]*math.pi/180)
        b = np.sin(theta[theta_idx]*math.pi/180)
        x0 = a*rho[rho_idx]
        y0 = b*rho[rho_idx]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(H2, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(os.path.join('output', 'ps1-8-a-1.png'), H2)
