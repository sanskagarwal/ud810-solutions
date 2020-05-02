#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:57:02 2020

@author: sanskar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import harris

if __name__ == '__main__':
    # Problem 1
    transA = cv2.imread('input/transA.jpg', 0)
    transB = cv2.imread('input/transB.jpg', 0)
    simA = cv2.imread('input/simA.jpg', 0)
    simB = cv2.imread('input/simB.jpg', 0)
    
    transAX = cv2.Sobel(transA, cv2.CV_64F, 1, 0, ksize=5)
    transAY = cv2.Sobel(transA, cv2.CV_64F, 0, 1, ksize=5)
    simAX = cv2.Sobel(simA, cv2.CV_64F, 1, 0, ksize=5)
    simAY = cv2.Sobel(simA, cv2.CV_64F, 0, 1, ksize=5)
    
    transComb = transAX.copy()
    transComb = np.append(transComb, transAY, axis=1)
    simComb = simAX.copy()
    simComb = np.append(simComb, simAY, axis=1)
    
    transComb = cv2.normalize(transComb, transComb, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    simComb = cv2.normalize(simComb, simComb, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite('output/ps4-1-a-1.png', transComb)
    cv2.imwrite('output/ps4-1-a-2.png', simComb)
    
    transAH = harris.harris_values(transA)
    transBH = harris.harris_values(transB)
    simAH = harris.harris_values(simA)
    simBH = harris.harris_values(simB)
    
    transAH = cv2.normalize(transAH, transAH, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    transBH = cv2.normalize(transBH, transBH, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    simAH = cv2.normalize(simAH, simAH, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    simBH = cv2.normalize(simBH, simBH, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite('output/ps4-1-b-1.png', transAH)
    cv2.imwrite('output/ps4-1-b-2.png', transBH)
    cv2.imwrite('output/ps4-1-b-3.png', simAH)
    cv2.imwrite('output/ps4-1-b-4.png', simBH)
    
    transAH = harris.harris_values(transA)
    transBH = harris.harris_values(transB)
    simAH = harris.harris_values(simA)
    simBH = harris.harris_values(simB)
    
    ctA = harris.harris_corners(transAH)
    ctB = harris.harris_corners(transBH)
    csA = harris.harris_corners(simAH)
    csB = harris.harris_corners(simBH)
    
    transAC = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
    transBC = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
    simAC = cv2.cvtColor(simA, cv2.COLOR_GRAY2RGB)
    simBC = cv2.cvtColor(simB, cv2.COLOR_GRAY2RGB)
    
    for x,y in ctA:
        cv2.drawMarker(transAC, (x,y), (255, 0, 0), cv2.MARKER_CROSS)
    for x,y in ctB:
        cv2.drawMarker(transBC, (x,y), (255, 0, 0), cv2.MARKER_CROSS)
    for x,y in csA:
        cv2.drawMarker(simAC, (x,y), (255, 0, 0), cv2.MARKER_CROSS)
    for x,y in csB:
        cv2.drawMarker(simBC, (x,y), (255, 0, 0), cv2.MARKER_CROSS)
    
    cv2.imwrite('output/ps4-1-c-1.png', transAC)
    cv2.imwrite('output/ps4-1-c-2.png', transBC)
    cv2.imwrite('output/ps4-1-c-3.png', simAC)
    cv2.imwrite('output/ps4-1-c-4.png', simBC)