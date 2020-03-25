#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:37:10 2020

@author: sanskar
"""

import os
import cv2
import numpy as np

def selectMidRange(w, h, x, y):
    diffx = w-x
    diffy = h-y
    if(diffx<0 or diffy<0): # Not Possible
        return -1
    stx = diffx//2
    sty = diffy//2
    return (stx, stx+x, sty, sty+y)

if __name__ == '__main__':
    # Problem 1
    img1 = cv2.imread(os.path.join("output", "ps0-1-a-1.png"), -1)
    img2 = cv2.imread(os.path.join("output", "ps0-1-a-2.png"), -1)
    
    # Problem 2
    b,g,r = cv2.split(img1)
    img1_copy = cv2.merge((r,g,b))
    # or
    # img1_copy1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join("output", "ps0-2-a-1.png"), img1_copy)
    
    img1_green = img1[:,:,1]
    img1_red = img1[:,:,0]
    cv2.imwrite(os.path.join("output", "ps0-2-b-1.png"), img1_green)
    cv2.imwrite(os.path.join("output", "ps0-2-c-1.png"), img1_red)
    
    # Problem 3
    img2_red = img2[:,:,0]
    
    x = y = 100
    w,h = img2_red.shape
    x_st,x_end,y_st,y_end = selectMidRange(w, h, x, y)
    
    w,h = img1_red.shape
    x1_st,x1_end,y1_st,y1_end = selectMidRange(w, h, x, y)

    img2_copy = img2_red.copy()
    img2_copy[x_st:x_end, y_st:y_end] = img1_red[x1_st:x1_end, y1_st:y1_end]
    cv2.imwrite(os.path.join("output", "ps0-3-a-1.png"), img2_copy)
    
    # Problem 4
    mini, maxi, mean, std = img1_green.min(), img1_green.max(), img1_green.mean(), img1_green.std()
    img1_green1 = img1_green.copy()
    img1_green1 = cv2.subtract(img1_green1, mean)
    img1_green1 = cv2.divide(img1_green1, std)
    img1_green1 = cv2.multiply(img1_green1, 10)
    img1_green1 = cv2.add(img1_green1, mean)
    cv2.imwrite(os.path.join("output", "ps0-4-b-1.png"), img1_green1)
    
    img1_green2 = img1_green.copy()
    M = np.float32([[1, 0, -2], [0, 1, 0]])
    r, c = img1_green.shape
    img1_green2 = cv2.warpAffine(img1_green2, M, (c, r))
    cv2.imwrite(os.path.join("output", "ps0-4-c-1.png"), img1_green2)
    
    img1_green3 = cv2.subtract(img1_green, img1_green2)
    cv2.imwrite(os.path.join("output", "ps0-4-d-1.png"), img1_green3)
        
    # Problem 5
    noise = np.zeros(img1.shape[:2], np.uint8)
    cv2.randn(noise, 0, 11)
    img1_green_noise = img1.copy()
    img1_green_noise[:,:,1] = cv2.add(img1[:,:,1], noise)
    cv2.imwrite(os.path.join("output", "ps0-5-a-1.png"), img1_green_noise)
    
    img1_blue_noise = img1.copy()
    img1_blue_noise[:,:,0] = cv2.add(img1[:,:,0], noise)
    cv2.imwrite(os.path.join("output", "ps0-5-b-1.png"), img1_blue_noise)
    