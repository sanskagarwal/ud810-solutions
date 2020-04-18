#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:12:25 2020

@author: sanskar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import geometry

if __name__ == '__main__':
    # Problem 1 a
    points_2d = np.loadtxt('input/pts2d-norm-pic_a.txt', dtype=np.float64)
    points_3d = np.loadtxt('input/pts3d-norm.txt', dtype=np.float64)

    M = geometry.lstsq(points_3d, points_2d) 
    print(M)
    np.savetxt('output/M_lstsq.txt', M)
    point_2d_proj = np.dot(M, np.append(points_3d[-1,:], 1))
    point_2d_proj /= point_2d_proj[2]
    point_2d_proj = point_2d_proj[:2]
    print(point_2d_proj)
    res = np.linalg.norm(point_2d_proj-points_2d[-1,:])
    print(res)
    
    # Problem 1 b
    points_2d = np.loadtxt('input/pts2d-pic_b.txt', dtype=np.float64)
    points_3d = np.loadtxt('input/pts3d.txt', dtype=np.float64)
    num_points = points_2d.shape[0]
    
    min_res = np.inf
    save_M = np.zeros((3,4))
    for k in [8, 12, 16]:
        for i in range(10):
            rand_ind = np.arange(0, num_points, step=1, dtype=np.int32)
            rand_ind = np.random.permutation(rand_ind)
            pt_2d = points_2d[rand_ind]
            pt_3d = points_3d[rand_ind]
            M = geometry.lstsq(pt_3d[:k,:], pt_2d[:k,:])
            res = 0
            for x in range(num_points-4, num_points):
                point_2d_proj = np.dot(M, np.append(pt_3d[x,:], 1))
                point_2d_proj /= point_2d_proj[2]
                point_2d_proj = point_2d_proj[:2]
                res += np.linalg.norm(point_2d_proj-pt_2d[x,:])
            res/=4
            # print(res)
            if(res<min_res):
                min_res = res
                save_M = M
    print(min_res)
    print(save_M)
    
    # Problem 1 c
    Q = save_M[:, :3]
    m4 = save_M[:, 3]
    C = np.dot(-np.linalg.inv(Q), m4)
    print(C)
    
    # Problem 2 a
    pl = np.loadtxt('input/pts2d-pic_a.txt', dtype=np.float64)
    pr = np.loadtxt('input/pts2d-pic_b.txt', dtype=np.float64)
    points = pl.shape[0]
    
    F = geometry.lstsqF(pl, pr)
    
    # Problem 2 b
    U, D, Vt = np.linalg.svd(F)
    D[-1] = 0
    D = np.diag(D)
    F = np.matmul(np.matmul(U, D), Vt)
    print(F)
    
    # Problem 2 c
    imgl = cv2.imread('input/pic_a.jpg', 1)
    imgr = cv2.imread('input/pic_b.jpg', 1)

    height, width, _ = imgl.shape
    tl = np.array([0, 0, 1])
    tr = np.array([width, 0, 1])
    bl = np.array([0, height, 1])
    br = np.array([width, height, 1])
    
    lineL = np.cross(tl, bl)
    lineR = np.cross(tr, br)
    
    for i in range(points):
        pa = pl[i,:]
        pb = pr[i,:]
        pa = np.append(pa, 1)
        pb = np.append(pb, 1)
        
        epilb = np.matmul(F, pa.T)
        epila = np.matmul(F.T, pb.T)
                
        pt1a = np.cross(lineL, epila)
        pt2a = np.cross(lineR, epila)
        pt1b = np.cross(lineL, epilb)
        pt2b = np.cross(lineR, epilb)
        
        pt1a /= pt1a[-1]
        pt2a /= pt2a[-1]
        pt1b /= pt1b[-1]
        pt2b /= pt2b[-1]
        
        cv2.line(imgl, tuple(pt1a[:2].astype(int)), tuple(pt2a[:2].astype(int)), color=(0, 255, 0), thickness=1)
        cv2.line(imgr, tuple(pt1b[:2].astype(int)), tuple(pt2b[:2].astype(int)), color=(0, 255, 0), thickness=1)

    cv2.imwrite('output/ps3-2-c-1.png', imgl)
    cv2.imwrite('output/ps3-2-c-2.png', imgr)
    