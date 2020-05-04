#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:57:02 2020

@author: sanskar
"""

import numpy as np
import cv2

# User Modules
import harris
import sift
import ransac

transA = cv2.imread('input/transA.jpg', 0)
transB = cv2.imread('input/transB.jpg', 0)
simA = cv2.imread('input/simA.jpg', 0)
simB = cv2.imread('input/simB.jpg', 0)


class LinePoints:
    def __init__(self, k1, k2, rw):
        self.x1 = int(k1.pt[0])
        self.x2 = int(k2.pt[0] + rw[1])
        self.y1 = int(k1.pt[1])
        self.y2 = int(k2.pt[1])


def generate_color_images():
    transAC = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
    transBC = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
    simAC = cv2.cvtColor(simA, cv2.COLOR_GRAY2RGB)
    simBC = cv2.cvtColor(simB, cv2.COLOR_GRAY2RGB)
    transCol = transAC.copy()
    transCol = np.append(transCol, transBC, axis=1)
    simCol = simAC.copy()
    simCol = np.append(simCol, simBC, axis=1)
    return transCol, simCol


def save_img(matches, name, img):
    for l in matches:
        cv2.drawMarker(img, (l.x1, l.y1), (255, 0, 0), cv2.MARKER_CROSS)
        cv2.drawMarker(img, (l.x2, l.y2), (255, 0, 0), cv2.MARKER_CROSS)
        cv2.line(img, (l.x1, l.y1), (l.x2, l.y2), color=(0, 255, 0), thickness=1)
    cv2.imwrite(name, img)


if __name__ == '__main__':
    # Problem 1
    transAX = cv2.Sobel(transA, cv2.CV_64F, 1, 0, ksize=5)
    transAY = cv2.Sobel(transA, cv2.CV_64F, 0, 1, ksize=5)
    transBX = cv2.Sobel(transB, cv2.CV_64F, 1, 0, ksize=5)
    transBY = cv2.Sobel(transB, cv2.CV_64F, 0, 1, ksize=5)
    simAX = cv2.Sobel(simA, cv2.CV_64F, 1, 0, ksize=5)
    simAY = cv2.Sobel(simA, cv2.CV_64F, 0, 1, ksize=5)
    simBX = cv2.Sobel(simB, cv2.CV_64F, 1, 0, ksize=5)
    simBY = cv2.Sobel(simB, cv2.CV_64F, 0, 1, ksize=5)

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

    cv2.imwrite('output/ps4-1-b-1.png', transAH)
    cv2.imwrite('output/ps4-1-b-2.png', transBH)
    cv2.imwrite('output/ps4-1-b-3.png', simAH)
    cv2.imwrite('output/ps4-1-b-4.png', simBH)

    ctA = harris.harris_corners(transAH, threshold=160)
    ctB = harris.harris_corners(transBH, threshold=159)
    csA = harris.harris_corners(simAH, threshold=150)
    csB = harris.harris_corners(simBH, threshold=137)

    transAC = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
    transBC = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
    simAC = cv2.cvtColor(simA, cv2.COLOR_GRAY2RGB)
    simBC = cv2.cvtColor(simB, cv2.COLOR_GRAY2RGB)

    for x, y in ctA:
        cv2.drawMarker(transAC, (x, y), (255, 0, 0), cv2.MARKER_CROSS)
    for x, y in ctB:
        cv2.drawMarker(transBC, (x, y), (255, 0, 0), cv2.MARKER_CROSS)
    for x, y in csA:
        cv2.drawMarker(simAC, (x, y), (255, 0, 0), cv2.MARKER_CROSS)
    for x, y in csB:
        cv2.drawMarker(simBC, (x, y), (255, 0, 0), cv2.MARKER_CROSS)

    cv2.imwrite('output/ps4-1-c-1.png', transAC)
    cv2.imwrite('output/ps4-1-c-2.png', transBC)
    cv2.imwrite('output/ps4-1-c-3.png', simAC)
    cv2.imwrite('output/ps4-1-c-4.png', simBC)

    # Problem 2
    transA_angle = np.arctan2(transAY, transAX)
    transB_angle = np.arctan2(transBY, transBX)
    simA_angle = np.arctan2(simAY, simAX)
    simB_angle = np.arctan2(simBY, simBX)

    transAC = cv2.cvtColor(transA, cv2.COLOR_GRAY2RGB)
    transBC = cv2.cvtColor(transB, cv2.COLOR_GRAY2RGB)
    simAC = cv2.cvtColor(simA, cv2.COLOR_GRAY2RGB)
    simBC = cv2.cvtColor(simB, cv2.COLOR_GRAY2RGB)

    kp1 = []
    kp2 = []
    kp3 = []
    kp4 = []
    for x, y in ctA:
        kp1.append(cv2.KeyPoint(x, y, _size=3, _angle=np.rad2deg(transA_angle[y, x]), _octave=0))
    for x, y in ctB:
        kp2.append(cv2.KeyPoint(x, y, _size=3, _angle=np.rad2deg(transB_angle[y, x]), _octave=0))
    for x, y in csA:
        kp3.append(cv2.KeyPoint(x, y, _size=3, _angle=np.rad2deg(simA_angle[y, x]), _octave=0))
    for x, y in csB:
        kp4.append(cv2.KeyPoint(x, y, _size=3, _angle=np.rad2deg(simB_angle[y, x]), _octave=0))

    cv2.drawKeypoints(transAC, kp1, transAC, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(transBC, kp2, transBC, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(simAC, kp3, simAC, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(simBC, kp4, simBC, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    transCol = transAC.copy()
    transCol = np.append(transCol, transBC, axis=1)
    simCol = simAC.copy()
    simCol = np.append(simCol, simBC, axis=1)

    cv2.imwrite('output/ps4-2-a-1.png', transCol)
    cv2.imwrite('output/ps4-2-a-2.png', simCol)

    _, des1 = sift.compute(transA, kp1)
    _, des2 = sift.compute(transB, kp2)
    _, des3 = sift.compute(simA, kp3)
    _, des4 = sift.compute(simB, kp4)

    match_img1 = sift.good_matches(des1, des2)
    match_img2 = sift.good_matches(des3, des4)

    line1 = []
    for match in match_img1:
        match = match[0]
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx
        line1.append(LinePoints(kp1[kp1_idx], kp2[kp2_idx], transA.shape))
    line2 = []
    for match in match_img2:
        match = match[0]
        kp3_idx = match.queryIdx
        kp4_idx = match.trainIdx
        line2.append(LinePoints(kp3[kp3_idx], kp4[kp4_idx], simA.shape))

    for line in line1:
        cv2.line(transCol, (line.x1, line.y1), (line.x2, line.y2), color=(0, 255, 0), thickness=1)
    for line in line2:
        cv2.line(simCol, (line.x1, line.y1), (line.x2, line.y2), color=(0, 255, 0), thickness=1)

    cv2.imwrite('output/ps4-2-b-1.png', transCol)
    cv2.imwrite('output/ps4-2-b-2.png', simCol)

    # Problem 3
    transCol, simCol = generate_color_images()
    _, in_matches = ransac.translation(np.array(line1), tolerance=10)
    save_img(in_matches, 'output/ps4-3-a-1.png', transCol)

    _, in_matches = ransac.similarity(np.array(line2), tolerance=5)
    save_img(in_matches, 'output/ps4-3-b-1.png', simCol)

    _, simCol = generate_color_images()
    _, in_matches = ransac.affine(np.array(line2), tolerance=2)
    save_img(in_matches, 'output/ps4-3-c-1.png', simCol)
