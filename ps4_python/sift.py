import cv2
import numpy as np


def compute(I, kp):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.compute(I, kp)


def good_matches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe's Ratio Test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good
