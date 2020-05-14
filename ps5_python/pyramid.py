import numpy as np
import cv2


def reduce(img1):
    img = img1.copy()
    img = img.astype(np.float32)
    kernel = np.atleast_2d([1, 4, 6, 4, 1]) / 16  # a= 0.375
    kernel = np.matmul(kernel.T, kernel)
    dst = cv2.filter2D(img, -1, kernel)
    dst = dst[::2, ::2]
    h, w = dst.shape
    return dst


def expand(img1):
    img = img1.copy().astype(np.float32)
    kernel = np.atleast_2d([1, 4, 6, 4, 1]) / 8
    kernel = np.matmul(kernel.T, kernel)
    h, w = img.shape
    dst = np.zeros((h * 2, w * 2), dtype=np.float32)
    dst[::2, ::2] = img
    dst = cv2.filter2D(dst, -1, kernel)
    return dst


def find_diff(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    minh = min(h1, h2)
    minw = min(w1, w2)
    return cv2.subtract(img1[:minh, :minw], img2[:minh, :minw])
