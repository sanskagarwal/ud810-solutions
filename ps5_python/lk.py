import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pyramid
import warp


def lucas_kanade(img1, img2, w_size=(15, 15)):
    img1c = img1.copy()
    img2c = img2.copy()
    img1c = cv2.GaussianBlur(img1c, w_size, 0)
    img2c = cv2.GaussianBlur(img2c, w_size, 0)
    img1c = img1c * (1.0 / 255.0)
    img2c = img2c * (1.0 / 255.0)
    Ix = cv2.Sobel(img1c, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1c, cv2.CV_64F, 0, 1, ksize=5)
    It = img2c - img1c

    Sxx = cv2.boxFilter(Ix * Ix, -1, ksize=w_size, normalize=False)
    Sxy = cv2.boxFilter(Ix * Iy, -1, ksize=w_size, normalize=False)
    Syy = cv2.boxFilter(Iy * Iy, -1, ksize=w_size, normalize=False)
    Sxt = cv2.boxFilter(Ix * It, -1, ksize=w_size, normalize=False)
    Syt = cv2.boxFilter(Iy * It, -1, ksize=w_size, normalize=False)

    h, w = img1c.shape
    flow = np.zeros((h, w, 2), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            A = np.array([Sxx[r, c], Sxy[r, c], Sxy[r, c], Syy[r, c]]).reshape((2, 2))
            b = np.array([Sxt[r, c], Syt[r, c]])
            flow[r, c, :], _, _, _ = np.linalg.lstsq(A, -b, rcond=None)
    return flow


def draw_flow_quiver(flow1, img, filename, step=None):
    flow = flow1.copy()
    flow = np.flip(flow, axis=0)
    h, w = img.shape
    if step is None:
        step = min(h, w) // 30
    flow = flow[:h, :w, :]  # Change in size because of gaussian pyramids
    x = np.arange(0, w, step)
    y = np.arange(0, h, step)
    x, y = np.meshgrid(x, y)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.quiver(x, y, flow[::step, ::step, 0], -flow[::step, ::step, 1], color='r')
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()


def level_lk(img1, img2, level=0, w_size=(21, 21)):
    im1 = img1.copy().astype(np.float32)
    im2 = img2.copy().astype(np.float32)
    for _ in range(level):
        im1 = pyramid.reduce(im1)
        im2 = pyramid.reduce(im2)
    im1 = cv2.normalize(im1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.normalize(im2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = lucas_kanade(im1, im2, w_size)
    for _ in range(level):
        flow1x = pyramid.expand(flow[:, :, 0])
        flow1y = pyramid.expand(flow[:, :, 1])
        flow = np.zeros((flow1x.shape[0], flow1x.shape[1], 2), dtype=np.float32)
        flow[:, :, 0] = flow1x
        flow[:, :, 1] = flow1y
        flow *= 2
    return flow, im1, im2


def equate_size(flow, shape):
    h, w = shape
    h1, w1, _ = flow.shape
    if h1 >= h and w1 >= w:
        flow = flow[:h, :w, :]
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        diff_h = h - h1
        diff_w = w - w1
        if diff_h > 0:
            for _ in range(diff_h):
                u = np.append(u, np.atleast_2d(np.zeros(w1, dtype=np.float32)), axis=0)
                v = np.append(v, np.atleast_2d(np.zeros(w1, dtype=np.float32)), axis=0)
                h1 += 1
        if diff_w > 0:
            for _ in range(diff_w):
                u = np.append(u, np.atleast_2d(np.zeros(h1, dtype=np.float32)).T, axis=1)
                v = np.append(v, np.atleast_2d(np.zeros(h1, dtype=np.float32)).T, axis=1)
                w += 1
        flow = np.stack((u, v), axis=-1)
    return flow


def hierarchical_lk(img1, img2, k=None):
    h, w = img1.shape
    if k is None:
        mini = min(h, w)
        k = int(math.floor(math.log2(mini)))
    lk = img1.copy().astype(np.float32)
    rk = img2.copy().astype(np.float32)
    levels_lk = []
    levels_rk = []
    for _ in range(k):
        levels_lk.append(lk.copy())
        levels_rk.append(rk.copy())
        lk = pyramid.reduce(lk)
        rk = pyramid.reduce(rk)
    u = np.zeros(lk.shape, dtype=np.float32)
    v = np.zeros(rk.shape, dtype=np.float32)

    for i in range(k - 1, -1, -1):
        up = pyramid.expand(u)
        vp = pyramid.expand(v)
        up *= 2
        vp *= 2
        flowp = np.stack((up, vp), axis=-1)
        flowp = equate_size(flowp, levels_lk[i].shape)
        img2_pred = warp.warp_back(levels_rk[i], flowp)
        flow_corr = lucas_kanade(levels_lk[i], img2_pred)
        print(i)
        u = flowp[:, :, 0] + flow_corr[:, :, 0]
        v = flowp[:, :, 1] + flow_corr[:, :, 1]
    return np.stack((u, v), axis=-1)
