import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def draw_flow_quiver(flow1, img, filename, step=5):
    flow = flow1.copy()
    flow = np.flip(flow, axis=0)
    h, w = img.shape
    x = np.arange(0, w, step)
    y = np.arange(0, h, step)
    x, y = np.meshgrid(x, y)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.quiver(x, y, flow[::step, ::step, 0], -flow[::step, ::step, 1], color='r')
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()
