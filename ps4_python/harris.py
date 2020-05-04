import numpy as np
import cv2


def harris_values(img, alpha=0.05, w_size=5, norm=True):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    Ixx = cv2.multiply(Ix, Ix)
    Iyy = cv2.multiply(Iy, Iy)
    Ixy = cv2.multiply(Ix, Iy)
    Iyx = cv2.multiply(Iy, Ix)

    Hres = np.zeros(img.shape, dtype=np.float64)
    h, w = img.shape
    st = w_size // 2
    for r in range(st, h - st):
        minr = r - st
        maxr = minr + st
        for c in range(st, w - st):
            minc = c - st
            maxc = minc + st
            wIxx = Ixx[minr:maxr, minc:maxc]
            wIxy = Ixy[minr:maxr, minc:maxc]
            wIyx = Iyx[minr:maxr, minc:maxc]
            wIyy = Iyy[minr:maxr, minc:maxc]

            MIxx = cv2.GaussianBlur(wIxx, (w_size, w_size), 0).sum()
            MIxy = cv2.GaussianBlur(wIxy, (w_size, w_size), 0).sum()
            MIyy = cv2.GaussianBlur(wIyy, (w_size, w_size), 0).sum()
            MIyx = cv2.GaussianBlur(wIyx, (w_size, w_size), 0).sum()

            M = np.array([MIxx, MIxy, MIyx, MIyy]).reshape((2, 2))
            Hres[r, c] = np.linalg.det(M) - alpha * (M.trace() ** 2)

    if (norm == True):
        Hres = cv2.normalize(Hres, Hres, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Hres


def harris_corners(H1, threshold=None, hoodSize=None):
    H = H1.copy()
    if (hoodSize == None):
        hoodSize = np.array(H.shape) // 50 + 1  # 2%
    if (threshold == None):
        threshold = 0.5 * np.max(H)

    h, w = H.shape
    hoodY, hoodX = hoodSize
    corners = []

    while (True):
        max_val = np.max(H)
        if (max_val < threshold):
            break
        pos = np.unravel_index(H.argmax(), H.shape)
        x = pos[1]
        y = pos[0]
        corners.append([x, y])
        stx = max(0, x - hoodX // 2)
        endx = min(w, x + hoodX // 2)
        sty = max(0, y - hoodY // 2)
        endy = min(h, y + hoodY // 2)

        H[sty:endy, stx:endx] = 0

    return corners
