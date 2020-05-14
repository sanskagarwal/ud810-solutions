import cv2
import numpy as np


def warp_back(img, flow1):
    h, w = img.shape
    flow = -flow1.copy()
    flow = flow.astype(np.float32)
    flow = flow[:h, :w, :]   # Change in size because of gaussian pyramids
    vx = flow[:, :, 0]
    vy = flow[:, :, 1]
    x = np.arange(w, dtype=np.float32)
    y = np.transpose(np.atleast_2d(np.arange(h, dtype=np.float32)))
    return cv2.remap(img, x + vx, y + vy, cv2.INTER_LINEAR)
