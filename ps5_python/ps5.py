#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 5 13:40:08 2020

@author: sanskar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
# User Modules
import lk


if __name__ == '__main__':
    # Problem 1
    shift0 = cv2.imread('input/TestSeq/Shift0.png', 0)
    shiftr2 = cv2.imread('input/TestSeq/ShiftR2.png', 0)
    shiftr5u5 = cv2.imread('input/TestSeq/ShiftR5U5.png', 0)
    shiftr10 = cv2.imread('input/TestSeq/ShiftR10.png', 0)
    shiftr20 = cv2.imread('input/TestSeq/ShiftR20.png', 0)
    shiftr40 = cv2.imread('input/TestSeq/ShiftR40.png', 0)

    flow1 = lk.lucas_kanade(shift0, shiftr2, w_size=(21, 21))
    flow2 = lk.lucas_kanade(shift0, shiftr5u5, w_size=(21, 21))
    flow3 = lk.lucas_kanade(shift0, shiftr10, w_size=(31, 31))
    flow4 = lk.lucas_kanade(shift0, shiftr20, w_size=(31, 31))
    flow5 = lk.lucas_kanade(shift0, shiftr40, w_size=(31, 31))

    lk.draw_flow_quiver(flow1, shift0, 'output/ps5-1-a-1.png')
    lk.draw_flow_quiver(flow2, shift0, 'output/ps5-1-a-2.png')
    lk.draw_flow_quiver(flow3, shift0, 'output/ps5-1-b-1.png')
    lk.draw_flow_quiver(flow4, shift0, 'output/ps5-1-b-2.png')
    lk.draw_flow_quiver(flow5, shift0, 'output/ps5-1-b-3.png')

