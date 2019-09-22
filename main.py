#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Jason Mess

# encoding: utf-8

import preprocessing
import cv2
import numpy as np
# from matplotlib import pyplot as plt

frame = preprocessing.image


def line_fitness(pts, image, color=(0, 0, 255)):
    h, w, ch = image.shape
    [vx, vy, x, y] = cv2.fitLine(np.array(pts), cv2.DIST_L1, 0, 0.01, 0.01)
    y1 = int((-x * vy / vx) + y)
    y2 = int(((w - x) * vy / vx) + y)
    cv2.line(image, (w - 1, y2), (0, y1), color, 2)
    return image


h, w, ch = frame.shape
gray = cv2.cvtColor(frame,
                    cv2.COLOR_BGR2GRAY)
ret, Binary = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dist = cv2.distanceTransform(Binary,
                             cv2.DIST_L1,
                             cv2.DIST_MASK_PRECISE)
dist = dist / 15
# 水平 垂直 投影 提取骨架
result = np.zeros((h, w), dtype=np.uint8)
ypts = []
for row in range(h):
    cx = 0
    cy = 0
    max_d = 0
    for col in range(w):
        d = dist[row][col]
        if d > max_d:
            max_d = d
            cx = col
            cy = row
    result[cy][cx] = 255
    ypts.append([cx, cy])

xpts = []
for col in range(w):
    cx = 0
    cy = 0
    max_d = 0
    for row in range(h):
        d = dist[row][col]
        if d > max_d:
            max_d = d
            cx = col
            cy = row
    result[cy][cx] = 255
    xpts.append([cx, cy])

frame = line_fitness(ypts, image=frame, color=(0, 0, 255))
frame = line_fitness(xpts, image=frame, color=(255, 0, 0))

cv2.namedWindow("Binary", 0)
cv2.resizeWindow("Binary", 640, 480)
cv2.imshow("Binary", frame)
cv2.waitKey()
