#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Jason Mess

# encoding: utf-8

import preprocessing
import cv2
import numpy as np

'''
imgcanny
提取轮廓 --- >> findcontour()
drawcontours

提取面积最大的轮廓， contourArea
多边形包围轮廓， approxolyDP
画出矩形 
convexhull
'''

imgCanny = preprocessing.imgCanny
image = preprocessing.image

# 画一个同尺寸的图片
canvas = np.zeros(image.shape, np.uint8)

# 提取轮廓
cat, contours, hierarchy = cv2.findContours(imgCanny,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)

# 提取面积最大的轮廓
cnt = contours[0]

'''
area = cv2.contourArea(cnt)
for cont in contours:
    if cv2.contourArea(cont) > area:
        cnt = cont
        area = cv2.contourArea(cont)
'''

approx = cv2.approxPolyDP(cnt,
                          30,
                          True)

# 2.寻找凸包，得到凸包的角点
hull = cv2.convexHull(approx)
# 把线画进图
label = ['1', '2', '3', '4', '5',
         '6', '7', '8', '9', '10']
for i in range(len(hull)-1):
    for x1, y1 in hull[i]:
        # 给坐标做标记
        # cv2.putText(canvas, label[i], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        for x2, y2 in hull[i+1]:
            cv2.line(canvas,
                     (x1, y1),
                     (x2, y2),   # approx[i+1]
                     (0, 0, 255),
                     1)

# 灰度图 二值
canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
dumy, canvas_threshold = cv2.threshold(canvas_gray,
                                       0,
                                       255,
                                       cv2.THRESH_BINARY)  # dumy no use
# 概率画线
image = preprocessing.painting_line_p(canvas_threshold, image)
# 直接画线
# image_end = preprocessing.painting_line(canvas_threshold, image)
# 显示图片
preprocessing.display_us_image(image)