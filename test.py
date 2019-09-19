# encoding: utf-8

import cv2
import numpy as np

image = cv2.imread('C_test1.png')

color1 = [([0, 148, 178],
           [178, 255, 255])]
color2 = [([0, 87, 185],
           [88, 142, 255])]

for (lower, upper) in color1:
    lower = np.array(lower, dtype="uint8")  # 颜色下限
    upper = np.array(upper, dtype="uint8")  # 颜色上限
    # 提取需要的区域
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # 灰度值
    gray = cv2.cvtColor(output,
                        cv2.COLOR_RGB2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray,
                                0,
                                255,
                                cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    # 侵蚀 和 膨胀
    dst = cv2.erode(binary, None, iterations=5)
    dst2 = cv2.dilate(dst, None, iterations=20)

    # 高斯模糊
    imgBlur = cv2.GaussianBlur(dst2, (3, 3), 0)
    # 检测轮廓
    imgCanny = cv2.Canny(imgBlur, 70, 210)

    minLineLength = 10
    maxLineGap = 50
    lines = cv2.HoughLinesP(imgCanny,
                            5,
                            np.pi / 180,
                            100,
                            minLineLength,
                            maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)



    cv2.namedWindow("w0", 0)
    cv2.resizeWindow("w0", 640, 480)
    cv2.imshow("w0", imgCanny)


    cv2.namedWindow("w1", 0)
    cv2.resizeWindow("w1", 640, 480)
    cv2.imshow("w1", image)
    cv2.waitKey(0)





'''
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


