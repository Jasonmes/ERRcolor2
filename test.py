# encoding: utf-8

import cv2
import numpy as np

image = cv2.imread('C_test1.png')

color1 = [([0, 148, 178],
           [178, 255, 255])]
color2 = [([0, 87, 185],
           [88, 142, 255])]


for (lower, upper) in color2:
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
    dst = cv2.erode(binary, kernel = None, iterations=20)
    dst2 = cv2.dilate(dst, kernel = None, iterations=20)

    # 高斯模糊
    imgBlur = cv2.GaussianBlur(dst2, (3, 3), 0)

    imgBlur2 = cv2.medianBlur(dst2,9)
    imgBlur3 = cv2.blur(imgBlur2,(10,10))

    # 检测轮廓
    '''
    def custom_blur_demo(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        dst = cv2.filter2D(image, -1, kernel=kernel)
        cv2.imshow("custom_blur_demo", dst)


    custom_blur_demo(imgBlur3)
    '''

    # 二值化
    ret, binary2 = cv2.threshold(imgBlur3,
                                0,
                                255,
                                cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)



    imgCanny = cv2.Canny(binary2, 200, 300)

    minLineLength = 5
    maxLineGap = 70
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
    cv2.imshow("w0",imgCanny)


    cv2.namedWindow("w1", 0)
    cv2.resizeWindow("w1", 640, 480)
    cv2.imshow("w1", binary2)
    cv2.waitKey(0)







