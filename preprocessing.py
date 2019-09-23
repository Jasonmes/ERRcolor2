# encoding: utf-8

import cv2
import numpy as np

image = cv2.imread('C_test1.PNG')


color1 = [([0, 148, 178],
           [178, 255, 255])]
color2 = [([0, 87, 185],
           [88, 142, 255])]


# 二值
def threshold(gray):
    ret, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return ret, binary


# 侵蚀和膨胀，模糊
def magic(binary):
    # 侵蚀 和 膨胀
    dst = cv2.erode(binary, kernel=None, iterations=4)
    dst2 = cv2.dilate(dst, kernel=None, iterations=4)

    # 高斯模糊
    imgBlur = cv2.GaussianBlur(dst2, (3, 3), 0)
    # 中值滤波
    # imgBlur = cv2.medianBlur(dst2, 9)
    imgBlur = cv2.blur(imgBlur, (10, 10))
    imgBlur = cv2.erode(imgBlur, kernel=None, iterations=4)

    return imgBlur


# 给原图划线
def paiting_line(imgCanny, image):
    # 划线
    minLineLength = 700  # 线段以像素为单位的最小长度，根据应用场景设置
    maxLineGap = 0.01  # 为一条线段的最大允许间隔
    lines = cv2.HoughLinesP(imgCanny,      # 导入轮廓图
                            80,            # 线段以像素为单位的距离精度，double类型的，推荐用1.0
                            np.pi / 3600,
                            800,           # 值越大，基本上意味着检出的线段越长，检出的线段个数越少
                            minLineLength,
                            maxLineGap)

    # 把线画进原图
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     5)
    return image


for (lower, upper) in color2:
    lower = np.array(lower, dtype="uint8")  # 颜色下限
    upper = np.array(upper, dtype="uint8")  # 颜色上限
    # 提取需要的区域
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # 灰度值
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

    # 侵蚀 膨胀 模糊
    binary = gray
    for i in range(8):
        binary = binary
        binary = magic(binary)
        ret, binary = threshold(binary)

    # 检测轮廓
    imgCanny = cv2.Canny(binary, 200, 300)

    # 划线
    # image = paiting_line(imgCanny, image)

    '''
    # 窗口分别打开轮廓图和二值图
    windows = ["imgCanny", "binary", "image"]
    pictures = [imgCanny, binary, image]
    for i in range(3):
        print(i)
        cv2.namedWindow(windows[i], 0)
        cv2.resizeWindow(windows[i], 640, 480)
        cv2.imshow(windows[i], pictures[i])

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''











