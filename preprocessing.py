# encoding: utf-8

import cv2
import numpy as np
import take_name


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
    imgblur = cv2.GaussianBlur(dst2, (3, 3), 0)
    # 中值滤波
    # imgblur = cv2.medianBlur(dst2, 9)
    imgblur = cv2.blur(imgblur, (10, 10))
    imgblur = cv2.erode(imgblur, kernel=None, iterations=4)

    return imgblur


# 给原图划线
def painting_line_p(img_Canny, image):
    # 划线
    minLineLength = 700  # 线段以像素为单位的最小长度，根据应用场景设置
    maxLineGap = 0.1  # 为一条线段的最大允许间隔
    lines = cv2.HoughLinesP(img_Canny,      # 导入轮廓图
                            0.1,            # 线段以像素为单位的距离精度，double类型的，推荐用1.0
                            np.pi / 720,
                            6,           # 值越大，基本上意味着检出的线段越长，检出的线段个数越少
                            minLineLength,
                            maxLineGap)

    # 把线画进原图
    print len(lines)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     5)
    return image


def painting_line(image_canny, image):
    lines = cv2.HoughLines(image_canny,     # 要检测的二值图（一般是阈值分割或边缘检测后的图）
                           100,               # 距离r的精度，值越大，考虑越多的线
                           np.pi/180,       # 角度θ的精度，值越小，考虑越多的线
                           200)             # 累加数阈值，值越小，考虑越多的线
    '''
    print (lines)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image,
                 (x1, y1),
                 (x2, y2),
                 (0, 0, 255),
                 2)

    '''
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            print ('a', a, 'b', b)
            x0 = a * rho
            y0 = b * rho
            print ('x0', x0, 'y0', y0)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            print ('x1', x1,
                   'y1', y1,
                   'x2', x2,
                   'y2', y2)
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),
                     (0, 0, 255))
    return image


def pre_process(color, image):
    for (lower, upper) in color:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 提取需要的区域
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # 灰度值
        gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        # 侵蚀 膨胀 模糊
        binary = gray
        for i in range(15):
            binary = binary
            binary = magic(binary)
            ret, binary = threshold(binary)

        # 检测轮廓
        img_Canny = cv2.Canny(binary, 200, 300)
    return img_Canny


def display_us_image(o_image):
    name = take_name.varname(o_image)
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 640, 480)
    cv2.imshow(name, o_image)
    cv2.waitKey(0)


color1 = [([0, 148, 178],
           [178, 255, 255])]
color2 = [([0, 87, 185],
           [88, 142, 255])]

picture = ['C_test1.PNG',
           'C_test2.PNG',
           'C_test3.PNG',
           'C_test4.PNG',
           'C_test5.PNG',
           'C_test6.PNG',
           'C_test7.PNG']


image = cv2.imread(picture[1])

imgCanny = pre_process(color1, image)
canvas = np.zeros(image.shape, np.uint8)
cat, contours, hierarchy = cv2.findContours(imgCanny,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)

# 提取面积最大的轮廓
cnt = contours[0]

area = cv2.contourArea(cnt)
for cont in contours:
    if cv2.contourArea(cont) > area:
        cnt = cont
        area = cv2.contourArea(cont)


# approx = cv2.approxPolyDP(cnt, 30, True)
approx = cv2.approxPolyDP(cnt,
                          100,
                          True)


print (approx)
# 2.寻找凸包，得到凸包的角点
hull = cv2.convexHull(approx)
# 把线画进图
label = ['1', '2', '3', '4', '5',
         '6', '7', '8', '9', '10']
for i in range(len(hull)-1):
    for x1, y1 in hull[i]:
        # 给坐标做标记
        # cv2.putText(canvas, label[i], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        l = i+1
        for x2, y2 in hull[l]:
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),   # approx[i+1]
                     (0, 0, 255),
                     1)
# display_us_image(image)






