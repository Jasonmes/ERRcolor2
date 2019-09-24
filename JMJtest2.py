# encoding: utf-8
import cv2
import numpy as np

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

def threshold(gray):
    ret, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return ret, binary

def painting_line(imgCanny, image):
    # 划线
    minLineLength = 800  # 线段以像素为单位的最小长度，根据应用场景设置
    maxLineGap = 0.1  # 为一条线段的最大允许间隔
    lines = cv2.HoughLinesP(imgCanny,      # 导入轮廓图
                            0.1,            # 线段以像素为单位的距离精度，double类型的，推荐用1.0
                            np.pi / 720,
                            10,           # 值越大，基本上意味着检出的线段越长，检出的线段个数越少
                            minLineLength,
                            maxLineGap)
    print len(lines)
    # 把线画进原图
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     5)
    return image


###########################################
############### processing ################

image = cv2.imread('C_test7.PNG')    # load image


color1 = [([0, 148, 178],
           [178, 255, 255])]    # define color space - free throw area
color2 = [([0, 87, 185],
           [88, 142, 255])]     # define color space - court

for (lower, upper) in color2:   # masking using color channel
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    # 提取需要的区域
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # 灰度值
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

    # 侵蚀 膨胀 模糊
    binary = gray
    for i in range(15):         # erosion and dilation loop 15 times ok
        binary = binary
        binary = magic(binary)
        ret, binary = threshold(binary)

    # 检测轮廓
    imgCanny = cv2.Canny(binary, 200, 300)

canvas = np.zeros(image.shape, np.uint8)
cat, contours, hierarchy = cv2.findContours(imgCanny,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)

#print len(contours[0])

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
        l = i+1
        for x2, y2 in hull[l]:
            cv2.line(canvas,
                     (x1, y1),
                     (x2, y2),   # approx[i+1]
                     (0, 0, 255),
                     1)

canvasG = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
dumy,canvasB = cv2.threshold(canvasG,0,255,cv2.THRESH_BINARY) # dumy no use


cv2.fillConvexPoly(canvas, hull, (0, 255, 0))
imgCanny2 = cv2.Canny(canvas, 1, 5)
print len(imgCanny2)


image = painting_line(canvasB, image)


cv2.namedWindow('Canny', 0)
cv2.resizeWindow('Canny', 640, 480)
cv2.imshow('Canny', imgCanny)


cv2.namedWindow('Hough', 0)
cv2.resizeWindow('Hough', 640, 480)
cv2.imshow('Hough', image)
cv2.waitKey(0)
