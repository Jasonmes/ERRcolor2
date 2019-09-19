# encoding: utf-8

import cv2
import numpy as np

image = cv2.imread('C_test1.png')

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
    dst2 = cv2.dilate(dst, kernel=None, iterations=5)

    # 高斯模糊
    imgBlur = cv2.GaussianBlur(dst2, (3, 3), 0)
    # 中值滤波
    # imgBlur = cv2.medianBlur(dst2, 9)
    imgBlur = cv2.blur(imgBlur, (10, 10))
    return imgBlur


for (lower, upper) in color1:
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
    minLineLength = 700
    maxLineGap = 0.1
    lines = cv2.HoughLinesP(imgCanny,      # 导入轮廓图
                            300,
                            np.pi / 180,
                            50,
                            minLineLength,
                            maxLineGap)
    print (len(lines))

    # 把线画进原图
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(image,
                     (x1, y1),
                     (x2, y2),
                     (0, 0, 255), 2)


    # 提取轮廓后，拟合外接多边形（矩形）
    '''
    _, contours, _ = cv2.findContours(binary,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    print("len(contours)=%d" % (len(contours)))
    print('\n================================\n')
    for idx, c in enumerate(contours):
        if len(c) < Config.min_contours:
            continue

        epsilon = Config.epsilon_start
        while True:
            approx = cv2.approxPolyDP(c,
                                      epsilon,
                                      True)
            print("idx,epsilon,len(approx),len(c)=%d,%d,%d,%d" % (idx,
                                                                  epsilon,
                                                                  len(approx),
                                                                len(c)))
            if len(approx) < 4:
                break
            if math.fabs(cv2.contourArea(approx)) > Config.min_area:
                if len(approx) > 4:
                    epsilon += Config.epsilon_step
                    print("epsilon=%d, count=%d" % (epsilon,
                                                    len(approx)))
                    continue
                else:
                    # for p in approx:
                    #    cv2.circle(binary,(p[0][0],p[0][1]),8,(255,255,0),thickness=-1)
                    approx = approx.reshape((4, 2))
                    # 点重排序, [top-left, top-right, bottom-right, bottom-left]
                    src_rect = order_points(approx)

                    cv2.drawContours(image,
                                     c,
                                     -1,
                                     (0, 255, 255),
                                     1)
                    cv2.line(image,
                             (src_rect[0][0], src_rect[0][1]),
                             (src_rect[1][0], src_rect[1][1]),
                             color=(100, 255, 100))
                    cv2.line(image,
                             (src_rect[2][0], src_rect[2][1]),
                             (src_rect[1][0], src_rect[1][1]),
                             color=(100, 255, 100))
                    cv2.line(image,
                             (src_rect[2][0], src_rect[2][1]),
                             (src_rect[3][0], src_rect[3][1]),
                             color=(100, 255, 100))
                    cv2.line(image, (src_rect[0][0],
                                     src_rect[0][1]),
                             (src_rect[3][0], src_rect[3][1]),
                             color=(100, 255, 100))

                    # 获取最小矩形包络
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box = box.reshape(4, 2)
                    box = order_points(box)
                    print("approx->box")
                    print(approx)
                    print(src_rect)
                    print(box)
                    w, h = point_distance(box[0], box[1]), point_distance(box[1], box[2])
                    print("w,h=%d,%d" % (w, h))
                    # 透视变换
                    dst_rect = np.array([
                        [0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]],
                        dtype="float32")
                    M = cv2.getPerspectiveTransform(src_rect,
                                                    dst_rect)
                    warped = cv2.warpPerspective(image,
                                                 M,
                                                 (w, h))
                    cv2.imwrite("transfer%d.png" % idx,
                                warped,
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                    break
            else:
                print("failed %d area=%f" % (idx, math.fabs(cv2.contourArea(approx))))
                break

    cv2.imwrite("4-contours.png",
                binary,
                [int(cv2.IMWRITE_PNG_COMPRESSION),
                 9])
    print("\n===寻找轮廓中的闭包，找出面积较大的区块，拟合成四边形，确定四边形的定点========\n")
    cv2.imwrite("5-cut.png",
                image,
                [int(cv2.IMWRITE_PNG_COMPRESSION),
                 9])
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










