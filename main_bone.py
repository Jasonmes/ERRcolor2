import numpy as np
import cv2
import preprocessing

image = preprocessing.image
binary = preprocessing.binary

class linedetector:
    def __init__(self):
        self.lines = []

    def find_lines(self, frame):
        h, w, ch = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("binary image", binary)
        cv2.imwrite("D:/Python/opencv/binary.png", binary)
        dist = cv2.distanceTransform(binary, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
        cv2.imshow("distance", dist / 15)
        dist = dist / 15
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
        cv2.imshow("lines", result)
        cv2.imwrite("D:/Python/opencv/skeleton.png", result)

        frame = self.line_fitness(ypts, image=frame)
        frame = self.line_fitness(xpts, image=frame, color=(255, 0, 0))

        cv2.imshow("fit-lines", frame)
        cv2.imwrite("D:/Python/opencv/fitlines.png", frame)
        return self.lines

    def line_fitness(self, pts, image, color=(0, 0, 255)):
        h, w, ch = image.shape
        [vx, vy, x, y] = cv2.fitLine(np.array(pts), cv2.DIST_L1, 0, 0.01, 0.01)
        y1 = int((-x * vy / vx) + y)
        y2 = int(((w - x) * vy / vx) + y)
        cv2.line(image, (w - 1, y2), (0, y1), color, 2)
        return image


if __name__ == "__main__":
    src = cv2.imread('D:/Python/opencv/lines.jpg')
    ld = linedetector()
    lines = ld.find_lines(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()