from collections import namedtuple
import cv2
import numpy as np


class LaneDetector:
    def __init__(self, strategy):
        self._strategy = strategy

    def detect(self, image):
        return self._strategy.algorithm(image)


class FindLane:
    def __init__(self):
        self.Dimension = namedtuple('Dimension', ['width', 'height'], verbose=True)
        self.Point = namedtuple('Point', ['x', 'y'], verbose=True)
        self.Line = namedtuple('Line', ['point_1', 'point_2'], verbose=True)

        self.frame_size = self.Dimension(width=160, height=68)
        self.kernel_size = (5, 5)
        self.left_detector_start = self.Point(0, self.frame_size.height-int(self.frame_size.height/4))
        self.left_detector_end = self.Point(int(self.frame_size.width/3), 
                                            self.frame_size.height-int(self.frame_size.height/4))
        self.right_detector_start = self.Point(int(self.frame_size.width - (self.frame_size.width/3)), 
                                               self.frame_size.height-int(self.frame_size.height/4))
        self.right_detector_end = self.Point(int(self.frame_size.width), 
                                             self.frame_size.height-int(self.frame_size.height/4))
        self.steering_range = int(self.left_detector_start.x + self.frame_size.width/2)

    def draw_overlay(self, image):
        cv2.line(image, self.left_detector_start, self.left_detector_end, (255, 0, 255), 2)
        middle = self.Point(int(self.frame_size.width/2), int(self.right_detector_start.y))
        cv2.line(image, (middle.x, middle.y+25), (middle.x, middle.y-25), (255, 255, 255), 2)
        cv2.line(image, self.right_detector_start, self.right_detector_end, (255, 0, 255), 2)

    def draw_guide_overlay(self, image, line_detector, line):
        x, y = line_intersection(line_detector, line)
        intersection = self.Point(int(x), int(y))
        line = self.Line(self.Point(intersection.x, intersection.y+25), 
                         self.Point(intersection.x, intersection.y-25))
        cv2.line(image, line.point_1, line.point_2, (0, 0, 255), 2)
        return intersection

    def steering_advice(self, image, left_line=None, right_line=None):
        action = "forward"
        if left_line and right_line:
            cv2.line(image, left_line.point_1, left_line.point_2, (100, 100, 255), 2)
            cv2.line(image, right_line.point_1, right_line.point_2, (100, 100, 255), 2)
            left_intersection = self.draw_guide_overlay(image, 
                                                           self.Line(self.left_detector_start, 
                                                           self.left_detector_end), left_line)
            right_intersection = self.draw_guide_overlay(image, 
                                                            self.Line(self.right_detector_start, 
                                                            self.right_detector_end), right_line)
            mid = int(((self.steering_range+left_intersection.x) + (right_intersection.x-self.steering_range)) / 2)
            cv2.line(image, (mid, self.right_detector_start.y+25), (mid, self.right_detector_start.y-25), (150, 255, 255), 2)
            tolerance = mid * 0.10
            if mid > mid + tolerance:
                action = "turn_right"
            elif mid < mid - tolerance:
                action = "turn_left"
            else:
                action = "forward"
        elif left_line:
            cv2.line(image, (left_line.point_1.x, left_line.point_1.y), (left_line.point_2.x, left_line.point_2.y), (100, 100, 255), 2)
            left_intersection = self.draw_guide_overlay(image, self.Line(self.left_detector_start, self.left_detector_end), left_line)
            mid = left_intersection.x + self.steering_range
            cv2.line(image, (mid, self.right_detector_start.y+25), (mid, self.right_detector_start.y-25), (150, 255, 255), 2)
            if mid > int(self.frame_size.width/2 + mid*0.10):
                action = "turn_right"
        elif right_line:
            cv2.line(image, (right_line.point_1.x, right_line.point_1.y), (right_line.point_2.x, right_line.point_2.y), (100, 100, 255), 2)
            right_intersection = self.draw_guide_overlay(image, self.Line(self.right_detector_start, self.right_detector_end), right_line) 
            mid = right_intersection.x - self.steering_range
            cv2.line(image, (mid, self.right_detector_start.y+25), (mid, self.right_detector_start.y-25), (150, 255, 255), 2)
            if mid < int(self.frame_size.width/2 - mid*0.10):
                action = "turn_left"

        return action 

    def algorithm(self, image):
        pass


class Contour(FindLane):
    def algorithm(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.dilate(image, self.kernel_size, iterations=1)
        image = cv2.erode(image, self.kernel_size, iterations=1)
        image = cv2.GaussianBlur(image, self.kernel_size, 0)
        edged = cv2.Canny(image, 50, 150, apertureSize=3)
        (image, contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        breakLoop = False
        left_line = None
        right_line = None
        for contour in contours:
            for x in contour:
                point = self.Point(x[0][0], x[0][1])
                if left_line is None:
                    if point.x >= self.left_detector_start.x and point.x <= self.left_detector_end.x:
                        left_line = self.Line(point, self.Point(point.x, point.y+1))
                if right_line is None:
                    if point.x >= self.right_detector_start.x and point.x <= self.right_detector_end.x:
                        right_line = self.Line(point, self.Point(point.x, point.y+1)) 
                if left_line and right_line:
                    breakLoop = True
                    break
            if breakLoop:
                break

        super().draw_overlay(image)
        return image, super().steering_advice(image, left_line, right_line)


class HoughLineTransform(FindLane):
    def __init__(self, threshold=0, 
                 lowerb=np.array([50, 50, 50], dtype=np.uint8), 
                 upperb=np.array([255, 255, 255], dtype=np.uint8)):
        self.lowerb = lowerb
        self.upperb = upperb
        self.threshold = threshold
        super().__init__()

    def preprocess(self, image):
        mask = cv2.inRange(image, self.lowerb, self.upperb)
        gauss = cv2.GaussianBlur(mask, ksize=self.kernel_size, sigmaX=0)
        canny_edge = cv2.Canny(gauss, 50, 150, apertureSize=3)
        return canny_edge
    
    def get_lines(self, lines):
        breakLoop = False
        left_line = None
        right_line = None
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000*(-b)); 
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b)); 
                    y2 = int(y0 - 1000*(a))
                    xy1 = self.Point(x1, y1)
                    xy2 = self.Point(x2, y2)
                    left_detector = intersect(xy1, xy2, self.left_detector_start, self.left_detector_end)
                    right_detector = intersect(xy1, xy2, self.right_detector_start, self.right_detector_end)
                    if left_detector and left_line is None:
                        left_line = self.Line(xy1, xy2)
                    elif right_detector and right_line is None:
                        right_line = self.Line(xy1, xy2)

                    if left_line and right_line:
                        breakLoop = True
                        break
                if breakLoop:
                    break
        return left_line, right_line

    def algorithm(self, image):
        canny_edge = self.preprocess(image)
        lines = cv2.HoughLines(canny_edge, 1, np.pi / 180, self.threshold)
        super().draw_overlay(image)
        left_line, right_line = self.get_lines(lines)
        return image, super().steering_advice(image, left_line, right_line)


class HoughLineTransformP(HoughLineTransform):
    def get_lines(self, lines):
        breakLoop = False
        left_line = None
        right_line = None
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    xy1 = self.Point(x1, y1)
                    xy2 = self.Point(x2, y2)
                    left_detector = intersect(xy1, xy2, self.left_detector_start, self.left_detector_end)
                    right_detector = intersect(xy1, xy2, self.right_detector_start, self.right_detector_end)
                    if left_detector and left_line is None:
                        left_line = self.Line(xy1, xy2)
                    elif right_detector and right_line is None:
                        right_line = self.Line(xy1, xy2)

                    if left_line and right_line:
                        breakLoop = True
                        break
                if breakLoop:
                    break
        return left_line, right_line

    def algorithm(self, image):
        canny_edge = super().preprocess(image)
        lines = cv2.HoughLinesP(canny_edge, 1, np.pi / 180, 0)
        super().draw_overlay(image)
        left_line, right_line = self.get_lines(lines)
        return image, super().steering_advice(image, left_line, right_line)


# Utility functions
def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_intersection(line1, line2):
    xdiff = (line1.point_1.x - line1.point_2.x, line2.point_1.x - line2.point_2.x)
    ydiff = (line1.point_1.y - line1.point_2.y, line2.point_1.y - line2.point_2.y)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


