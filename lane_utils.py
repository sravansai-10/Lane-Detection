import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[
        (100, height),
        (image.shape[1] - 100, height),
        (image.shape[1] // 2, int(height * 0.6))
    ]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_lane = make_line_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_lane = make_line_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_lane, right_lane

def make_line_points(image, line):
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, left_line, right_line):
    line_image = np.zeros_like(image)
    if left_line is not None and right_line is not None:
        pts = np.array([[left_line[0], left_line[1]],
                        [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]],
                        [right_line[0], right_line[1]]])
        cv2.fillPoly(line_image, [pts], (0, 255, 100))
    for line in [left_line, right_line]:
        if line is not None:
            cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
    return line_image
