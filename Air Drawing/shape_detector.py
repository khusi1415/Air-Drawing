import cv2
import numpy as np


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, points):

        if len(points) < 10:
            return None

        contour = np.array(points, dtype=np.int32)

        perimeter = cv2.arcLength(contour, False)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            False
        )

        corners = len(approx)

        # Triangle
        if corners == 3:
            return "Triangle"

        # Rectangle / Square
        elif corners == 4:

            x, y, w, h = cv2.boundingRect(approx)

            ratio = w / float(h)

            if 0.95 <= ratio <= 1.05:
                return "Square"

            return "Rectangle"

        # Circle
        elif corners > 6:
            return "Circle"

        return "Line"