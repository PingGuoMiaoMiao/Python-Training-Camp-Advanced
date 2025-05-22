import cv2
import numpy as np

def contour_detection(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours_info = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours_info) == 3:
            _, contours, _ = contours_info
        else:
            contours, _ = contours_info

        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

        return img_contours, list(contours)
    except Exception as e:
        print(f"Error: {e}")
        return None, None
