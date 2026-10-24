"""Binary image of lane line pixels: yellow colour mask OR a threshold on the red channel."""
import cv2
import numpy as np

# Yellow lane line colour in HSV (OpenCV hue range is 0-179)
YELLOW_LOWER = np.array([15, 80, 50])
YELLOW_UPPER = np.array([35, 255, 255])

# White and yellow lines are the brightest pixels in the red channel; light concrete
# reaches about 230, white lines are above 235
RED_THRESHOLD = 232


def yellow_mask(img):
    """Pixels whose colour is in the yellow HSV range."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    return mask


def red_channel_mask(img):
    """Pixels whose red channel value is above RED_THRESHOLD."""
    ret, thr = cv2.threshold(img[:, :, 2], RED_THRESHOLD, 255, cv2.THRESH_BINARY)
    return thr


def binary_image(img):
    """Lane line pixels (255), everything else 0."""
    return cv2.bitwise_or(yellow_mask(img), red_channel_mask(img))
