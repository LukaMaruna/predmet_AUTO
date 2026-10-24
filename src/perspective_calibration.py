"""One-time measurement of the perspective transform points on a straight road image."""
import itertools

import cv2
import numpy as np

# Image rows of the trapezoid's top and bottom edge: Y_BOTTOM is just above the car
# hood, Y_TOP is as far ahead as the lane lines are still sharp
Y_TOP = 450
Y_BOTTOM = 630

# Destination rectangle (chosen, not measured): lane lines become vertical at
# x = 320 and x = 960 and span the full image height.
# Point order: top-left, bottom-left, bottom-right, top-right
DST = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

# Lane line colours in HSV (OpenCV hue range is 0-179)
YELLOW_LOWER = np.array([15, 80, 120])
YELLOW_UPPER = np.array([35, 255, 255])
WHITE_LOWER = np.array([0, 0, 190])
WHITE_UPPER = np.array([179, 40, 255])

# A line centre further than this from the fitted line belongs to some other line
INLIER_TOLERANCE = 5    # px

# US highway standards: lane width 12 ft, dash length of a dashed line 10 ft
LANE_WIDTH_M = 3.7
DASH_LENGTH_M = 3.0

# Dashes are measured in a column band of +-DASH_BAND px around a line in the top-down
# view; shorter pixel runs are noise, not dashes
DASH_BAND = 20      # px
MIN_DASH = 30       # px


def lane_line_mask(img):
    """Mask (255) of yellow and white pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    return cv2.bitwise_or(yellow, white)


def line_centres(mask, side):
    """Centre of the lane line closest to the image centre, in every 5th row."""
    height, width = mask.shape
    middle = width // 2
    ys, xs = [], []
    for y in range(Y_TOP - 10, Y_BOTTOM + 1, 5):
        cols = np.nonzero(mask[y])[0]
        cols = cols[cols < middle] if side == "left" else cols[cols >= middle]
        if len(cols) == 0:
            continue

        x = cols.max() if side == "left" else cols.min()
        start, end = x, x
        while start > 0 and mask[y, start - 1]:
            start -= 1
        while end < width - 1 and mask[y, end + 1]:
            end += 1

        ys.append(y)
        xs.append((start + end) / 2)
    return np.array(ys, float), np.array(xs, float)


def fit_line_ransac(ys, xs, tolerance=INLIER_TOLERANCE):
    """Fit x = a*y + b with RANSAC, ignoring points that belong to another line."""
    best = None
    for i, j in itertools.combinations(range(len(ys)), 2):
        a = (xs[j] - xs[i]) / (ys[j] - ys[i])
        b = xs[i] - a * ys[i]
        inliers = np.abs(xs - (a * ys + b)) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers

    a, b = np.polyfit(ys[best], xs[best], 1)
    return a, b, best


def find_src_points(undistorted):
    """Measure the source trapezoid on an undistorted straight road image."""
    mask = lane_line_mask(undistorted)
    lines = {}
    for side in ("left", "right"):
        ys, xs = line_centres(mask, side)
        a, b, inliers = fit_line_ransac(ys, xs)
        residual = np.abs(xs[inliers] - (a * ys[inliers] + b)).max()
        lines[side] = {"ys": ys, "xs": xs, "a": a, "b": b, "inliers": inliers, "residual": residual}

    def x_at(side, y):
        return lines[side]["a"] * y + lines[side]["b"]

    src = np.float32([
        [x_at("left", Y_TOP), Y_TOP],
        [x_at("left", Y_BOTTOM), Y_BOTTOM],
        [x_at("right", Y_BOTTOM), Y_BOTTOM],
        [x_at("right", Y_TOP), Y_TOP],
    ])
    return src, lines


def dash_lengths(binary, x):
    """Lengths of the complete dashes around column x of the binary top-down image, px."""
    rows = (binary[:, int(x) - DASH_BAND:int(x) + DASH_BAND] > 0).any(axis=1)
    lengths, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            # A dash cut off by the top of the image is not complete
            if start > 0 and y - start >= MIN_DASH:
                lengths.append(y - start)
            start = None
    return lengths


def pixel_scale(binary, dst):
    """Metres per pixel in x (lane width) and y (dash length) of the top-down view."""
    xm_per_pix = LANE_WIDTH_M / (dst[2][0] - dst[1][0])
    lengths = dash_lengths(binary, dst[1][0]) + dash_lengths(binary, dst[2][0])
    ym_per_pix = DASH_LENGTH_M / np.mean(lengths)
    return xm_per_pix, ym_per_pix, lengths


def vehicle_x(src, dst, image_width):
    """Column of the vehicle (image centre of the camera view) in the top-down view."""
    M = cv2.getPerspectiveTransform(src, dst)
    point = np.float32([[[image_width / 2, Y_BOTTOM]]])
    return cv2.perspectiveTransform(point, M)[0, 0, 0]


def save(path, **values):
    np.savez(path, **values)


def load(path):
    return dict(np.load(path))
