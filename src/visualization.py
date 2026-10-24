"""Drawing the detected lane and the measurements onto the road image."""
import cv2
import numpy as np

from src import perspective
from src.measurements import line_x

LANE_COLOUR = (0, 255, 0)
TEXT_COLOUR = (255, 255, 255)


def draw_lane(undistorted, lines, src, dst):
    """Undistorted image with the area between the two lane lines filled in."""
    ploty = np.arange(undistorted.shape[0])
    left = np.stack([line_x(lines["left"]["fit"], ploty), ploty], axis=1)
    right = np.stack([line_x(lines["right"]["fit"], ploty), ploty], axis=1)

    # Polygon down the left line and back up the right line, drawn in the top-down view
    lane_warped = np.zeros_like(undistorted)
    cv2.fillPoly(lane_warped, [np.vstack([left, right[::-1]]).astype(np.int32)], LANE_COLOUR)

    # Back to the camera view and blended with the road image
    lane = perspective.warper(lane_warped, dst, src)
    return cv2.addWeighted(undistorted, 1, lane, 0.3, 0)


def draw_measurements(img, radius, offset):
    """Write the lane radius and the vehicle offset in the top left corner."""
    side = "right" if offset > 0 else "left"
    texts = [f"Radius of curvature: {radius:.0f} m", f"Vehicle is {abs(offset):.2f} m {side} of centre"]
    for i, text in enumerate(texts):
        cv2.putText(img, text, (60, 110 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOUR, 3)
    return img
