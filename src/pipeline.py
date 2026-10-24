"""The complete lane finding pipeline for one image."""
from src import calibration, lane_detection, measurements, perspective, thresholds, visualization


def process_image(img, mtx, dist, scale):
    """Detect the lane on a BGR image; returns the result image and all intermediate steps."""
    undistorted = calibration.undistort(img, mtx, dist)
    warped = perspective.warper(undistorted, scale["src"], scale["dst"])
    binary = thresholds.binary_image(warped)
    lines, histogram = lane_detection.find_lane_lines(binary)
    measured = measurements.measure(lines, scale, binary.shape[0])

    result = visualization.draw_lane(undistorted, lines, scale["src"], scale["dst"])
    visualization.draw_measurements(result, measured["radius"], measured["offset"])

    steps = {"undistorted": undistorted, "warped": warped, "binary": binary,
             "lines": lines, "histogram": histogram, "measurements": measured}
    return result, steps
