"""Lane finding pipeline entry point.

    python main.py calibrate    compute camera calibration, perspective points and pixel scale,
                                save calibration.npz and perspective.npz
    python main.py images       run the pipeline steps on test images, save writeup images
"""
import argparse

import cv2
import numpy as np

from src import calibration, perspective, perspective_calibration, pipeline, thresholds
from src.paths import (CALIBRATION_FILE, CAMERA_CAL_DIR, OUTPUT_IMAGES_DIR, PERSPECTIVE_FILE,
                       STRAIGHT_ROAD_IMAGE, TEST_IMAGES_DIR)

# Test images taken with the calibrated camera (the 960x540 images come from another camera)
TEST_IMAGE_NAMES = ["straight_lines1", "straight_lines2"] + [f"test{i}" for i in range(1, 7)]


def side_by_side(left, right, left_title, right_title):
    out = np.hstack([left, right])
    for x, title in [(20, left_title), (left.shape[1] + 20, right_title)]:
        cv2.putText(out, title, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return out


def save_image(name, img):
    OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)
    path = OUTPUT_IMAGES_DIR / name
    cv2.imwrite(str(path), img)
    print(f"Saved {path.relative_to(OUTPUT_IMAGES_DIR.parent)}")


def cmd_calibrate(_args):
    paths = sorted(CAMERA_CAL_DIR.glob("*.jpg"))
    result = calibration.calibrate(paths)
    mtx, dist = result["mtx"], result["dist"]
    calibration.save(CALIBRATION_FILE, mtx, dist, result["rvecs"], result["tvecs"])

    print(f"Used {len(result['used'])}/{len(paths)} images, skipped (pattern not detected): "
          f"{[p.name for p in result['skipped']]}")
    print(f"Reprojection error: RMS {result['rms']:.3f} px, total error (L2 / N) {result['error']:.4f}")
    print(f"Camera matrix:\n{np.array2string(mtx, precision=1, suppress_small=True)}")
    print(f"Distortion [k1 k2 p1 p2 k3]: {np.array2string(dist.ravel(), precision=4)}")
    print(f"Saved {CALIBRATION_FILE.name}")

    # Detected corners on one of the boards used for calibration
    img = calibration.load_image(result["used"][0])
    save_image("01_chessboard_corners.jpg", calibration.draw_corners(img))

    # calibration1 is one of the images skipped during calibration,
    # so it is a convenient image for validating the result.
    img = calibration.load_image(CAMERA_CAL_DIR / "calibration1.jpg")
    out = side_by_side(img, calibration.undistort(img, mtx, dist), "Original", "Undistorted")
    save_image("01_undistort_chessboard.jpg", out)

    # Perspective points, measured on the lane lines of the undistorted straight road image
    undistorted = calibration.undistort(calibration.load_image(STRAIGHT_ROAD_IMAGE), mtx, dist)
    src, lines = perspective_calibration.find_src_points(undistorted)
    dst = perspective_calibration.DST

    for side, line in lines.items():
        print(f"{side} line: {line['inliers'].sum()}/{len(line['ys'])} rows used, "
              f"max distance from fitted line {line['residual']:.1f} px")
    print(f"Source points:\n{np.array2string(src, precision=1)}")
    save_image("03_perspective_points.jpg", draw_perspective_measurement(undistorted, src, lines))

    # Metres per pixel and vehicle position in the top-down view of the same image
    binary = thresholds.binary_image(perspective.warper(undistorted, src, dst))
    xm_per_pix, ym_per_pix, dashes = perspective_calibration.pixel_scale(binary, dst)
    vehicle_x = perspective_calibration.vehicle_x(src, dst, undistorted.shape[1])
    print(f"Dash lengths: {dashes} px")
    print(f"xm_per_pix = {xm_per_pix:.5f} m, ym_per_pix = {ym_per_pix:.5f} m, vehicle at x = {vehicle_x:.1f}")

    perspective_calibration.save(PERSPECTIVE_FILE, src=src, dst=dst, xm_per_pix=xm_per_pix,
                                 ym_per_pix=ym_per_pix, vehicle_x=vehicle_x)
    print(f"Saved {PERSPECTIVE_FILE.name}")


def draw_perspective_measurement(img, src, lines):
    """Measured line centres (green: used, red: rejected), fitted lines and the trapezoid."""
    out = img.copy()
    y_end = perspective_calibration.Y_TOP - 10
    for line in lines.values():
        for y, x, used in zip(line["ys"], line["xs"], line["inliers"]):
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0) if used else (0, 0, 255), -1)
        top = (int(line["a"] * y_end + line["b"]), y_end)
        bottom = (int(line["a"] * out.shape[0] + line["b"]), out.shape[0])
        cv2.line(out, top, bottom, (255, 0, 0), 1)
    cv2.polylines(out, [src.astype(np.int32)], True, (0, 0, 255), 2)
    return out


def draw_histogram(binary, histogram):
    """Binary image with the column histogram of its lower half drawn over it."""
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    height = out.shape[0]
    cv2.line(out, (0, height // 2), (out.shape[1], height // 2), (0, 0, 255), 1)
    scale = (height // 2) / histogram.max()
    points = np.stack([np.arange(len(histogram)), height - histogram * scale], axis=1)
    cv2.polylines(out, [points.astype(np.int32)], False, (0, 255, 255), 2)
    return out


def draw_lane_lines(binary, lines):
    """Windows (green), left/right line pixels (red/blue) and fitted polynomials (yellow)."""
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) // 3
    ploty = np.arange(out.shape[0])
    for line, colour in ((lines["left"], (0, 0, 255)), (lines["right"], (255, 0, 0))):
        out[line["ys"], line["xs"]] = colour
        for x_low, y_low, x_high, y_high in line["windows"]:
            cv2.rectangle(out, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)
        a, b, c = line["fit"]
        plotx = a * ploty ** 2 + b * ploty + c
        cv2.polylines(out, [np.stack([plotx, ploty], axis=1).astype(np.int32)], False, (0, 255, 255), 3)
    return out


def save_step_images(name, img, steps, src, dst):
    """Writeup images of the individual pipeline steps, each from the test image that shows it best."""
    if name == "test1":
        # Distortion correction, and the binary image on light concrete
        save_image("02_undistort_road.jpg", side_by_side(img, steps["undistorted"], "Original", "Undistorted"))
        binary = cv2.cvtColor(steps["binary"], cv2.COLOR_GRAY2BGR)
        save_image("04_binary.jpg", side_by_side(steps["warped"], binary, "Warped", "Binary"))

    if name == STRAIGHT_ROAD_IMAGE.stem:
        # Perspective transform on a straight road: source trapezoid -> destination rectangle
        undistorted, warped = steps["undistorted"].copy(), steps["warped"].copy()
        cv2.polylines(undistorted, [src.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(warped, [dst.astype(np.int32)], True, (0, 0, 255), 3)
        save_image("03_warped_straight_lines.jpg",
                   side_by_side(undistorted, warped, "Source points", "Warped, destination points"))

    if name == "test2":
        # Lane line pixels and polynomial fit on a curved road
        binary, lines = steps["binary"], steps["lines"]
        save_image("05_lane_fit.jpg", side_by_side(draw_histogram(binary, steps["histogram"]),
                                                   draw_lane_lines(binary, lines),
                                                   "Histogram", "Sliding windows and fit"))


def cmd_images(_args):
    mtx, dist = calibration.load(CALIBRATION_FILE)
    scale = perspective_calibration.load(PERSPECTIVE_FILE)

    for name in TEST_IMAGE_NAMES:
        img = calibration.load_image(TEST_IMAGES_DIR / f"{name}.jpg")
        result, steps = pipeline.process_image(img, mtx, dist, scale)
        save_image(f"07_result_{name}.jpg", result)
        save_step_images(name, img, steps, scale["src"], scale["dst"])

        m, lines = steps["measurements"], steps["lines"]
        fits = ", ".join(f"{side} x = {a:.2e}*y^2 + {b:.3f}*y + {c:.1f}"
                         for side, (a, b, c) in ((s, lines[s]["fit"]) for s in ("left", "right")))
        print(f"  {name}: radius left {m['left_radius']:.0f} m, right {m['right_radius']:.0f} m, "
              f"lane {m['radius']:.0f} m; vehicle offset {m['offset']:+.2f} m; {fits}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="compute camera calibration and perspective points").set_defaults(func=cmd_calibrate)
    sub.add_parser("images", help="run the pipeline on test images").set_defaults(func=cmd_images)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
