"""Lane finding pipeline entry point.

    python main.py calibrate    compute camera calibration, save calibration.npz
"""
import argparse

import cv2
import numpy as np

from src import calibration
from src.paths import CALIBRATION_FILE, CAMERA_CAL_DIR, OUTPUT_IMAGES_DIR


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="compute camera calibration").set_defaults(func=cmd_calibrate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
