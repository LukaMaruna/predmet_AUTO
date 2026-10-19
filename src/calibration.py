"""Camera calibration from chessboard images and distortion correction."""
import cv2
import numpy as np

# Define the chess board inner corners per row and per column
PATTERN_SIZE = (9, 6)
IMAGE_SIZE = (1280, 720)    # (width, height) of the project camera

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)


def load_image(path, image_size=IMAGE_SIZE):
    """Read a BGR image and bring it to the calibrated size (two boards are 1281x721)."""
    img = cv2.imread(str(path))
    if (img.shape[1], img.shape[0]) != image_size:
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img


def find_corners(img, pattern_size=PATTERN_SIZE):
    """Return refined chessboard corners, or None if the full pattern is not visible."""
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # Make sure the chess board pattern was found in the image
    if not ret:
        return None

    # Refine the corner position
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate(image_paths, pattern_size=PATTERN_SIZE, image_size=IMAGE_SIZE):
    """Calibrate the camera from chessboard images."""
    rows, cols = pattern_size

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0). They are the same for all images
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []
    used, skipped = [], []

    # Loop over the image files
    for path in image_paths:
        corners = find_corners(load_image(path, image_size), pattern_size)
        if corners is None:
            skipped.append(path)
            continue

        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)
        used.append(path)

    # Calibrate the camera
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, image_size, None, None)

    # Reproject the board corners with the found parameters and measure the error
    error = 0
    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    return {
        "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs, "rms": rms,
        "error": error / len(objectPointsArray),
        "used": used, "skipped": skipped,
    }


def draw_corners(img, pattern_size=PATTERN_SIZE):
    """Return a copy of img with the detected chessboard corners drawn."""
    out = img.copy()
    corners = find_corners(img, pattern_size)
    cv2.drawChessboardCorners(out, pattern_size, corners, corners is not None)
    return out


def save(path, mtx, dist, rvecs, tvecs):
    np.savez(path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def load(path):
    data = np.load(path)
    return data["mtx"], data["dist"]


def undistort(img, mtx, dist):
    """Remove lens distortion, keeping every source pixel (black border around the result)."""
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
    return undistortedImg
