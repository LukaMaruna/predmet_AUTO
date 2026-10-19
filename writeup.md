## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/01_chessboard_corners.jpg "Detected corners"
[image1]: ./output_images/01_undistort_chessboard.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code is in `src/calibration.py` (functions `find_corners()`, `calibrate()`, `undistort()`) and is run with `python main.py calibrate`, which saves the result to `calibration.npz`.

The calibration images in `camera_cal/` show a chessboard with 10×7 squares, i.e. **9×6 inner corners** (points where four squares meet). For every image:

1. The image is converted to grayscale and `cv2.findChessboardCorners()` looks for all 9×6 inner corners.
2. If they are found, `cv2.cornerSubPix()` refines each corner position to sub-pixel accuracy.
3. The refined corners are appended to `imgPointsArray` (*image points*: where the corners appear in the image, in pixels), and a copy of `objectPoints` is appended to `objectPointsArray` (*object points*: where the same corners are on the physical board).

The object points are the same for every image: a regular grid (0,0,0), (1,0,0), …, (8,5,0) in units of one square. The board is flat, so z = 0 for every corner.

Two of the images are 1281×721 instead of 1280×720, so they are resized to 1280×720 before corner detection. The camera matrix only applies to a single image size.

The pattern was detected in 17 of the 20 images. In `calibration1.jpg` and `calibration5.jpg` part of the board is outside the frame. In `calibration4.jpg` all corners are visible, but the board touches the image border and the detector does not recognise it. Those three images are skipped.

`cv2.calibrateCamera()` then takes the 17 pairs of object/image point sets and finds the camera matrix (focal length and optical centre) and the distortion coefficients that best map the board grid onto the detected corners.

To check the result, the board corners are projected back into each image with `cv2.projectPoints()`, using the calibrated camera and the board pose returned for that image (`rvecs`, `tvecs`), and compared with the detected corners. The RMS distance is about one pixel.

Detected corners on `calibration10.jpg`:

![alt text][image0]

To remove distortion, `undistort()` computes a new camera matrix with `cv2.getOptimalNewCameraMatrix()` (alpha = 1, so every source pixel is kept and a black border appears around the corrected image) and applies `cv2.undistort()`.

The example below uses `calibration1.jpg`, one of the three images that were skipped during calibration. Since the calibration never saw it, it is a convenient image for validating the result: the curved board edges on the left become straight on the right.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

TODO: Add your text here!!!

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

