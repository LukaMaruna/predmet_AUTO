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
[image2]: ./output_images/02_undistort_road.jpg "Road Transformed"
[image3]: ./output_images/04_binary.jpg "Binary Example"
[image3a]: ./output_images/03_perspective_points.jpg "Perspective points"
[image4]: ./output_images/03_warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/05_lane_fit.jpg "Fit Visual"
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

Every image and video frame is first corrected with the calibration from the previous section: `calibration.load()` reads the camera matrix and distortion coefficients from `calibration.npz`, and `calibration.undistort()` removes the lens distortion. The example below (`test1.jpg`) is produced by `python main.py images`.

The difference is most visible near the image edges: the guard rail on the left and the car hood at the bottom are straightened. Because every source pixel is kept (alpha = 1), a black border appears around the corrected image. It lies outside the road area used by the next steps.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The binary image is created from the warped (top-down) image, after the perspective transform described in the next section. At that point only the road in front of the car is left in the image, so trees, sky and cars at the side cannot produce false line pixels.

The code is in `src/thresholds.py`. `binary_image()` combines two colour masks with a logical OR:

* `yellow_mask()` converts the image to HSV and keeps pixels in the yellow range with `cv2.inRange()`: hue 15–35, saturation ≥ 80, value ≥ 50. HSV separates colour (hue) from brightness, so yellow lines are found on dark asphalt, on light concrete and partly in shadow.
* `red_channel_mask()` thresholds the red channel with `cv2.threshold()` at 232. White lines have high red, green and blue values, and yellow lines have high red and green, so both are among the brightest pixels in the red channel.

No gradient (Sobel) threshold is used: the two colour masks were enough on all test images.

The thresholds were chosen on the eight test images from the project camera (`straight_lines1-2.jpg`, `test1-6.jpg`):

* Red threshold: white line pixels have a red value of at least 235 on every image. The road surface between the lines reaches about 100 on asphalt, but up to 232 on the light concrete in `test1.jpg` and `test4.jpg`. A threshold of 232 keeps the lines and removes the concrete; at 225 the concrete in `test1.jpg` already adds about 12 000 false pixels.
* Yellow range: a saturation limit of 80 instead of 100 recovers the part of the yellow line in `test4.jpg` that lies in shadow. A value limit of 50 keeps dark brown ground out of the mask. Lowering it to 30 adds noise rather than line pixels.

The hardest image is `test5.jpg`, where tree shadows cover the far part of the yellow line. Both lines are still detected in the lower part of the image.

Example on `test1.jpg` (light concrete):

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In this pipeline the perspective transform is applied right after distortion correction, before the binary image is created, so that thresholding only sees the road in front of the car.

The code is in `src/perspective.py`. The function `warper(img, src, dst)` computes the transform matrix with `cv2.getPerspectiveTransform()` and applies it with `cv2.warpPerspective()`. The output image has the same size as the input. The inverse transform, used later to draw the lane back onto the road, is the same function with `src` and `dst` swapped.

The source and destination points are found once, together with the camera calibration (`python main.py calibrate`), and saved to `perspective.npz`. The pipeline then only loads them.

The code that finds them is in `src/perspective_calibration.py`. It uses the undistorted `straight_lines1.jpg`. On that straight road the two lane lines are parallel, so a section of the lane is a rectangle on the road and a trapezoid in the image. The source points are the corners of that trapezoid, measured on the lane lines in `find_src_points()`:

1. `lane_line_mask()` marks yellow and white pixels using HSV colour ranges (`cv2.inRange`).
2. `line_centres()` goes through every 5th row between y = 440 and y = 630. In each row it takes, on the left and on the right half of the image, the marked pixel closest to the image centre, and the centre of the run of pixels it belongs to.
3. Some of those centres belong to other lines, for example the next lane line on the right where the dashed line has a gap. `fit_line_ransac()` fits x = a·y + b with RANSAC: every pair of points is tried as a candidate line, and the line with the most points within 5 px wins. The final line is a least-squares fit through those points only.
4. The source points are the two fitted lines at y = 450 (far end, where the lines are still sharp) and y = 630 (just above the car hood).

On the left line 36 of 39 rows were used, on the right line 10 of 17. The used points are at most 2 px from the fitted lines, which also confirms that the distortion correction made the lines straight.

In the image below, green dots are used line centres, red dots are rejected ones, blue lines are the fits and the red trapezoid is the result:

![alt text][image3a]

The destination points are not measured but chosen: a rectangle with the lane lines at x = 320 and x = 960, spanning the full image height (`DST` in `perspective_calibration.py`).

| Source | Destination |
|:-------------:|:-------------:|
| 612.7, 450 | 320, 0 |
| 349.2, 630 | 320, 720 |
| 988.1, 630 | 960, 720 |
| 706.0, 450 | 960, 0 |

After the transform both lines of `straight_lines1.jpg` are vertical and parallel:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is in `src/lane_detection.py`. `find_lane_lines()` takes the binary top-down image and runs three steps for each lane line.

**1. Column histogram (`column_histogram()`, `line_bases()`).** For every column of the lower half of the binary image, the number of line pixels is counted. In the top-down view the lane lines are almost vertical, so all pixels of a line fall into a few neighbouring columns and give a clear peak. Only the lower half is used because near the car the lines are almost straight, while further ahead they may bend and the peak would be smeared. The highest peak in the left half of the image is the starting column of the left line, and the highest peak in the right half is the starting column of the right line.

**2. Sliding windows (`sliding_window()`).** Starting at the histogram peak, 9 windows (`NWINDOWS`), each 80 px high and 200 px wide (`MARGIN` = 100 px to each side), are placed on top of each other from the bottom to the top of the image. The line pixels inside a window are assigned to that line. If there are more than 50 of them (`MINPIX`), the next window is centred on their mean x position; otherwise it keeps the position of the previous one. In a curve the windows therefore follow the line.

**3. Polynomial fit (`fit_poly()`).** A second order polynomial is fitted through the collected pixels of each line with `np.polyfit()`:

x = a·y² + b·y + c

x is written as a function of y because the lines are almost vertical: a vertical line cannot be written as y = f(x), since many y values belong to one x value. a describes the curvature (a = 0 is a straight line), b the slope and c the position of the line at the top of the image (y = 0).

Example on `test2.jpg` (left curve). The left panel shows the binary image with the column histogram of its lower half (yellow); its peaks are at x = 365 (left line) and x = 1003 (right line). The right panel shows the sliding windows (green), the pixels of the left and right line (red and blue) and the fitted polynomials (yellow):

```
left line:  x = -2.730e-04·y² + 0.402·y + 229.3
right line: x = -8.869e-05·y² + 0.254·y + 903.3
```

![alt text][image5]

On this image the right line has only two dashes in view, so its polynomial is based on few pixels. In the video this is handled by smoothing the fits over several frames (see the video section).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Both values are computed in `src/measurements.py` (`measure()`), in metres, at the bottom row of the top-down image (y = 719), which is the part of the road closest to the car.

**Pixel to metre conversion.** The conversion factors are measured once by `python main.py calibrate` (`pixel_scale()` in `src/perspective_calibration.py`) on the top-down binary image of `straight_lines1.jpg`, and saved to `perspective.npz`. They use two US highway standards: a lane width of 3.7 m (12 ft) and a dash length of 3 m (10 ft).

* x: the lane lines are at x = 320 and x = 960 in the top-down view, so 640 px correspond to 3.7 m: `xm_per_pix` = 3.7 / 640 = 0.00578 m/px.
* y: `dash_lengths()` measures the dashes of the dashed line in a narrow column band around it. Dashes cut off by the image border are ignored. The two complete dashes are 102 and 91 px long, so `ym_per_pix` = 3.0 / 96.5 = 0.0311 m/px. The top-down image therefore covers about 22 m of road ahead of the car.

As a check, the lane width measured from the fitted lines at the bottom of all eight test images is 3.67–3.99 m.

**Radius of curvature (`radius_of_curvature()`).** The pixel coordinates of each line are multiplied by `ym_per_pix` and `xm_per_pix`, and a second order polynomial x = A·y² + B·y + C is fitted again, now in metres. The radius at the bottom row is

R = (1 + (2·A·y + B)²)^(3/2) / |2·A|

The lane radius is the mean of the left and right line radius.

**Vehicle position (`vehicle_offset()`).** The camera is mounted in the middle of the car, so the vehicle is at the horizontal centre (x = 640) of the camera image. Because the source trapezoid is not exactly symmetric around x = 640, that point is not at x = 640 in the top-down view. `vehicle_x()` maps it there with the perspective matrix (`cv2.perspectiveTransform`), which gives x = 611.3. The lane centre is the mean of the left and right polynomial at the bottom row:

offset = (vehicle_x − lane_centre) · `xm_per_pix`

A positive offset means that the vehicle is right of the lane centre, a negative offset that it is left of it.

Results (`python main.py images`):

| Image | Left line | Right line | Lane radius | Vehicle offset |
|---|---|---|---|---|
| `straight_lines1.jpg` | 2504 m | 6752 m | 4628 m | 0.17 m left |
| `test2.jpg` | 306 m | 943 m | 625 m | 0.56 m left |

On the straight road the radius is several kilometres, which is expected: a straight line has an infinite radius, and a small curvature from measurement noise gives a large but finite value. On the curved road the radius is a few hundred metres, a realistic value for a highway curve. The difference between the left and right line on `test2.jpg` comes from the right line, which has only two dashes in view.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

