## Advanced Lane Finding Project

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

[image1]: ./report/camera.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./report/color_fit_lines.jpg "Fit Visual"
[image6]: ./report/out_test1.jpg "Output"
[image7]: ./report/test1.jpg "Road Transformed"
[image8]: ./report/s_channel.jpg "S Channel"
[image9]: ./report/grad.jpg "Gradients"
[image10]: ./report/transforms.jpg "Transforms"
[image11]: ./report/radius.png "Radius"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

##### Writeup / README
You're reading it!

##### Camera Calibration

The code for this step is contained in the file called `camera.py`. The function `calibrate_camera()` returns the camera matrix and distortion coefficients computed from the images contained in the `camera_cal` folder.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

#### Pipeline (single images)

##### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
First, I find the camera calibration and distortion coefficients. In the next step, the coefficients are used in the undistort the input image.
```
mtx, dist = calibrate_camera()
dst = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(dst)
```
After applying the distortion correction using coefficients found during the camera calibration the resulting image is shown below:
![alt text][image7]

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image using the function `color_grad_filters()` contained in the file `project.py`.

The color selection was done in the function `colors.hls_s()` by transforming the input RGB image into HLS color space and selecting the S channel with values between 199 and 255 as specified in `s_thresh`:

```
# Convert to HLS color space and separate the V channel
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
s_channel = hsv[:,:,2]
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
```
Here's an example of my output for this step:
![alt text][image8]

In the next step I applied gradient thresholding using a combination of magnitude and direction thresholds. All functions are contained in the file `sobel.py`.

```
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))

sx_binary = np.zeros_like(dir_binary)
sx_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```
Here's an example of my output for this step:
![alt text][image9]
In the final step, the resulting images from the color and gradient transforms were combined into a single binary image:
````
binary[(s_binary == 1) | (sx_binary == 1)] = 1
````
The resulting image is shown below:
![alt text][image10]
One can see that the combination of color filtering and gradient filtering results in better visibility of both lane lines.

##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 60 through 76 in the file `project.py`.  The `warp()` function takes as input an image and performs a perspective transform using the following source and destination points:

|Location | Source        | Destination   |
|:-------:|:-------------:|:-------------:|
| top right    | 766 500       | 825, 500        |
| bottom right | 1025, 675     | 1025, 675      |
| bottom left  | 275, 675     | 275, 675      |
| top left     | 522, 500      | 675, 500        |


##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

The code for identifing the lane segments is contained in the function `find_segments()` in `project.py`. It uses a sliding window search to look for lane-line pixels. Each window is formed by a vertical subselection of the input image and pixels corresponding to the left lane and right lane are selected.
```
# Step through the windows one by one
for window in range(nwindows):
  ...
  # Identify the nonzero pixels in x and y within the window
  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
  # Append these indices to the lists
  left_lane_inds.append(good_left_inds)
  right_lane_inds.append(good_right_inds)
  ...
```
After pixels corresponding to both lanes are identified a second order polynomial is fit through each of the lane pixels.
```
# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

![alt text][image5]

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating the radius of curvature is containd in the function `compute_radius()` in the file `project.py`. The calculation is based on this formula for finding the curvature:
![alt text][image11]

To make sure the radius is in meters and not only in pixels, the following conversion coefficients were used:
```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
##### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `project.py` in the function `draw_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

More examples of final outputs can be found in the `output_images` directory.

#### Pipeline (video)

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I faced during the implementation is an appropriate selection of parameters used for color filtering, gradient filtering and projective transformation. Since each of these methods requires certain thresholds it is hard to find values that are suitable to every image. My current pipeline is likely to fail under different light settings and road conditions. The pipeline could be improved by using different parameters if none or incorrect lanes are found for a given video frame. Moreover, filtering of the lane coefficients could be done to smooth the final outcome.
