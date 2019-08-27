# **Advanced Lane Finding on the Road WRITEUP** 

## REFLECTION

This document is the reflection of the development of the pipeline to find lane lines in a real road, in this case, in the Highway 280 of California.


This document is divided in 3 sections: the first one, is the **pipeline** whit its corresponding experiments. The second one, is the **limitations** of the pipeline. And the last one is the possible **improvements** of the pipeline.

This is the advance version of the Finding Lane Lines on the Road project ([GitHub repository](https://github.com/rscova/CarND-LaneLines-P1))


The pipeline is based in 6 steps and 2 extras:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Advanced Lane Finding Step By Step

### Step 1: Camera Calibration
**1.1 Extract object points and image points for camera calibration**
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

![png](output_images/output_8_1.png)


**1.2 Calibrate and calculate distortion coefficients**
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![png](output_images/output_10_1.png)


### Step 2: Distortion correction
To demonstrate this step,I will describe how I apply the distortion correction to a real road scenario and see the differences between the original image and the undistorted image:

![png](output_images/output_12_1.png)


### Step 3: Color Spaces and Gradients

**3.1 Color Spaces: RGB, HSV and HLS**

The channel S(HLS) and R(RGB) darkened, are the most suitable channels to detect lines. S detects a little bit better the yellow and white marks in different iluminations, but get less information than R dark. Despite, R dark don't take acount the shadows, but it is work worst than S (sometimes) because detects more light. Here there are the channels that it tried to increase the detection: 

![png](output_images/output_16_1.png)

![png](output_images/output_16_2.png)

![png](output_images/output_16_3.png)

![png](output_images/output_16_4.png)

![png](output_images/output_16_5.png)

![png](output_images/output_16_6.png)

![png](output_images/output_16_7.png)


**3.2 Thresholds to Channels R(RGB),S(HLS)**
A simple binarization is enough for the R darked and S channels. But, for a better detection in different illumination conditions I have used the `cv2.THRESH_TRIANGLE` method. So, the best threshold that I found are 100:

![png](output_images/output_18_1.png)


**3.3 Gradients**


```python
# Sobel x
sobelx = cv2.Sobel(dark_r, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 20
thresh_max = 200
gradx = np.zeros_like(scaled_sobelx)
gradx[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1

# Sobel y
sobely = cv2.Sobel(dark_r, cv2.CV_64F, 0, 1) # Take the derivative in x
abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

# Threshold y gradient
thresh_min = 20
thresh_max = 200
grady = np.zeros_like(scaled_sobely)
grady[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1

#Magnitud of the Gradient
grad_mag = np.sqrt(sobelx**2+sobely**2)
scale_factor_mag = np.max(grad_mag)/255 
grad_mag = (grad_mag/scale_factor_mag).astype(np.uint8)

mag_thresh=(30, 100)
mag_binary = np.zeros_like(grad_mag)
mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1

#Direction of the Gradient
grad_dirct = np.arctan2(abs_sobely, abs_sobelx)

dir_thresh=(np.pi/4, 4*np.pi/9)
dir_binary = np.zeros_like(grad_dirct)
dir_binary[(grad_dirct >= dir_thresh[0]) & (grad_dirct <= dir_thresh[1])] = 1

combined = np.zeros_like(dir_binary)
#combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
combined[((mag_binary == 1) & (dir_binary == 1))] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(gradx,cmap='gray')
ax1.set_title('Sobel X', fontsize=15)
ax2.imshow(grady,cmap='gray')
ax2.set_title('Sobel Y', fontsize=15)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(mag_binary,cmap='gray')
ax1.set_title('Magnitude', fontsize=15)
ax2.imshow(dir_binary,cmap='gray')
ax2.set_title('Directions', fontsize=15)

plt.figure(figsize=(7,7))
plt.imshow(combined,cmap='gray')
plt.title("Combined", fontsize=15)


```




    Text(0.5, 1.0, 'Combined')




![png](output_images/output_20_1.png)



![png](output_images/output_20_2.png)



![png](output_images/output_20_3.png)


**3.4 Color and Gradient**


```python
# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack((binary_r, binary_s, gradx)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(binary_r)
combined_binary[((binary_r == 1) & ((binary_s == 1) | (gradx == 1)))] = 1

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')

```




    <matplotlib.image.AxesImage at 0x7fc98f9d0828>




![png](output_images/output_22_1.png)


### Step 4: Perspective transform (Bird's Eye)


```python
# define 4 source points
src_points = [(193,720), (577,460),(705,460),(1126,720)]

# define 4 destination points
dst_points = [(320,720), (320,0)  ,(960,0)  ,(960,720)]

undist_roi = undist.copy()
cv2.polylines(undist_roi,np.array([src_points],dtype=np.int32),True,(255,0,0),4)

# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))

# Warp the image 
warped = cv2.warpPerspective(undist, M, img_size)
rgb_warp = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
cv2.imwrite("prueba.jpg",rgb_warp)
warped_binary = cv2.warpPerspective(combined_binary, M, img_size)
cv2.imwrite("prueba2.jpg",warped_binary*255)

warped_roi = warped.copy()
cv2.polylines(warped_roi,np.array([dst_points],dtype=np.int32),True,(255,0,0),4)

warped_binary = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, np.ones((3,3)))
warped_binary = cv2.morphologyEx(warped_binary, cv2.MORPH_TOPHAT, np.ones((2,65)))


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,25))
ax1.imshow(undist_roi)
ax1.set_title('Undistorted ROI', fontsize=15)
ax2.imshow(warped_roi)
ax2.set_title('Warped Image', fontsize=15)
ax3.imshow(warped_binary, cmap='gray')
ax3.set_title('Warped Binary Image', fontsize=15)

```




    Text(0.5, 1.0, 'Warped Binary Image')




![png](output_images/output_24_1.png)


### Step 5: Detect lane lines

**5.1 Compute Histogram and Locate Lane Lines**

Take the bottom half of the image to compute the histogram, thus, the lane lines are more vertical.

Compute the histogram, the peaks will be regions with more probability to be a lane line.

Divide the histogram in to equal parts and get the maximum peak of each side. The peaks are the first position of the left and right lines.


```python
# Take a histogram of the bottom half of the image
# Lane lines are likely to be mostly vertical nearest to the car
histogram = np.sum(warped_binary[warped_binary.shape[0]//2:,:], axis=0)
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)

leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
print(leftx_base,rightx_base)
plt.plot(histogram)


```

    328 945





    [<matplotlib.lines.Line2D at 0x7fc99e904f98>]




![png](output_images/output_27_2.png)


Updated version using bins to agroup cols of pixels


```python
pix_per_bin = 40
windows = warped_binary.shape[1]//pix_per_bin
hist = []
for window in range(windows):
    w_low = window * pix_per_bin
    w_high = (window +1) * pix_per_bin
    hist.append(np.max(np.sum(warped_binary[:,w_low:w_high], axis=0)))

#print(hist)
hist = np.array(hist)
plt.plot(hist)

leftx_base = (np.argmax(hist[:windows//2]) * pix_per_bin) + pix_per_bin//2
rightx_base = (np.argmax(hist[windows//2:]) + warped_binary.shape[1]//pix_per_bin//2) * pix_per_bin + pix_per_bin//2
print(leftx_base,rightx_base)


```

    340 940



![png](output_images/output_29_1.png)


**5.2 Sliding Windows and Fit a Polynomial**

As shown in the previous animation, we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go.


```python
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 70
# Set minimum number of pixels found to recenter window
minpix = 300

# Create an output image to draw on and visualize the result
out_img = np.dstack((warped_binary, warped_binary, warped_binary))
    
# Set height of windows - based on nwindows above and image shape
window_height = np.int(warped_binary.shape[0]//nwindows)
# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
nonzero = warped_binary.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Current positions to be updated later for each window in nwindows
leftx_current = leftx_base
rightx_current = rightx_base

print(leftx_current,rightx_current)
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low  = warped_binary.shape[0] - (window+1)*window_height
    win_y_high = warped_binary.shape[0] - window*window_height
    
    # Find the four below boundaries of the window ###
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),
    (win_xleft_high,win_y_high),(0,255,0), 3) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),
    (win_xright_high,win_y_high),(0,255,0), 3) 

    # Identify the nonzero pixels in x and y within the window
    good_left_inds  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # If you found > minpix pixels, recenter next window
    if good_left_inds.shape[0] > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if good_right_inds.shape[0] > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    

# Concatenate the arrays of indices (previously was a list of lists of pixels)
try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
except ValueError:
    # Avoids an error if the above is not implemented fully
    pass

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each line
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
try:
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
except TypeError:
    # Avoids an error if `left` and `right_fit` are still none or incorrect
    print('The function failed to fit a line!')
    left_fitx = 1*ploty**2 + 1*ploty
    right_fitx = 1*ploty**2 + 1*ploty

## Visualization ##
# Colors in the left and right lane regions
out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

# Plots the left and right polynomials on the lane lines
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')

plt.imshow(out_img)
```

    340 940





    <matplotlib.image.AxesImage at 0x7fc98fcb0400>




![png](output_images/output_31_2.png)


**5.3 Search from Prior**

Using the full algorithm from before and starting fresh on every frame may seem inefficient, as the lane lines don't necessarily move a lot from frame to frame.

In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous lane line position, like in the above image. The green shaded area shows where we searched for the lines this time. So, once you know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame.


```python
# HYPERPARAMETER
# Choose the width of the margin around the previous polynomial to search
# The quiz grader expects 100 here, but feel free to tune on your own!
margin = 70

# Grab activated pixels
nonzero = warped_binary.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Set the area of search based on activated x-values
# within the +/- margin of our polynomial function
left_lane_inds  = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) &
                   (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                        left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                        right_fit[2] - margin)) &
                   (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                        right_fit[2] + margin)))

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each line
left_fit = np.polyfit(lefty,leftx,2)
right_fit = np.polyfit(righty,rightx,2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])

# Calc both polynomials
left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

## Visualization ##
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                          ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))

right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                          ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

# Plot the polynomial lines onto the image
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
## End visualization steps ##

plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7fc98fd285f8>




![png](output_images/output_33_1.png)


### Step 6: Determine the lane curvature

**6.1 Radius of Curvature**

The radius of curvature <A HREF="https://www.intmath.com/applications-differentiation/8-radius-curvature.php" target="_blank">(awesome tutorial)</A> at any point $x$ of the function $x = f(y)$, where $f(y)$ is a second order polynomial, is given as follows:

$$R_{curve} = \frac{(1+(2Ay+B)^{2})^{3/2}}{|2A|}$$

Where:

$$ f'(y) = \frac{dx}{dy} = 2Ay + B  $$
$$ f''(y) = \frac{d^2 x}{d^2 y} = 2A  $$


```python
# Define y-value where we want radius of curvature
# We'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)

# Calculation of radius of curvature
# y = ax² + bx + c; y'=2ax + b; y''= 2a

left_curverad  = ((1+(2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2)) / np.absolute(2*left_fit[0])
right_curverad = ((1+(2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2)) / np.absolute(2*right_fit[0])

print(left_curverad, right_curverad)
```

    45668.33472657573 27554.843613629633


**6.2 From Pixels to Real-World**

We've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So we actually need to repeat this calculation after converting our x and y values to real world space.

This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide.

Let's say that the camera image has 720 relevant pixels in the y-dimension (remember, the image is perspective-transformed!), and we'll say roughly 700 relevant pixels in the x-dimension.

Once the parameters of the parabole ($x = ay^2 + by + c$) are calculated, this formula can convert the coeficients from pixels to meters:
$$ x = \frac {mx}{my^2} ay^2 + \frac{mx}{my}by + c$$



```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

y_eval = y_eval * ym_per_pix

# Calculation of radius of curvature in meters
A = (xm_per_pix / (ym_per_pix ** 2))* left_fit[0]
B = (xm_per_pix / ym_per_pix) * left_fit[1]
left_curverad  = ((1+(2*A + B)**2)**(3/2)) / np.absolute(2*A)

A = (xm_per_pix / (ym_per_pix ** 2))* right_fit[0]
B = (xm_per_pix / ym_per_pix) * right_fit[1]
right_curverad  = ((1+(2*A + B)**2)**(3/2)) / np.absolute(2*A)

mean_curvature = (left_curverad + right_curverad)/2

curvature_string = "Radius of curvature: %.2f km" % (abs(mean_curvature)/1000.0)
print(curvature_string)
```

    Radius of curvature: 12.01 km


**6.3 Car lane center offset**


```python
lane_center = (rightx[719] + leftx[719])/2
center_offset_pixels = abs(img_size[0]/2 - lane_center)
center_offset_mtrs = xm_per_pix*center_offset_pixels
offset_string = "Center offset: %.2f m" % center_offset_mtrs
print(offset_string)
```

    Center offset: 0.06 m


### Plot Everythig together


```python
warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
M = cv2.getPerspectiveTransform(np.float32(dst_points), np.float32(src_points))
newwarp = cv2.warpPerspective(color_warp, M,(color_warp.shape[1],color_warp.shape[0]))
# Combine the result with the original image
final = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


if (abs(mean_curvature) >= 5000):
    lane_string = "Straight Lane"
elif (mean_curvature >= 0 and mean_curvature < 5000):
    lane_string = "Right Curve"
elif (mean_curvature < 0 and mean_curvature > -5000):
    lane_string = "Left Curve"

cv2.putText(final,curvature_string,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
cv2.putText(final,offset_string,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
cv2.putText(final,lane_string,(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            

plt.imshow(final)
```




    <matplotlib.image.AxesImage at 0x7f5546229e10>




![png](output_images/output_43_1.png)

