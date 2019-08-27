# CarND-Advanced-Lane-Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This repository contains my development of the [Project: Advanced Lane Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) proposed by the Udacity's Self-Driving Cars Nanodegree. Here you can see how it works on a real scenario:


<p align="center">
  <img src="output_images/project_video.gif" alt="project video" />
</p>


This project goal is to develop a pipeline to identify the lane boundaries in a real scenario. 

This is the advance version of the *Finding Lane Lines on the Road project* ([GitHub repository](https://github.com/rscova/CarND-LaneLines-P1))

The pipeline is based in 6 steps:
1. Camera Calibration: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Distortion Image Correction: Apply a distortion correction to raw images.
3. Color Spaces and Gradients Thresholds: Use color transforms, gradients, etc., to create a thresholded binary image
4. Perspective transform: Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane lines: Detect lane pixels and fit to find the lane boundary.
6. Determine the lane curvature: Determine the curvature of the lane and vehicle position with respect to center.

Extra:

7. Warp the detected lane boundaries back onto the original image

8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

To achieve that I have implemented two clases: `ImageProcessor()` and `LineFinder()`. And some naive functions.

To understand this repository code, check first the `writeup.md` and `pipeline_step_by_step.ipynb`. These files have the pipeline explained step by step with the results of each part.

---

## Overview

### Project Structure
* `test_images/` Directory with test images
* `output_images/` Directory with the output images
* `README.md` Repository's Readme file
* `writeup.md` Project writeup, pipeline and methods explained
* `pipeline_step_by_step.ipynb` Jupyter notebook with the pipeline explained with example images
* `advanced_lane_finding.ipnyb` Implementation of the pipeline in a Jupyter Notebooks
* `advanced_lane_finding.py` Implementation of the pipeline in a Python File
* `cal_data.p` Calibration pickle data
* `requirements.txt` Install requirements file
* `License` License File

If you want to download the `test_videos/` and `camera_cal` folders with the videos and images to test the pipeline in videos and calibrate the camera from scratch you can do it from [Google Drive]()

### Objectives
* Make a pipeline that finds lane lines on the road
* Test it in a short videos from the 280 highway of California, United States
* Reflect my work in a  <A HREF="https://github.com/rscova/CarND-Advanced-Lane-Lines/blob/master/writeup.md" target="_blank"> written report</A>.

###  Requirements
*1.* This project use Python3 version.  If you don't have it installed, check out  https://www.python.org/downloads/ 

*2.* Install the requirement packages.
> pip3 install -r requirements.txt >


## Documentation
You can find the documentation of this code and the explanation of the pipeline whit the output of each part in the <A HREF="https://github.com/rscova/CarND-Advanced-Lane-Lines/blob/master/writeup.md" target="_blank"> written report</A>.

## License 
This repository is Copyright © 2019 Saul Cova-Rocamora. It is free software, and may be redistributed under the terms specified in the <A HREF="https://github.com/rscova/CarND-Advanced-Lane-Lines/blob/master/LICENSE" target="_blank">LICENSE</A> file.




