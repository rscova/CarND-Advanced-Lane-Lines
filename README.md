# CarND-Advanced-Lane-Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This repository contains my development of the [Project: Advanced Lane Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) proposed by the Udacity's Self-Driving Cars Nanodegree 


<p align="center">
  <img src="output_videos/project_video.gif" alt="project video" />
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
* `output_images/` All media to the writeup_media
* `README.md` Repository's Readme file
* `writeup.md` Project writeup, pipeline and methods explained
* `pipeline_step_by_step.ipynb` Jupyter notebook with the pipeline explained with example images
* `advanced_lane_finding.ipnyb` Implementation of the pipeline in a Jupyter Notebooks
* `advanced_lane_finding.py` Implementation of the pipeline in a Python File
* `requirements.txt
* `License` License File

### Objectives
* Make a pipeline that finds lane lines on the road
* Test it in a short videos from the 280 highway of California, United States
* Reflect my work in a  <A HREF="https://github.com/rscova/CarND-Advanced-Lane-Lines/blob/master/writeup.md" target="_blank"> written report</A>.

###  Requirements
*1.* This project use Python3 version.  If you don't have it installed, check out  https://www.python.org/downloads/ 

*2.* Install the requirement packages.
> pip3 install -r requeriments.txt >




----
In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

