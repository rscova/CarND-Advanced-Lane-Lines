#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import glob
import pickle
import sys

from moviepy.editor import VideoFileClip
from IPython.display import HTML


def getImageChannels(image,space='RGB'):
    """
    Split the image in channels
    :param image: Source image
    :param space: Current Color space of the image
    :return: The channels of the image
    """
    if (space == 'RGB'):
        return cv2.split(image)
    elif (space == 'HSV'):
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    elif (space == 'HLS'):
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
    elif (space == 'LAV'):
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    elif (space == 'LUV'):
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LUV))
    elif (space == 'YCrCb'):
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb))

def darkenImage(image,gamma=1.0):
    """
    Apply gamma correction using the lookup table
    :param image: Source image
    :param gamma: Gamma value
    :return: Corrected Image
    """
    gamma_inverse = 1.0 / gamma
    lut_table = np.array([((i / 255.0) ** gamma_inverse) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, lut_table)

def binarizeImage(image,thresh=100,method=cv2.THRESH_BINARY):
    """
    Binarize an image using the threshold and the method
    :param image: Source image
    :param thresh: Threshold value to binarizate
    :param method: Use diverse opencv methods to binarizate: Normal, Otsu, triangle.
    :return: Binary Image
    """
    ret, binary = cv2.threshold(image,thresh,1,method)
    return binary

def getSobelGradient(image,orientation='x',thresh=(0,255)):
    """
    Compute the Soble's Gradient
    :param image: Source image
    :param orientation: Gradient in X or Y
    :param thresh: Threshold values to binarizate, min and max
    :return: binary gradient &  absolute value
    """ 
    if (orientation == 'x'):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)   
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad = np.zeros_like(scaled_sobel)
    grad[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad,abs_sobel

def getGradientMagnitude(image,grad_x,grad_y,thresh=(0,255)):
    """
    Compute the Magnitude of the gradient
    :param image: Source image
    :param grad_x: Gradient in X 
    :param grad_y: Gradient in Y
    :param thresh: Threshold values to binarizate, min and max
    :return: Magnitud of the Gradient Image
    """ 
    grad_mag = np.sqrt(grad_x**2+grad_y**2)
    scale_factor_mag = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor_mag).astype(np.uint8)

    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1
    return mag_binary

def getGradientDirection(image,abs_sobel_x,abs_sobel_y,thresh=(0,np.pi/2)):
    """
    Compute the Direction of the gradient
    :param image: Source image
    :param abs_sobel_x: Absolute Sobel in X 
    :param abs_sobel_y: Absolute Sobel in Y
    :param thresh: Threshold values to binarizate, min and max
    :return: Direction of the Gradient Image
    """ 
    grad_dirct = np.arctan2(abs_sobel_y, abs_sobel_x)

    dir_binary = np.zeros_like(grad_dirct)
    dir_binary[(grad_dirct >= thresh[0]) & (grad_dirct <= thresh[1])] = 1
    return dir_binary

def transformPerspective(image,src_points,dst_points):
    """
    Transform Image Perspective to another image (Bird's eye for example)
    :param image: Source image
    :param src_points: Source Points of the Image
    :param dst_points: New position of the source points in the new Image
    :return: Transformed Image
    """ 
    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    return  cv2.warpPerspective(image, M,(image.shape[1],image.shape[0]))

def drawROI(image,points):
    """
    Draw A Polygonal Region (Region Of Interest) in the image
    :param image: Source image
    :param points: Source Points of the polygon
    :return: Drawed Image
    """ 
    result = image.copy()
    return cv2.polylines(result,np.array([points],dtype=np.int32),True,(255,0,0),4)

class ImageProcessor:
    """
    Class that incorporate the high-level functions to process 
    the raw image into a binary image to compute the lane
    """ 
    def __init__(self,img_shape=(720,1280,3)):
        """
        Class constructor, all the class variables are defined here 
        :param self: referenc of the class instance
        :param img_shape: shape of the image
        """
        
        self.img_shape_ = img_shape
        self.img_size_  = (img_shape[1], img_shape[0])
        
        self.mtx_  = []
        self.dist_ = []
        
        self.gamma_ = 0.7
        
        self.r_channel_thresh_ = 100
        self.s_channel_thresh_ = 150
        self.grad_x_thresh_    = (20,200)
        self.grad_y_thresh_    = (20,200)
        self.grad_mag_thresh_  = (10, 100)
        self.grad_dir_thresh_  = (5*np.pi/18, 4*np.pi/9)
    
        self.src_points_ = [(193,720), (577,460),(705,460),(1126,720)]#[(220, 720),(570, 500),(722 , 500),(1110, 720)]#
        self.dst_points_ = [(320,720), (320,0)  ,(960,0)  ,(960,720)]#[(320, 720),(410, 1),(790,1),(920 , 720)]#

    def readImage(self,img_path):
        """
        Image reader and updates the img_shape_ variable
        :param self: referenc of the class instance
        :param img_path: path of the image
        :return: Image
        """
        image = cv2.imread(img_path)
        self.img_shape_ = image.shape
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def cameraCalibration(self,img_paths="camera_cal/", row_corners=9, col_corners=6):
        """
        Calibration of the camera, it uses a chessboard pattern 
        Get the camera matrix (mtx_) and vector of distortion coefficients(dist_)
        :param self: referenc of the class instance
        :param img_paths: path of the images
        :param row_corners: chessboard inside row corners
        :param col_corners: chessboard inside col corners
        """
        objp = np.zeros((col_corners*row_corners,3), np.float32)
        objp[:,:2] = np.mgrid[0:row_corners, 0:col_corners].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        
        img_calibration_paths = glob.glob(img_paths + "*.jpg")

        for idx,img_cal in enumerate(img_calibration_paths):
            image = cv2.imread(img_cal)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (row_corners,col_corners), None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if DRAW_IMAGES:
                    cv2.drawChessboardCorners(image, (row_corners,col_corners), corners, ret)
                    plt.subplot(5,4,idx)
                    plt.imshow(image)  

        ret, self.mtx_, self.dist_, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,self.img_size_,None,None)
        

    
    def saveCalibrationData(self,path="camera_cal/cal_data.p"):
        """
        Save the camera matrix (mtx_) and vector of distortion coefficients(dist_)
        :param self: referenc of the class instance
        :param path: path to save data
        """
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx_
        dist_pickle["dist"] = self.dist_
        pickle.dump( dist_pickle, open(path + "cal_data.p", "wb" ) )
        
    def getCalibrationData(self,path_name="camera_cal/cal_data.p"):
        """
        Get the camera matrix (mtx_) and vector of distortion coefficients(dist_)
        from a saved file
        :param self: referenc of the class instance
        :param path_name: path to get data
        """
        file = open(path_name, 'rb')
        data = pickle.load(file)
        file.close()
        self.mtx_ = data['mtx']
        self.dist_ = data['dist']
        
    def correctImageDistortion(self,image):
        """
        Correct the image distortion
        :param self: referenc of the class instance
        :param image: Source image
        :return Undistorted Image
        """
        self.img_shape_ = image.shape
        return cv2.undistort(image, self.mtx_, self.dist_, None, self.mtx_)
    
    def applyColorAndGradient(self,image):
        """
        High Level function to apply color and gradient thresholds
        :param self: referenc of the class instance
        :param image: Source image
        :return Binary Image combined with the methods
        """
        r_channel, _, _ = getImageChannels(image,space='RGB')
        _, _, s_channel = getImageChannels(image,space='HLS')
        
        r_channel = darkenImage(r_channel,self.gamma_)
        
        r_binary = binarizeImage(r_channel,self.r_channel_thresh_,cv2.THRESH_BINARY)
        s_binary = binarizeImage(s_channel,self.r_channel_thresh_,cv2.THRESH_BINARY)
        
        s_binary_auto = binarizeImage(s_channel,self.s_channel_thresh_,cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        
        grad_x,_ = getSobelGradient(r_channel,'x',self.grad_x_thresh_)

        combined_binary = np.zeros_like(r_binary)

        combined_binary[((r_binary == 1) & ((s_binary == 1) | (s_binary_auto == 1) | (grad_x == 1)))] = 1
        return combined_binary
        
        
    def restorePerspective(self,warped,undist,left_fitx,right_fitx,ploty):
        """
        Restore the bird's eye image and draw the lane
        :param self: referenc of the class instance
        :param warped: Birds eye image
        :param undist: Unditorted image
        :param left_fitx: Left line 
        :param right_fitx: Right line 
        :param ploty: Vector of Y values
        :return Restored Image
        """
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = transformPerspective(color_warp,self.dst_points_,self.src_points_)
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        return result


class LaneFinder:
    """
    Class that finds the lane lines using the ImageProcessor Class
    and some methods to fit the lane marks into a curve.
    """ 
    def __init__(self,image_shape=(720,1280,3)):
        """
        Class constructor, all the class variables are defined here 
        :param self: referenc of the class instance
        :param img_shape: shape of the image
        """
        self.img_processor_ = ImageProcessor()
        self.img_processor_.getCalibrationData()
        
        self.image_shape_ = image_shape
        
        self.number_sliding_windows_  = 9
        self.windows_width_           = 70
        self.window_height_           = np.int(self.image_shape_[0]//self.number_sliding_windows_)
        self.windows_min_pixels_      = 70
        
        self.filter_order_ = 15
        
        self.left_x_base_  = 0
        self.right_x_base_ = 0
        
        self.current_left_fit_  = None
        self.current_right_fit_ = None
        
        self.filtered_left_fit_  = []
        self.filtered_right_fit_ = []
        
        self.left_fit_x_         = None
        self.right_fit_x_        = None
        self.mean_fit_x_         = None
        self.plot_y_             = np.linspace(0, image_shape[0]-1, image_shape[0])
        
        self.y_max_curvature_     = 0
        self.center_lane_offset_  = 0
        self.curvature_           = 0
        
        self.need_deep_search_   = True
        self.use_last_fit_       = False
        
        self.ym_per_pix_ = 15.0/260
        self.xm_per_pix_ = 3.7/600
        
    def locateBaseLines(self,image,pix_per_bin=40):
        """
        Compute the histogram of the image to locate the Left and Right base lines
        :param self: referenc of the class instance
        :param image: Source image
        :param pix_per_bin: Number of pixels for each bin
        """
        self.image_shape_ = image.shape

        windows = self.image_shape_[1]//pix_per_bin
        hist = []
        for window in range(windows):
            w_low = window * pix_per_bin
            w_high = (window +1) * pix_per_bin
            hist.append(np.max(np.sum(image[:,w_low:w_high], axis=0)))
        
        self.left_x_base_ = (np.argmax(hist[:windows//2]) * pix_per_bin) + pix_per_bin//2
        self.right_x_base_ = (np.argmax(hist[windows//2:]) + self.image_shape_[1]//pix_per_bin//2) * pix_per_bin + pix_per_bin//2
        
        n_pix_left = np.max(hist[:windows//2])
        n_pix_right = np.max(hist[windows//2:])
        
        dist_base_lines = self.right_x_base_-self.left_x_base_
        
        if (n_pix_left < 50 or n_pix_right < 50):
            self.use_last_fit_ = True
        
        elif (dist_base_lines < 300 or dist_base_lines > 900):
            self.use_last_fit_ = True
        
        else:
            self.use_last_fit_ = False
        
    def deepLineSearch(self,image):
        """
        Search the line using a deep method, creating sliding windows
        Fit the lane pixels in a polynomial curve
        :param self: referenc of the class instance
        :param image: Source image
        """
        self.need_deep_search_ = False    
        out_img = np.dstack((image, image, image))
        out_img = out_img * 255
        
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.number_sliding_windows_):
            win_y_low  = image.shape[0] - (window + 1 ) * self.window_height_
            win_y_high = image.shape[0] - window * self.window_height_

            win_xleft_low = self.left_x_base_ - self.windows_width_
            win_xleft_high = self.left_x_base_ + self.windows_width_
            win_xright_low = self.right_x_base_ - self.windows_width_
            win_xright_high = self.right_x_base_ + self.windows_width_

            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 5) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 5) 

            good_left_inds  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if (good_left_inds.shape[0] > self.windows_min_pixels_):
                self.left_x_base_ = np.int(np.mean(nonzerox[good_left_inds]))
            if (good_right_inds.shape[0] > self.windows_min_pixels_):        
                self.right_x_base_ = np.int(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        leftx  = nonzerox[left_lane_inds]
        lefty  = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if (len(leftx) < 3000 or len(rightx) < 3000):
            self.need_deep_search_ = True
            self.use_last_fit_ = True
            return None
        
        try:
            self.current_left_fit_ = np.polyfit(lefty,leftx,2)
            self.current_right_fit_ = np.polyfit(righty,rightx,2)
            
            self.left_fit_x_ = self.current_left_fit_[0]*self.plot_y_**2 + self.current_left_fit_[1]*self.plot_y_ + self.current_left_fit_[2]
            self.right_fit_x_ = self.current_right_fit_[0]*self.plot_y_**2 + self.current_right_fit_[1]*self.plot_y_ + self.current_right_fit_[2]
                
        except:
            self.need_deep_search_ = True
            self.use_last_fit_ = True
            pass
        
        left_slope_bottom, right_slope_bottom, left_slope_top, right_slope_top = self.getSlopes()
        
        abs_diff_slope_bottom = abs(right_slope_bottom - left_slope_bottom)
        abs_diff_slope_top= abs(right_slope_top - left_slope_top)
        
        if (abs_diff_slope_bottom > 0.2 or  abs_diff_slope_top > 0.2):
            self.need_deep_search_ = True
            self.use_last_fit_ = True

        elif (np.in1d(self.right_fit_x_.astype(int),self.left_fit_x_.astype(int)).any()):
            self.need_deep_search_ = True
            self.use_last_fit_ = True
        
        elif(self.need_deep_search_ == False):
            self.linesFilter()
            
            self.left_fit_x_ = self.current_left_fit_[0]*self.plot_y_**2 + self.current_left_fit_[1]*self.plot_y_ + self.current_left_fit_[2]
            self.right_fit_x_ = self.current_right_fit_[0]*self.plot_y_**2 + self.current_right_fit_[1]*self.plot_y_ + self.current_right_fit_[2]

            self.y_max_curvature_ = np.max(self.plot_y_)
            
            self.need_deep_search_ = False
                
        
    
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        
        
        return out_img
            
    def smartLineSearch(self,image):
        """
        Search the line using a smart method, searching just in the most probably
        Fit the lane pixels in a polynomial curve
        :param self: reference of the class instance
        :param image: Source image
        """
        self.need_deep_search_ = False
        out_img = np.dstack((image, image, image))
        out_img = out_img * 255
        
        nonzero  = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds  = ((nonzerox > (self.current_left_fit_[0]*(nonzeroy**2) + self.current_left_fit_[1]*nonzeroy + 
                                self.current_left_fit_[2] - self.windows_width_)) &
                           (nonzerox < (self.current_left_fit_[0]*(nonzeroy**2) + self.current_left_fit_[1]*nonzeroy +
                                self.current_left_fit_[2] + self.windows_width_)))
        right_lane_inds = ((nonzerox > (self.current_right_fit_[0]*(nonzeroy**2) + self.current_right_fit_[1]*nonzeroy +
                                self.current_right_fit_[2] - self.windows_width_)) &
                           (nonzerox < (self.current_right_fit_[0]*(nonzeroy**2) + self.current_right_fit_[1]*nonzeroy +
                                self.current_right_fit_[2] + self.windows_width_)))
        
        leftx  = nonzerox[left_lane_inds]
        lefty  = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if (len(leftx) < 2500 or len(rightx) < 2500):
            self.need_deep_search_ = True
            return None
        elif  (len(leftx) > 35000 or len(rightx) > 35000):
            self.need_deep_search_ = True
            return None
        
        try:
            self.current_left_fit_ = np.polyfit(lefty,leftx,2)
            self.current_right_fit_ = np.polyfit(righty,rightx,2)
            
            self.left_fit_x_ = self.current_left_fit_[0]*self.plot_y_**2 + self.current_left_fit_[1]*self.plot_y_ + self.current_left_fit_[2]
            self.right_fit_x_ = self.current_right_fit_[0]*self.plot_y_**2 + self.current_right_fit_[1]*self.plot_y_ + self.current_right_fit_[2]

        except:
            self.need_deep_search_ = True
            pass
        
        
        left_slope_bottom, right_slope_bottom, left_slope_top, right_slope_top = self.getSlopes()
        abs_diff_slope_bottom = abs(right_slope_bottom - left_slope_bottom)
        abs_diff_slope_top= abs(right_slope_top - left_slope_top)

        if (abs_diff_slope_bottom > 0.2 or  abs_diff_slope_top > 0.2):
            self.need_deep_search_ = True
            
        elif (np.in1d(self.right_fit_x_.astype(int),self.left_fit_x_.astype(int)).any()):
            self.need_deep_search_ = True
        
        elif(np.max(self.right_fit_x_) >= 1280 or np.max(self.left_fit_x_) <= 0):
            self.need_deep_search_ = True
        
        elif(self.need_deep_search_ == False):
            self.linesFilter()
            
            self.left_fit_x_ = self.current_left_fit_[0]*self.plot_y_**2 + self.current_left_fit_[1]*self.plot_y_ + self.current_left_fit_[2]
            self.right_fit_x_ = self.current_right_fit_[0]*self.plot_y_**2 + self.current_right_fit_[1]*self.plot_y_ + self.current_right_fit_[2]

            self.y_max_curvature_ = np.max(self.plot_y_)
        
            
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fit_x_-self.windows_width_, self.plot_y_]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fit_x_+self.windows_width_, 
                                  self.plot_y_])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fit_x_-self.windows_width_, self.plot_y_]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fit_x_+self.windows_width_,self.plot_y_])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        mid_line_window1 = np.array([np.transpose(np.vstack([self.mean_fit_x_-3, self.plot_y_]))])
        mid_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.mean_fit_x_+3,self.plot_y_])))])
        mid_line_pts = np.hstack((mid_line_window1, mid_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([mid_line_pts]), (255,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        return out_img
            
    def laneCurvature(self):
        """
        Compute the lane curvature using the mean line of the lane.
        :param self: reference of the class instance
        """
        self.mean_fit_ = (self.current_right_fit_ + self.current_left_fit_)/2 
        self.mean_fit_x_ = self.mean_fit_[0]*(self.plot_y_**2) + self.mean_fit_[1]*self.plot_y_ + self.mean_fit_[2]
        
        y_eval = self.plot_y_ * self.ym_per_pix_
        self.xm_per_pix_ = 3.7 / (np.mean(self.right_fit_x_[360:])-np.mean(self.left_fit_x_[360:]))
        
        A = (self.xm_per_pix_ / (self.ym_per_pix_ ** 2))* self.mean_fit_[0]
        B = (self.xm_per_pix_ / self.ym_per_pix_) * self.mean_fit_[1]
        
        curvature = ((1+(2*A*y_eval + B)**2)**(3/2)) / (2*A)
        
        self.curvature_ = np.mean(curvature)
        
        if abs(self.curvature_) > 10000: #10km
            self.curvature_ = 10000
    
    def carLaneCenterDesviation(self):
        """
        Compute the car's desviation to te lane center
        :param self: reference of the class instance
        """
        lane_center = (self.right_fit_x_[719] + self.left_fit_x_[719])/2.0
        self.center_lane_offset_ = (self.image_shape_[1]/2 - lane_center) * self.xm_per_pix_

                    
    def linesFilter(self):
        """
        Line fit values filter to smooth the lane detection in each iteration
        Low-pass filter
        :param self: reference of the class instance
        """
        if (len(self.filtered_left_fit_) >= self.filter_order_):
            self.filtered_left_fit_.pop(0)
            self.filtered_right_fit_.pop(0)
        
        self.filtered_left_fit_.append(self.current_left_fit_)
        self.filtered_right_fit_.append(self.current_right_fit_)
        
        self.current_left_fit_ = np.sum(self.filtered_left_fit_, axis=0)/len(self.filtered_left_fit_)
        self.current_right_fit_ = np.sum(self.filtered_right_fit_, axis=0)/len(self.filtered_right_fit_)
        
    def getSlopes(self):
        """
        Funtion to get the slopes of each line (left-right).
        :param self: reference of the class instance
        """
        left_bottom_x = self.right_fit_x_[719]
        left_mid_x = self.right_fit_x_[360]
        left_top_x = self.right_fit_x_[0]
        
        right_bottom_x = self.right_fit_x_[719]
        right_mid_x = self.right_fit_x_[360]
        right_top_x = self.right_fit_x_[0]
        
        left_slope_bottom = (360 - 719) / (left_mid_x - left_bottom_x +  0.0000001)
        right_slope_bottom = (360 - 719) / (right_mid_x - right_bottom_x +  0.0000001)
        
        left_slope_top = (0 - 360) / (left_top_x - left_mid_x +  0.0000001)
        right_slope_top = (0 - 360) / (right_top_x - right_mid_x +  0.0000001)
        
        return left_slope_bottom, right_slope_bottom, left_slope_top, right_slope_top
    
    def runPipeline(self,image):
        """
        Funtion to run all the pipeline
        :param self: reference of the class instance
        """
        #Image Processing
        undist = self.img_processor_.correctImageDistortion(image)
        
        binary_undist = self.img_processor_.applyColorAndGradient(undist)
        
        warped_binary = transformPerspective(binary_undist,self.img_processor_.src_points_,
                                             self.img_processor_.dst_points_)
        
        warped_binary = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, np.ones((3,3)))
        warped_binary = cv2.morphologyEx(warped_binary, cv2.MORPH_TOPHAT, np.ones((2,65)))
                
        #Lane Detection
        search_image = None

        if (self.need_deep_search_ == False):
            search_image = self.smartLineSearch(warped_binary)
        
        if (self.need_deep_search_ == True):
            self.locateBaseLines(warped_binary)
            if (self.use_last_fit_ == False):
                search_image = self.deepLineSearch(warped_binary)
 
        if(self.use_last_fit_ == False):
            self.laneCurvature()
            self.carLaneCenterDesviation()
            
        elif((self.use_last_fit_ == True or self.need_deep_search_ == True) and len(self.filtered_left_fit_) > 0):
            self.current_left_fit_ = np.sum(self.filtered_left_fit_, axis=0)/len(self.filtered_left_fit_)
            self.current_right_fit_ = np.sum(self.filtered_right_fit_, axis=0)/len(self.filtered_right_fit_)
            
            self.left_fit_x_ = self.current_left_fit_[0]*self.plot_y_**2 + self.current_left_fit_[1]*self.plot_y_ + self.current_left_fit_[2]
            self.right_fit_x_ = self.current_right_fit_[0]*self.plot_y_**2 + self.current_right_fit_[1]*self.plot_y_ + self.current_right_fit_[2]
            
            self.use_last_fit_ = False
        
        if (self.current_left_fit_ is not None):
            final = self.img_processor_.restorePerspective(warped_binary,undist,self.left_fit_x_, \
                                                self.right_fit_x_,self.plot_y_)

            offset_string = "Center offset: %.2f m" % self.center_lane_offset_
            curvature_string = "Radius of curvature: %.2f km" % (abs(self.curvature_)/1000.0)
            
            if (abs(self.curvature_) >= 5000):
                lane_string = "Straight Lane"
            elif (self.curvature_ >= 0 and self.curvature_ < 5000):
                lane_string = "Right Curve"
            elif (self.curvature_ < 0 and self.curvature_ > -5000):
                lane_string = "Left Curve"

            cv2.putText(final,curvature_string,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(final,offset_string,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(final,lane_string,(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        else:
            final = undist
        
        if search_image is  None:
            search_image = np.dstack((warped_binary, warped_binary, warped_binary))
            search_image = search_image * 255
            
        return np.concatenate((search_image,final), axis=1)
        
####################################################################

path_in = "test_images/"
path_out = "experiments/"
# Make a list of test images

img_names = sorted(glob.glob(path_in + "*.jpg"))
video_names = os.listdir("test_videos/")
lane_finder = LaneFinder()


"""
###################### Save video ######################
for video_name in video_names:
    video_output = 'output_videos/' + video_name
    clip3 = VideoFileClip('test_videos/' + video_name)
    video_clip = clip3.fl_image(lane_finder.runPipeline)
    video_clip.write_videofile(video_output, audio=False)

""" 

""" """
###################### Process video fram to frame ######################
import time
video_name = sys.argv[1]
cv2.namedWindow("final",cv2.WINDOW_NORMAL)
vidcap = cv2.VideoCapture(video_name)
success,image = vidcap.read()

while success:
    
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final = lane_finder.runPipeline(image)

        cv2.imshow("final",final)
        
        cv2.waitKey(1)

        success,image = vidcap.read()
    except KeyboardInterrupt:
        break
        pass

cv2.destroyAllWindows()


"""
###################### Save images from video ######################
vidcap = cv2.VideoCapture('test_videos/harder_challenge_video.mp4')
success,image = vidcap.read()

while success:
    
    success,image = vidcap.read()
  
    if count % 3 == 0:
        #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite("test_images/harder_challenge_video/test"+str(idx_name)+".jpg",image)
        idx_name += 1

cv2.destroyAllWindows()
"""

""" """


