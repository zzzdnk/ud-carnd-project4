# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:08:37 2017

@author: Zdenek

The goal of camera calibration is to compute transformation between 3D 
object points in the world and 2D image points

The goal of distrotion correction is to ensure that the geometrical shape
of objects is represented consistently, no matter where they appear
in the image

"""

import cv2
import numpy as np
import glob
import pickle

def calibrate_camera(cal_images='./camera_cal/calibration*.jpg', nx=9, ny=6):
    """
    Calibrate camera with calibration chessboards and return distortion matrices
    
    nx = the number of inside corners in x
    the number of inside corners in y
    
    """
    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    # Make a list of calibration images
    images = glob.glob(cal_images)
    
    for image in images:
        img = cv2.imread(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If found, draw corners
        if ret == True:
            print('Corners found in {}'.format(image))
            imgpoints.append(corners)
            objpoints.append(objp)            
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    with open('camera_cal.pickle', 'wb') as f:
        pickle.dump({'imgpoints': imgpoints, 'objpoints': objpoints, 'mtx': mtx, 'dist': dist}, f)

    return mtx, dist
    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
#import matplotlib.pyplot as plt
#
#img = plt.imread('./test_images/test1.jpg')
#mtx, dist = calibrate_camera()
#dst = cv2.undistort(img, mtx, dist, None, mtx)
#vertices = np.array([[[620,450],[250,700],[1200,700], [720,450]]])
#plt.imshow(img), plt.show()
##plt.imshow(region_of_interest(img, vertices))
