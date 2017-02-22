# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:21:38 2017

@author: Zdenek
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg

img = plt.imread('./curved-lane.jpg')

#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#plt.imshow(gray, cmap='gray')
#
#sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#
#abs_sobelx = np.abs(sobelx)
#abs_sobely = np.abs(sobely)
#
#scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
#
#scaled_sobel = scaled_sobelx
#
#thresh_min = 30
#thresh_max = 100
#sxbinary = np.zeros_like(scaled_sobel)
#sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#plt.imshow(sxbinary, cmap='gray'), plt.show()


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Return the magnitude of the gradient for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255), convert=True):
    """
    Define a function that takes an image, gradient orientation,
    and threshold min / max values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    

    # Return the result
    return binary_output
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Define a function to threshold an image for a given range and Sobel kernel
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
    
def thresholds(img, ksize=3):

    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
    
#plt.imshow(thresholds(img), cmap='gray')
