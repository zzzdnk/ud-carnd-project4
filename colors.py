# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:10:19 2017

@author: Zdenek
"""

import numpy as np
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg



def pipeline(img, s_thresh=(199, 255), sx_thresh=(20, 100)):
    """
    Apply both the Sobel-x transform and S-channel filtering
    """
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.dstack(( (sxbinary != s_binary).astype(np.int), sxbinary, s_binary))
    return color_binary
    
def hls_s(img, s_thresh=(199, 255)):
    """
    Apply a threshold to the S channel of the input image
    """    
    # Convert to HLS color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return s_binary
    
    
    
#result = pipeline(image)
#
## Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#
#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=40)
#
#ax2.imshow(result)
#ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#    
