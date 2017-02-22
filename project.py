# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:21:31 2017

@author: Zdenek
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
import pickle
import os

#%%
def calibrate():
    """
    Calibrate the camera or load the previously saved values
    """
    if os.path.exists('./camera_cal.pickle'):
        print('Loading camera coefficients...')
    
        with open('./camera_cal.pickle', 'rb') as f:
            data = pickle.load(f)
            mtx = data['mtx']
            dist = data['dist']
    else:
        import camera
        print('Calibrating the camera...')
        mtx, dist = camera.calibrate_camera()
    
    return mtx, dist

#%%2 
def color_grad_filters(img):
    """
    Apply color and gradient thresholds
    """
    import colors
    # Filter out S-channel values
    s_binary = colors.hls_s(img)
    #plt.imshow(s_binary, cmap='gray') , plt.title('S-channel'), plt.show()
    
    # Apply gradient threshods
    from sobel import abs_sobel_thresh, thresholds
    #sx_binary = abs_sobel_thresh(img, orient='y', thresh=(np.pi/6, np.pi/4), sobel_kernel=15)
    sx_binary = thresholds(img)
    #plt.imshow(sx_binary, cmap='gray'), plt.title('Sobel x'), plt.show()
    binary = np.zeros_like(s_binary)
    
    binary[(s_binary == 1) | (sx_binary == 1)] = 1

    return binary

#%%
"""
Apply perspective transform
"""
def warp(img, dst = np.float32([[825, 500], [1025, 675], [275, 675], [475, 500]])):
    """ 
    Define perspective transform function
    """
    img_size = (img.shape[1], img.shape[0])
    
    tr = (766,500)
    br = (1025,675)
    bl = (275,675)
    tl = (522,500)
    src = np.float32([tr, br, bl, tl])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, Minv

def find_segments(binary_warped):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
       
#    global left_segments
#    global right_segments
#    
#        
    # Check the absolute coefficient   
#    if not (250 < left_fit[2] < 350):
#        print('invalid left line: {}'.format(left_fit[2]))
#        if not left_segments.shape[0] == 0:
#            left_fit = left_segments.iloc[left_segments.shape[0]-1,:]
#        else:
#            left_fit[2] = 300
#    
#    if not (1000 < right_fit[2] < 1150):
#        print('invalid right line {}'.format(right_fit[2]))
#        if not right_segments.shape[0] == 0:
#            right_fit = right_segments.iloc[right_segments.shape[0]-1,:]
#        else:
#            right_fit[2] = 1075

#    n = 3
#    if left_segments.shape[0] >= n:
#        print(left_segments.rolling(n).mean()[-1:].values[0])
#        
#
#    if right_segments.shape[0] >= n:
#        right_fit = right_segments.rolling(n).mean()[-1:].values[0]
#        print(right_fit)
#        
    left_segments.loc[left_segments.shape[0]] = left_fit
    right_segments.loc[left_segments.shape[0]] = right_fit


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, ploty, left_fitx, right_fitx
#%%    

def compute_radius(left_fit, right_fit, ploty, leftx, rightx):
    """
    Return the right and left radius of curvature
    """
    
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    #print('left', left_fit_cr)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_r = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_r = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    radius = np.min([left_r, right_r])
       
    return radius
    
def compute_distance(left_fit, right_fit, ploty, leftx, rightx):
    """
    Compute the distance from the center of the lane
    """
    offset = 0.0
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    offset = (leftx[-1] + rightx[-1]) / 2
    offset = xm_per_pix*(offset - 640.0)

    return offset
    

def draw_lines(img, w, Minv):
    # Find the lanes
    left_fit, right_fit, ploty, left_fitx, right_fitx = find_segments(w)
    
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    # Get the radius of curvature
    radius = compute_radius(left_fit, right_fit, ploty, left_fitx, right_fitx)
    #print(radius)
    
    offset = compute_distance(left_fit, right_fit, ploty, left_fitx, right_fitx)
    
    
    cv2.putText(result,"Radius of curvature:  {:.0f} m".format(radius), (10,100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),3)
    cv2.putText(result,"Offset: {:.2f} m".format(offset), (10,120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
    #cv2.putText(result,"Right radius of curvature: {:.1f} m".format(right_r), (10,120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
    
    #plt.imshow(result), plt.show()
    
    return result

#%% 
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import pandas as pd

mtx, dist = calibrate()
left_segments = pd.DataFrame(columns=['l2', 'l1', 'l0'])
right_segments = pd.DataFrame(columns=['r2', 'r1', 'r0'])

def process_image(image):
    
    image = cv2.undistort(image, mtx, dist, None, mtx)
    
    binary = color_grad_filters(image)
    
    #plt.imshow(binary, cmap='gray'), plt.show()
    
    w, Minv = warp(binary, np.float32([[1025, 200], [1025, 675], [275, 675], [275, 200]]))  
    #plt.imshow(w, cmap='gray'), plt.show()
    
    result = draw_lines(image, w, Minv)
    
    
    return result
    
def process_frames(get_frame, t):
  
    image = get_frame(t)
    result = process_image(image)
    #cv2.putText(result,"Frame: {}".format(np.int(30*t)), (100,200), cv2.FONT_HERSHEY_DUPLEX, 1, 100,3)

    return result

#    
#images = glob.glob('./test_images/*.jpg')
#for image in images:
#    fig = plt.figure(figsize=(10,10))
#    img = process_image(plt.imread(image))
#    plt.imshow(img)
#    plt.title(image)
#    mpimg.imsave('./output_images/out_' + image[image.find('\\') + 1:], img)
#    plt.show()
    
#%%
#plt.figure(figsize=(12,10))
#img = mpimg.imread('./test_images/test1.jpg')    
#plt.imshow(process_image(img)), plt.show()

video = 'project_video'

output = video + '_out.mp4'
clip = VideoFileClip('./' + video + '.mp4')
#subclip = clip.subclip(30, 60)
subclip = clip
out_clip = subclip.fl(process_frames) 
out_clip.write_videofile(output, audio=False)
