# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:25:28 2017

@author: Zdenek

The goal of perspective transformation is to transform an image such that
we are effectively viewing objects from a different angle or perspective

"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
import pickle


tl = (124,50)
bl = (124,80)
tr = (199,74)
br = (204,105)

img = mpimg.imread('./stop.png')

plt.imshow(img)
plt.plot(*tl, '.')
plt.plot(*bl, '.')
plt.plot(*tr, '.')
plt.plot(*br, '.')
plt.show()

def warp(img):
    """ 
    Define perspective transform function
    """
    img_size = (img.shape[1], img.shape[0])
    
    # Source points
    tr = (199,74)
    br = (204,105)
    bl = (124,80)
    tl = (124,50)
    src = np.float32([tr, br, bl, tl])

    # Destination points
    dst = np.float32([[200, 75],
                     [200, 105],
                     [125, 105],
                     [125, 75]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

fig = plt.figure()
fig.add_subplot(121)
plt.imshow(img)
fig.add_subplot(122)
plt.imshow(warp(img))
