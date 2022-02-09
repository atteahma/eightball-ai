from ImageData import BBox

import numpy as np
import cv2
from numba import njit

def applyBoundingBox(im, bbs):
    isVideo = len(im.shape) == 4
    
    if type(bbs) is BBox:
        bbs = [bbs]

    for bb in bbs:
        yTop, xLeft = bb.topLeft
        yBottom, xRight = bb.bottomRight
        if isVideo:
            im = im[: , yTop:yBottom , xLeft:xRight]
        else:
            im = im[yTop:yBottom , xLeft:xRight]
    
    return im

def get3Dmask(mask):
    return np.stack((mask,mask,mask), axis=2).astype(bool)

def mask3Dwith2D(im, mask, copy=True):
    if copy:
        im = im.copy()
    mask_3d = get3Dmask(mask)
    im[~mask_3d] = 0
    return im

def dilate(image, k=3, i=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image, kernel, iterations=i)

def erode(image, k=3, i=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(image, kernel, iterations=i)