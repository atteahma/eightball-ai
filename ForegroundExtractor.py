from Utils import applyBoundingBox,mask3Dwith2D

import numpy as np
import cv2
import numba as nb


class ForegroundExtractor:

    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def update(self, im):
        fgMask = self.backSub.apply(im, learningRate=0.0005)
        return cv2.bitwise_and(im,im,mask=fgMask)

    def resetParameters(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()
