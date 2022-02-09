from Utils import applyBoundingBox,mask3Dwith2D

import numpy as np
import cv2
import numba as nb

from time import time


class ForegroundExtractor:

    def __init__(self):
        self.learnRate = 0.00025 # assumes 10 fps
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.lastUpdateTime = time()

    def update(self, im):
        lrSuggested = self.learnRate * ((time() - self.lastUpdateTime)/0.1)
        self.lastUpdateTime = time()
        lr = min(0.1, lrSuggested)
        fgMask = self.backSub.apply(im, learningRate=lr)
        return cv2.bitwise_and(im, im, mask=fgMask)

    def resetParameters(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()
