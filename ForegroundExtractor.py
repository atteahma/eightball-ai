from Utils import applyBoundingBox,mask3Dwith2D

import numpy as np
import cv2
import numba as nb

from time import time


class ForegroundExtractor:

    def __init__(self, configWindow=None):
        self.configWindow = configWindow
        self.ballsLearnRate = 0.05
        self.aimLearnRate = 0.00001
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.lastUpdateTime = time()

    def update(self, im, isAiming):
        lr = self.aimLearnRate if isAiming else self.ballsLearnRate
        
        fgMask = self.backSub.apply(im, learningRate=lr)
        fg = cv2.bitwise_and(im, im, mask=fgMask)

        if self.configWindow:
            self.configWindow.addDrawEvent('fg', fg)
            lrStr = str(round(lr, 7))
            self.configWindow.addDrawEvent('moglr', lrStr + '0'*(7-len(lrStr)))

        self.lastUpdateTime = time()

        return fg

    def resetParameters(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()
