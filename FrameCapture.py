from ImageData import BBox
from Utils import applyBoundingBox

import numpy as np
import cv2
from mss.windows import MSS as mss

from win32 import win32gui
from dataclasses import dataclass
import threading
from time import sleep, time
from math import ceil


@dataclass
class FrameCaptureRuntimeData:
    maxBufferLen: int
    curBufferLen: int
    numFramesCaptured: int
    numFramesReturned: int
    curBufferExceeded: bool

class FrameCapture:

    def __init__(self, hwnd='', updatePos=False, bufferLen=1000, maxFps=15, stepSize=1, onlyBorders=False):
        self.hwnd = hwnd
        self.bufferLen = bufferLen
        self.updatePos = updatePos
        self.stepSize = stepSize
        self.maxFps = maxFps

        self.borderBBox = None
        
        if onlyBorders:
            return
        
        if updatePos:
            self.windowRect = None
        else:
            self.windowRect = self.getMSSWindowRect()

        self.sct = mss()

        self.windowsTracesBBox = BBox( (int(40/self.stepSize), int(15/self.stepSize)),
                                       (int(-10/self.stepSize),int(-15/self.stepSize)) )

        self.frameBuffer = np.zeros((bufferLen, *self.getRawWindowSize(), 3), dtype=np.uint8)

        self.nextFrameToReturn = 0
        self.curFrame = 0

        self.run = True
        self.captureThread = threading.Thread(target=self.captureWindowLoop, args=(), daemon=True)

    def getNextFrame(self, raw=False, num=1):
        while not (self.nextFrameToReturn < self.curFrame):
            sleep(0.1)

        frame = self.frameBuffer[self.nextFrameToReturn % self.bufferLen]        
        self.nextFrameToReturn += 1

        if not raw:
            frame = self.removeWindowsTraces(frame)
            frame = self.removeBorders(frame)

        return frame

    def captureWindow(self):
        if self.updatePos:
            self.windowRect = self.getMSSWindowRect()
            assert np.all(np.array(self.getRawWindowSize()) == np.array(self.frameBuffer.shape[1:3]))
        
        frame = np.asarray(self.sct.grab(self.windowRect))
        frame = frame[::self.stepSize,::self.stepSize,:3] # fast bgra -> bgr and downsizing
        self.frameBuffer[self.curFrame % self.bufferLen] = frame
        self.curFrame += 1
        
        if (self.curFrame - self.nextFrameToReturn) > self.bufferLen:
            assert False, 'frameCapture buffer ran out'
    
    def captureWindowLoop(self):
        while self.run:
            tStart = time()
            self.captureWindow()
            sleep((1/self.maxFps) - (time() - tStart))

    def startCapture(self):
        self.captureThread.start()

    def stopCapture(self):
        self.run = False
        self.sct.close()

    def getMSSWindowRect(self):
        x1,y1,x2,y2 = win32gui.GetWindowRect(self.hwnd)
        return {'left': x1, 'top': y1, 'width': x2-x1, 'height': y2-y1}

    def getRawWindowSize(self):
        x1,y1,x2,y2 = win32gui.GetWindowRect(self.hwnd)
        xSize = ceil(abs(x2 - x1) / self.stepSize)
        ySize = ceil(abs(y2 - y1) / self.stepSize)
        return ySize,xSize

    def removeWindowsTraces(self, im):
        return applyBoundingBox(im, self.windowsTracesBBox)
    
    def removeBorders(self, im):
        if self.borderBBox is None:
            self.borderBBox = self.calculateBordersBBox(im)
        return applyBoundingBox(im, self.borderBBox)
    
    def calculateBordersBBox(self, im):
        thresh = 5

        flatIm = np.max(im, axis=2)
        rows = np.where(np.max(flatIm, axis=0) > thresh)[0]
        cols = np.where(np.max(flatIm, axis=1) > thresh)[0]

        if rows.size:
            xLeft = rows[0]
            xRight = rows[-1] + 1
        else:
            xLeft = 0
            xRight = im.shape[1]

        if cols.size:
            yTop = cols[0]
            yBottom = cols[-1] + 1
        else:
            yTop = 0
            yBottom = im.shape[0]

        return BBox((yTop, xLeft), (yBottom, xRight))

    def getRuntimeData(self):
        return FrameCaptureRuntimeData(self.bufferLen,
                                       self.curFrame - self.nextFrameToReturn,
                                       self.curFrame,
                                       self.nextFrameToReturn,
                                       self.curFrame - self.nextFrameToReturn >= self.bufferLen)