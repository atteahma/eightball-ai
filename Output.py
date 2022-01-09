import numpy as np
import cv2

import threading
from queue import Queue

class Output:

    def __init__(self, winName):
        self.winName = winName
        self.frameQueue = Queue()
        self.run = True
        self.winThread = threading.Thread(target=self.outputWindowLoop, args=(), daemon=True)

    def outputWindowLoop(self):
        cv2.namedWindow(self.winName)
        while self.run:
            self.frameQueue = Queue()
            frame = self.frameQueue.get() # is blocking
            cv2.imshow(self.winName, frame)
            if cv2.waitKey(5) == 27:
                self.stopOutput()

    def imshow(self, im):
        self.frameQueue.put(im)

    def startOutput(self):
        self.winThread.start()

    def stopOutput(self):
        self.run = False
        cv2.destroyWindow(self.winName)
    
    def getQueueSize(self):
        return self.frameQueue.qsize()
        