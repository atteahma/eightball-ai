import numpy as np
import cv2

class Tracker:

    def __init__(self, configWindow=None, trailsDecayRate=0.95):
        self.configWindow = configWindow
        self.trailsDecayRate = trailsDecayRate
        
        self.trailsCanvas = None
        self.balls = []

    def getBalls(self, frameBGR):
        frame = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2GRAY)

        H = frameBGR.shape[0]
        #frame = cv2.medianBlur(frame, 5)
        minDia = 38 * H / 832
        maxDia = 50 * H / 832
        balls = cv2.HoughCircles(frame,
                                 cv2.HOUGH_GRADIENT,
                                 1, minDist=minDia,
                                 param1=100, param2=30,
                                 minRadius=int(round(minDia/2))-1, maxRadius=int(round(maxDia/2))+1 )
        return balls

    def update(self, frame):

        self.balls = self.getBalls(frame)

        if self.configWindow:
            self.drawTrailsCanvas(frame)
                
    def drawTrailsCanvas(self, frame):
        
        if self.trailsCanvas is None:
            self.trailsCanvas = np.zeros_like(frame)
        
        self.trailsCanvas = self.trailsCanvas * self.trailsDecayRate
        self.trailsCanvas = self.trailsCanvas.astype(np.uint8)
        if self.balls is not None:
            self.balls = self.balls[0]
            for x,y,r in self.balls:
                self.trailsCanvas = cv2.circle(self.trailsCanvas, (int(x),int(y)), int(round(r*0.5)), color=(0,0,255), thickness=-1)
        
        trailsCanvasMask = np.any(self.trailsCanvas > 0, axis=2)
        trailsCanvasMask3D = np.stack((trailsCanvasMask, trailsCanvasMask, trailsCanvasMask), axis=2)
        finalImage = np.where(trailsCanvasMask3D, self.trailsCanvas, frame)

        self.configWindow.addDrawEvent('tracker', finalImage)
