import numpy as np
import cv2
from time import time
from scipy.linalg import lstsq

class GameState:

    def __init__(self, configWindow=None):
        self.configWindow = configWindow

        self.boundingBoxesFl = {
            'selfTimer': (0.4, 0.462, 0.01, 0.145),
            'oppTimer': (1 - 0.462, 1 - 0.4, 0.01, 0.145),
            'timerProgress': (0.12, 0.90, 0.12, 0.90),
            'selfName': (0.365, 0.39, 0.05, 0.09),
            'oppName': (1 - 0.39, 1 - 0.365, 0.05, 0.09),
        }
        self.boundingBoxes = {}
        self.progressPad = 2

        self.turn = -1
        self.timeLeft = -1

        self.lastSelfBrightness = -1
        self.lastOppBrightness = -1
        self.turnHistory = np.ones(50) * -1.0
        self.timeLeftHistory = np.ones((50,2)) * -1.0

    def isAiming(self):
        return not np.all(self.timeLeftHistory[:5 , 0] == self.timeLeftHistory[0 , 0])

    def initBoundingBoxes(self, frame):
        # convert from relative float coordinates to absolute integer coordinates
        Y, X = frame.shape[:2]
        for name, bb in self.boundingBoxesFl.items():
            self.boundingBoxes[name] = tuple(int(bb[i] * V)
                                             for i, V in enumerate((X, X, Y, Y)))

        # this one uses timer image size
        timer = self.crop(frame, self.boundingBoxes['selfTimer'])
        Y, X = timer.shape[:2]
        self.boundingBoxes['timerProgress'] = tuple(int(self.boundingBoxesFl['timerProgress'][i] * V)
                                                    for i, V in enumerate((X, X, Y, Y)))

    def crop(self, frame, bb):
        return frame[ bb[2] : bb[3] , bb[0] : bb[1] ]

    def computeTime(self, timerFrame):

        # crop to having progress bar on exterior edge
        progressFrame = self.crop(timerFrame, self.boundingBoxes['timerProgress'])

        H, W = progressFrame.shape[:2]
        K = self.progressPad

        # get outer edges
        top = progressFrame[0:K , :].reshape((K, W, 3))
        right = progressFrame[K:H-K , W-K:W].reshape((H - 2*K, K, 3))
        bottom = progressFrame[H-K:H , :].reshape((K, W, 3))
        left = progressFrame[K:H-K , 0:K].reshape((H - 2*K, K, 3))
        
        # average over K rings
        top = np.mean(top, axis=0).astype(np.uint8)
        right = np.mean(right, axis=1).astype(np.uint8)
        bottom = np.mean(bottom, axis=0).astype(np.uint8)
        left = np.mean(left, axis=1).astype(np.uint8)

        # merge into 1D vector
        progressBGR = np.concatenate((top, right, bottom[::-1], left[::-1]), axis=0)
        progressVal = np.max(progressBGR, axis=1)

        # buffer using 0 and max value for a smoother start and end of timer
        progressMinBuffer = np.ones(len(progressVal) // 4, dtype=np.uint8) * 0#progressVal.min()
        progressMaxBuffer = np.ones(len(progressVal) // 4, dtype=np.uint8) * progressVal.max()
        progressValBuffered = np.concatenate((progressVal, progressMinBuffer, progressMaxBuffer), dtype=np.uint8)

        # apply adaptive threshold
        threshOTSU, progressBuffered = cv2.threshold(progressValBuffered, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # remove buffer
        progress = progressBuffered[:len(progressVal)].flatten()

        if self.configWindow:
            # roll to first falling edge
            fallingEdge = np.nonzero(progress[:-1] & ~progress[1:])[0] + 1
            progressBGROut = np.roll(progressBGR, -fallingEdge, axis=0)
            progressOut = np.roll(progress, -fallingEdge, axis=0)
            
            self.configWindow.addDrawEvent('progressbar', np.expand_dims(progressBGROut, 0))
            self.configWindow.addDrawEvent('progressbarthresh', np.expand_dims(progressOut, 0))
        
        timeFloat = np.sum(progress) / len(progress)

        return timeFloat

    def computeMeanBrightness(self, frame):
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        meanBrightness = np.sum(brightness) / np.product(brightness.shape)
        return meanBrightness

    def computeTurn(self, frame):
        selfName = self.crop(frame, self.boundingBoxes['selfName'])
        oppName = self.crop(frame, self.boundingBoxes['oppName'])

        selfBrightness = self.computeMeanBrightness(selfName)
        oppBrightness = self.computeMeanBrightness(oppName)

        proposedTurn = int(selfBrightness < oppBrightness)

        self.lastSelfBrightness = selfBrightness
        self.lastOppBrightness = oppBrightness

        return proposedTurn
    
    def update(self, frame):

        if not self.boundingBoxes:
            self.initBoundingBoxes(frame)
        
        self.turn = self.computeTurn(frame)
        self.turnHistory = np.roll(self.turnHistory, 1)
        self.turnHistory[0] = self.turn

        if self.turn == 0:
            timerBB = 'selfTimer'
        elif self.turn == 1:
            timerBB = 'oppTimer'
        timer = self.crop(frame, self.boundingBoxes[timerBB])

        self.timeLeft = self.computeTime(timer)
        self.timeLeftHistory = np.roll(self.timeLeftHistory, 1, axis=0)
        self.timeLeftHistory[0][0] = self.timeLeft
        self.timeLeftHistory[0][1] = time()

        if self.configWindow:
            self.configWindow.addDrawEvent('timer', timer)
            self.drawStats()
    
    def estimateTimeLeft(self, minDataPoints=10):
        # # filter out invalid
        # validMask = np.all(self.timeLeftHistory != -1, axis=1)
        # timeLeftTS = self.timeLeftHistory[validMask , :]
        # if len(timeLeftTS) == 0:
        #     return -1
        dataTS = self.timeLeftHistory

        # cut to only current turn
        turnStarts = np.flatnonzero(np.diff(dataTS[:,0], 1) < -0.3)
        if len(turnStarts) == 0:
            turnStartIndex = len(dataTS)
        else:
            return -1
            #turnStartIndex = max(turnStarts[0] - 2, 0)
        dataTS = dataTS[:turnStartIndex , :]
        if len(dataTS) < minDataPoints:
            return -1

        # build lst squares (for fun)
        timeLeftTS = dataTS[:,0]
        secondsTS = dataTS[:,1]

        A = np.stack((timeLeftTS, np.ones_like(timeLeftTS)), axis=1)
        b = secondsTS

        x, _, _, _ = lstsq(A, b)

        estimatedEndTime = np.dot(np.array([0,1]), x)
        return estimatedEndTime - time()

    def drawStats(self):
        turnStr = ['self', 'opp', 'invalid'][self.turn]
        currTime = str(round(self.timeLeft, 3))
        aimStr = [('balls moving', (0,0,255)), ('aiming', (0,255,0))][int(self.isAiming())]
        if self.isAiming():
            estTime = str(round(self.estimateTimeLeft(), 3))
        else:
            estTime = -1

        self.configWindow.addDrawEvent('turn', turnStr)
        self.configWindow.addDrawEvent('timeLeft', currTime)
        self.configWindow.addDrawEvent('aiming', aimStr)
        #self.configWindow.addDrawEvent('timelefthistory', [self.timeLeftHistory])
        self.configWindow.addDrawEvent('estTimeLeft', estTime)
