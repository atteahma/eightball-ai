import numpy as np
import cv2

from Utils import dilate

class LineExtractor:

    def __init__(self):
        self.primaryParams = { 'rho': 1,
                               'theta': np.pi/180,
                               'threshold': 40,
                               'minLineLength': 25,
                               'maxLineGap': 100 }
        self.secondaryParams = { 'rho': 1,
                                 'theta': np.pi/180,
                                 'threshold': 10,
                                 'minLineLength': 15,
                                 'maxLineGap': 20 }

    def getLines(self, fgIm, tol=10):

        onlyLinesIm = 255*np.all(np.isclose(fgIm, 255, atol=tol), axis=2).astype(np.uint8)
        linesIm = dilate(onlyLinesIm, i=1)
        lines = []

        # primary line
        primaryLines = cv2.HoughLinesP(linesIm, **self.primaryParams)
        if primaryLines is None:
            return lines
        
        nLines = len(primaryLines)
        primaryLines = primaryLines.reshape((nLines, 2, 2))

        for p1,p2 in primaryLines[:1]:
            
            lines.append((p1,p2))

            # fill in primary line so that
            # secondary lines can be found
            cv2.line(linesIm, p1, p2, 0, 25)
        
        secondaryLines = cv2.HoughLinesP(linesIm, **self.secondaryParams)
        if secondaryLines is None:
            return lines
        
        nLines = len(secondaryLines)
        secondaryLines = secondaryLines.reshape((nLines, 2, 2))
        for p1,p2 in secondaryLines[:2]:
            lines.append((p1,p2))
        
        return lines
