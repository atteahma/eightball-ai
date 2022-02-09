import numpy as np
import cv2

class LineRasterizer:

    def __init__(self, lineColor=(0,0,255), lineThickness=3):
        self.lineColor = lineColor
        self.lineThickness = lineThickness

    def draw(self, im, lines):
        
        outIm = im.copy()
        maxMag = max(im.shape)

        for p1, p2 in lines:
            dirVec = p2 - p1
            dirVecUnit = dirVec / np.linalg.norm(dirVec)

            extP1 = tuple((p1 - maxMag * dirVecUnit).astype(int))
            extP2 = tuple((p1 + maxMag * dirVecUnit).astype(int))

            cv2.line(outIm, extP1, extP2, self.lineColor, self.lineThickness)
        
        return outIm