from Cropper import Cropper
from ForegroundExtractor import ForegroundExtractor
from FrameCapture import FrameCapture
from LineExtractor import LineExtractor
from LineRasterizer import LineRasterizer
from Output import Output

from winGuiAuto import findTopWindow
import cv2

from time import time

def main():

    captureHWND = findTopWindow('airserver')

    frameCapture = FrameCapture(captureHWND, stepSize=1)
    cropper = Cropper()
    foregroundExtractor = ForegroundExtractor()
    lineExtractor = LineExtractor()
    lineRasterizer = LineRasterizer()
    output = Output('8 Ball Foreground')

    frameCapture.startCapture()
    output.startOutput()
    while True:
        
        phoneFrame = frameCapture.getNextFrame()
        tableFrame = cropper.rawToTable(phoneFrame)
        surfaceFrame = cropper.rawToSurface(phoneFrame)
        foreground = foregroundExtractor.update(surfaceFrame)
        if not output.catchup:
            lines = lineExtractor.getLines(foreground)
            if len(lines) > 0:
                linesTable = cropper.convertSurfaceToTable(lines)
                linesIm = lineRasterizer.draw(tableFrame, linesTable)
                output.imshow(linesIm)
            #cv2.imwrite(f'./fg_ims/frame_{str(int(round(time(), 3)*1000))}.jpg', foreground)
            else:
                output.imshow(tableFrame)
        else:
            output.imshow(tableFrame)
        
        fcData = frameCapture.getRuntimeData()
        if fcData.curBufferLen > 10:
            print(f'input queue len: {fcData.curBufferLen}')
            print(f'output queue len: {output.getQueueSize()}')

        if not output.run:
            break
    
    frameCapture.stopCapture()
    output.stopOutput()

if __name__ == '__main__':
    main()
