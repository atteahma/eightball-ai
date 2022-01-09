from Cropper import Cropper
from ForegroundExtractor import ForegroundExtractor
from FrameCapture import FrameCapture
from Output import Output

from winGuiAuto import findTopWindow
import cv2

from time import time

def main():

    captureHWND = findTopWindow('airserver')

    frameCapture = FrameCapture(captureHWND, stepSize=1)
    cropper = Cropper()
    foregroundExtractor = ForegroundExtractor()
    output = Output('8 Ball Foreground')

    frameCapture.startCapture()
    output.startOutput()
    while True:
        
        phoneFrame = frameCapture.getNextFrame()
        surfaceFrame = cropper.rawToSurface(phoneFrame)
        foreground = foregroundExtractor.update(surfaceFrame)

        cv2.imwrite(f'./fg_ims/frame_{str(int(round(time(), 3)*1000))}.jpg', foreground)

        output.imshow(foreground)

        fcData = frameCapture.getRuntimeData()
        print(f'input queue len: {fcData.curBufferLen}')
        # print(f'output queue len: {output.getQueueSize()}')

        if not output.run:
            break
    
    frameCapture.stopCapture()
    output.stopOutput()

if __name__ == '__main__':
    main()
