from Cropper import Cropper
from ForegroundExtractor import ForegroundExtractor
from FrameCapture import FrameCapture
from GameState import GameState
from LineExtractor import LineExtractor
from LineRasterizer import LineRasterizer

from rdr2_ai.configWindow.configWindow import ConfigWindow
from rdr2_ai.configWindow.configWindowTemplate import ConfigWindowTemplate, ContentType
from rdr2_ai.utils.capture import Capture
from rdr2_ai.utils.fps import FPSCounter

from winGuiAuto import findTopWindow
import cv2

from time import time, sleep


configWindowTemplate = ConfigWindowTemplate() \
    .setSize(1200,2000) \
    .addStaticText('Foreground', (25,25), (50,800)) \
    .addContentBox('fg', ContentType.Image, (100,25), (400,800)) \
    .addStaticText('Lines (shaky)', (525,25), (50,800)) \
    .addContentBox('lines', ContentType.Image, (600,25), (600,1200)) \
    \
    .addStaticText('MOG LR:', (50, 875), (40, 200)) \
    .addContentBox('moglr', ContentType.Text, (50, 875 + 200 + 25), (40, 200)) \
    \
    .addContentBox('aiming', ContentType.Text, (50, 875 + 200 + 25 + 200 + 25), (60, 400)) \
    \
    .addStaticText('Turn:', (100, 875), (40, 200)) \
    .addContentBox('turn', ContentType.Text, (100, 875 + 200 + 25), (40, 200)) \
    .addStaticText('Time Left:', (150, 875), (40, 200)) \
    .addContentBox('timeLeft', ContentType.Text, (150, 875 + 200 + 25), (40, 200)) \
    .addStaticText('Est Time Left:', (150, 875 + 200 + 25 + 200 + 25), (40, 200)) \
    .addContentBox('estTimeLeft', ContentType.Text, (150, 875 + 200 + 25 + 200 + 25), (40, 200)) \
    .addContentBox('timer', ContentType.Image, (200, 875), (300, 300)) \
    .addContentBox('progressbar', ContentType.Image, (200, 875+300+25), (25, 500)) \
    .addContentBox('progressbarthresh', ContentType.Image, (250, 875+300+25), (25, 500)) \
    .addContentBox('timelefthistory', ContentType.Plot, (600, 875), (350,800)) \
    \
    .addContentBox('fps', ContentType.Text, (25, 1800), (40, 80)) \
    .addStaticText('FPS', (25, 1900), (40, 80)) \


class Main:
    
    def __init__(self):
        self.configWindow = ConfigWindow('8 Ball AI Tool',
                                         (25,25),
                                         template=configWindowTemplate,
                                         drawFps=30)
        self.capture = Capture('airserver', updateWindow=False)
        self.frameCapture = FrameCapture(self.capture.hwnd, bufferLen=1, onlyBorders=True)
        self.cropper = Cropper()
        self.gameState = GameState(configWindow=self.configWindow)
        self.foregroundExtractor = ForegroundExtractor(configWindow=self.configWindow)
        self.lineExtractor = LineExtractor()
        self.lineRasterizer = LineRasterizer(configWindow=self.configWindow)
        self.fpsCounter = FPSCounter(configWindow=self.configWindow)

    def runLoop(self, maxFPS=15):
        self.configWindow.startLoop()
        while True:
            
            # log fps
            self.fpsCounter.tick()

            # get frame from airserver
            rawFrame = self.capture.captureWindow()
            phoneFrame = self.frameCapture.removeBorders(rawFrame[3:])

            # update internal game state
            self.gameState.update(phoneFrame)

            # compute and draw things
            self.doShaky(phoneFrame)
            
            if self.configWindow.done():
                break
            
            # maintain max fps
            self.fpsCounter.sleep(maxFPS)
        
        self.configWindow.endLoop()

    def doShaky(self, phoneFrame):
        # extract moving parts
        tableFrame = self.cropper.rawToTable(phoneFrame)
        surfaceFrame = self.cropper.rawToSurface(phoneFrame)
        foreground = self.foregroundExtractor.update(surfaceFrame, self.gameState.isAiming())

        if self.gameState.isAiming():
            # find aiming lines
            lines = self.lineExtractor.getLines(foreground, tol=10)
            if lines:
                # convert coordinates from surface space to table space
                linesTable = self.cropper.convertSurfaceToTable(lines)

                # draw extended lines
                self.lineRasterizer.draw(tableFrame, linesTable)

if __name__ == '__main__':
    main = Main()
    main.runLoop()
