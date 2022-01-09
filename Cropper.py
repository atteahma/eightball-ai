from ImageData import BBox
from Utils import applyBoundingBox


class Cropper:

    rawToTableBBoxFL = (( 0.16927083333333334 , 0.8333333333333334 ),
                        ( 0.15202702702702703 , 0.9403153153153153 ))
    
    tableToSurfaceBBoxFL = (( 0.034 , 0.6285 ),
                            ( 0.064 , 0.72 ))

    
    def __init__(self):
        self.rawToTableBBox = None
        self.tableToSurfaceBBox = None

    def rawToSurface(self, im):
        if self.rawToTableBBox is None or self.tableToSurfaceBBox is None:
            self.computeAbsoluteBBox(*im.shape[:2])
        
        return applyBoundingBox(im, [self.rawToTableBBox, self.tableToSurfaceBBox])
    
    def computeAbsoluteBBox(self, h, w):
        (xL, xR), (yT, yB) = Cropper.rawToTableBBoxFL
        self.rawToTableBBox = BBox((int(h*yT), int(w*xL)), (int(h*yB), int(w*xR)))

        (xL, xR), (yT, yB) = Cropper.tableToSurfaceBBoxFL
        self.tableToSurfaceBBox = BBox((int(h*yT), int(w*xL)), (int(h*yB), int(w*xR)))
