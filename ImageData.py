from dataclasses import dataclass


@dataclass
class BBox:
    topLeft: tuple[int,int]      # y,x
    bottomRight: tuple[int,int]  # y,x
