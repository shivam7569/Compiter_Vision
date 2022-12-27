import sys
import cv2
import numpy as np

from RCNN.utils.metrics import iou
from RCNN.utils.util import argmax


class selectiveSearch:

    def __init__(self) -> None:
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def loadAlgo(self, img, strategy='q'):

        self.ss.setBaseImage(img)

        if strategy == 's':
            self.ss.switchToSingleStrategy()
        elif strategy == 'f':
            self.ss.switchToSelectiveSearchFast()
        elif strategy == 'q':
            self.ss.switchToSelectiveSearchQuality()
        else:
            sys.exit()

    def getAnchors(self):
        anchors = self.ss.process()
        anchors[:, 2] += anchors[:, 0]
        anchors[:, 3] += anchors[:, 1]

        return anchors  # [[x1, y1, x2, y2], [...], [...], ...]


def compute_IOUs(rects, bndboxes):

    boundboxes = np.array(
        [
            [
                bb["x1"],
                bb["y1"],
                bb["x2"],
                bb["y2"]
            ] for bb in bndboxes
        ]
    )

    iou_list = list()
    for rect in rects:
        scores = iou(rect, boundboxes)
        bbox_ind, best_iou = argmax(scores)
        iou_list.append([bndboxes[bbox_ind], best_iou])

    return iou_list
