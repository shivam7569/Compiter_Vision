import time
import shutil
import cv2
import os
import random
import numpy as np
from RCNN.globalParams import Global
from RCNN.utils.util import check_dir, parse_xml
from RCNN.computations import selectiveSearch, compute_IOUs


def parse_annotation_jpeg(annotation_path, jpeg_path, ss):
    """ 
    Get positive and negative samples (note: ignore the label bounding box whose property difficult is True) 
    Positive sample: Candidate suggestion and labeled bounding box IoU is greater than or equal to 0.5 
    Negative sample: IoU is greater than 0 and less than 0.5. In order to further limit the number of negative samples, its size must be larger than 1/5 of the label box 
    """

    img = cv2.imread(jpeg_path)
    height, width, _ = img.shape
    ss.loadAlgo(img)

    rects = ss.getAnchors()
    np.random.shuffle(rects)
    rects = rects[:Global.NUM_PROPOSALS, :]

    bndboxes = parse_xml(annotation_path)

    maximum_bndbox_size = 0
    for bndbox in bndboxes:
        x1, y1, x2, y2 = int(bndbox["x1"] * width), int(bndbox["y1"] * height), int(
            bndbox["x2"] * width), int(bndbox["y2"] * height)
        area = (x2 - x1) * (y2 - y1)

        if area > maximum_bndbox_size:
            maximum_bndbox_size = area

    iou_list = compute_IOUs(rects, bndboxes, width, height)
    positive_list = []
    negative_list = []

    assert len(iou_list) == rects.shape[0]

    for i in range(rects.shape[0]):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)
        iou_score = iou_list[i][1]
        gt_details = iou_list[i][0]

        if iou_score >= 0.5:

            entry = {
                "proposal_coord": [str(xmin), str(ymin), str(xmax), str(ymax)],
                "proposal_class": str(gt_details["class_id"]),
                "gts": [
                    [
                        str(int(i["x1"] * width)),
                        str(int(i["y1"] * height)),
                        str(int(i["x2"] * width)),
                        str(int(i["y2"] * height)),
                        str(i["class_id"])
                    ] for i in bndboxes
                ]

            }
            positive_list.append(entry)

        elif (0.0 < iou_score < 0.5) and (rect_size > maximum_bndbox_size / 5):
            entry = {
                "proposal_coord": [str(xmin), str(ymin), str(xmax), str(ymax)],
                "proposal_class": "0",
                "gts": [
                    [
                        str(int(i["x1"] * width)),
                        str(int(i["y1"] * height)),
                        str(int(i["x2"] * width)),
                        str(int(i["y2"] * height)),
                        str(i["class_id"])
                    ] for i in bndboxes
                ]

            }
            negative_list.append(entry)  # background

    return positive_list, negative_list
