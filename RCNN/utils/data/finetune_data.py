import json
import os
import random
import shutil
import cv2
import time
import numpy as np
from RCNN.utils.globalParams import Global
from RCNN.utils.util import check_dir, parse_xml
from RCNN.utils.computations import compute_IOUs, selectiveSearch


def parse_annotation_jpeg(coco, imgID, directory, ss):
    """ 
    Get positive and negative samples (note: ignore the label bounding box whose property difficult is True) 
    Positive sample: Candidate suggestion and labeled bounding box IoU is greater than or equal to 0.5 
    Negative sample: IoU is greater than 0 and less than 0.5. In order to further limit the number of negative samples, its size must be larger than 1/5 of the label box 
    """

    img_metadata = coco.loadImgs([imgID])
    image_name = img_metadata[0]['file_name'].split(".")[0]
    jpeg_path = os.path.join(Global.DATA_DIR, directory, image_name + ".jpg")
    img = cv2.imread(jpeg_path)
    
    ss.loadAlgo(img)

    rects = ss.getAnchors()
    np.random.shuffle(rects)
    rects = rects[:Global.NUM_PROPOSALS]

    annIds = coco.getAnnIds(imgIds=[imgID])
    anns = coco.loadAnns(annIds)
    bndboxes = []

    for data in anns:
        x1, y1, x2, y2 = int(data["bbox"][0]), int(data["bbox"][1]), int(
            data["bbox"][0] + data["bbox"][2]), int(data["bbox"][1] + data["bbox"][3])
        if (x2 - x1) > 16 and (y2 - y1) > 16:
            bndboxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": int(data["category_id"]),
                    "class_name": Global.CLASS_LABELS[int(data["category_id"])]
                }
            )

    if len(bndboxes) == 0:
        return [], []

    maximum_bndbox_size = 0
    for bndbox in bndboxes:
        x1, y1, x2, y2 = bndbox["x1"], bndbox["y1"], bndbox["x2"], bndbox["y2"]
        area = (x2 - x1) * (y2 - y1)

        if area > maximum_bndbox_size:
            maximum_bndbox_size = area

    iou_list = compute_IOUs(rects, bndboxes)
    positive_list = {
        "image_name": None,
        "proposal_coord": [],
        "gts": None
    }
    negative_list = {
        "image_name": None,
        "proposal_coord": [],
        "gts": None
    }

    assert len(iou_list) == rects.shape[0]

    for i in range(rects.shape[0]):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)
        iou_score = iou_list[i][1]
        gt_details = iou_list[i][0]

        if iou_score >= 0.5:

            positive_list["proposal_coord"].append(
                [str(xmin), str(ymin), str(xmax), str(
                    ymax), str(gt_details["class_id"])]
            )

        elif (0.0 < iou_score < 0.5) and (rect_size > maximum_bndbox_size / 5):

            negative_list["proposal_coord"].append(
                [str(xmin), str(ymin), str(xmax), str(ymax), "0"]
            )

    num_positives = len(positive_list["proposal_coord"])
    if len(negative_list["proposal_coord"]) > num_positives:
        negative_list["proposal_coord"] = random.sample(
            negative_list["proposal_coord"], num_positives)

    gts = [
        [
            str(i["x1"]),
            str(i["y1"]),
            str(i["x2"]),
            str(i["y2"]),
            str(i["class_id"])
        ] for i in bndboxes
    ]

    positive_list["image_name"] = image_name
    negative_list["image_name"] = image_name
    positive_list["gts"] = gts
    negative_list["gts"] = gts

    return positive_list, negative_list


def process_data(args):

    directory, coco, sample_id = args
    ss = selectiveSearch()

    dst_root_dir = os.path.join(Global.FINETUNE_DATA_DIR, directory)
    dst_annotation_dir = os.path.join(dst_root_dir, "Annotations")

    check_dir(dst_root_dir)
    check_dir(dst_annotation_dir)

    start_time = time.time()

    positive_list, negative_list = parse_annotation_jpeg(
        coco, sample_id, directory, ss)

    if len(positive_list) > 0 and len(negative_list) > 0:

        sample_name = positive_list["image_name"]

        dst_positive_annot_path = os.path.join(
            dst_annotation_dir, sample_name + "_1" + ".json")
        dst_negative_annot_path = os.path.join(
            dst_annotation_dir, sample_name + "_0" + ".json")

        with open(dst_positive_annot_path, "w") as f:
            json.dump(positive_list, f)

        with open(dst_negative_annot_path, "w") as f:
            json.dump(negative_list, f)

    end_time = time.time()
    time_taken = end_time - start_time

    print("Parsed {} in {:.0f}m {:.0f}s".format(
        sample_id, time_taken // 60, time_taken % 60))
