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


def parse_annotation_jpeg(annotation_path, jpeg_path, ss):
    """ 
    Get positive and negative samples (note: ignore the label bounding box whose property difficult is True) 
    Positive sample: Candidate suggestion and labeled bounding box IoU is greater than or equal to 0.5 
    Negative sample: IoU is greater than 0 and less than 0.5. In order to further limit the number of negative samples, its size must be larger than 1/5 of the label box 
    """

    image_name = jpeg_path.split("/")[-1].split(".")[0]
    img = cv2.imread(jpeg_path)
    height, width, _ = img.shape
    ss.loadAlgo(img)

    rects = ss.getAnchors()
    np.random.shuffle(rects)

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
                "image_name": image_name,
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
                "image_name": image_name,
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

    num_positives = len(positive_list)
    if len(negative_list) > num_positives: negative_list = random.sample(negative_list, num_positives)

    return positive_list, negative_list


def process_data(args):

    directory, sample_name = args
    ss = selectiveSearch()

    src_root_dir = os.path.join(Global.DATA_DIR, directory)
    src_annotation_dir = os.path.join(src_root_dir, "Annotations")
    src_jpeg_dir = os.path.join(src_root_dir, "JPEGImages")

    assert len(os.listdir(src_annotation_dir)) == len(os.listdir(src_jpeg_dir))

    dst_root_dir = os.path.join(Global.FINETUNE_DATA_DIR, directory)
    dst_annotation_dir = os.path.join(dst_root_dir, "Annotations")
    dst_jpeg_dir = os.path.join(dst_root_dir, "JPEGImages")

    check_dir(dst_root_dir)
    check_dir(dst_annotation_dir)
    check_dir(dst_jpeg_dir)

    start_time = time.time()

    src_annotation_path = os.path.join(
        src_annotation_dir, sample_name + ".xml")
    src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + ".jpg")
    dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")

    positive_list, negative_list = parse_annotation_jpeg(
        src_annotation_path, src_jpeg_path, ss)

    dst_positive_annot_path = os.path.join(
        dst_annotation_dir, sample_name + "_1" + ".json")
    dst_negative_annot_path = os.path.join(
        dst_annotation_dir, sample_name + "_0" + ".json")

    shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    with open(dst_positive_annot_path, "w") as f:
        json.dump(positive_list, f)

    with open(dst_negative_annot_path, "w") as f:
        json.dump(negative_list, f)

    end_time = time.time()
    time_taken = end_time - start_time

    print("Parsed {}.png in {:.0f}m {:.0f}s".format(
        sample_name, time_taken // 60, time_taken % 60))
