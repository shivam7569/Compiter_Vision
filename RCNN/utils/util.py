import json
import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from RCNN.utils.globalParams import Global


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_xml(xml_path):
    object_list = []

    tree = ET.parse(open(xml_path, "r"))
    root = tree.getroot()

    size = root.find("size")
    height, width = int(size.findtext("height")), int(size.findtext("width"))

    objects = root.findall("object")

    for obj in objects:
        bndbox = obj.find("bndbox")
        xmin = round(int(float(bndbox.findtext("xmin"))) / width, 3)
        ymin = round(int(float(bndbox.findtext("ymin"))) / height, 3)
        xmax = round(int(float(bndbox.findtext("xmax"))) / width, 3)
        ymax = round(int(float(bndbox.findtext("ymax"))) / height, 3)
        class_name = obj.findtext("name")

    object_list.append(
        {
            "x1": xmin,
            "x2": xmax,
            "y1": ymin,
            "y2": ymax,
            "class_name": class_name,
            "class_id": Global.CLASS_LABELS[class_name]
        }
    )

    return object_list


def save_model(model, save_name):
    check_dir(Global.MODEL_PATH)
    torch.save(model.state_dict(), Global.MODEL_PATH + save_name)


def plot_loss(x, y, name):
    _ = plt.figure()

    plt.plot(x, y)
    plt.title(name)
    plt.savefig(Global.OUTPUT_DIT + f"{name}.png")


def argmax(iterable):
    max_index, max_value = max(enumerate(iterable), key=lambda x: x[1])
    return (max_index, max_value)


def verifyDataGeneration(num_samples):

    types = ["train", "val", "test"]

    for _ in range(num_samples):
        selected_type = random.choice(types)

        images = os.listdir(os.path.join(
            Global.FINETUNE_DATA_DIR, selected_type, "JPEGImages"))
        img_name = random.choice(images).split(".")[0]

        img = cv2.imread(os.path.join(Global.FINETUNE_DATA_DIR,
                         selected_type, "JPEGImages", img_name + ".jpg"))

        with open(os.path.join(Global.FINETUNE_DATA_DIR, selected_type, "Annotations", img_name + "_1.json"), "r") as f:
            positive_proposals = json.load(f)

        with open(os.path.join(Global.FINETUNE_DATA_DIR, selected_type, "Annotations", img_name + "_0.json"), "r") as f:
            negative_proposals = json.load(f)

        count = 0
        for pp in positive_proposals:
            x1, y1, x2, y2 = [int(i) for i in pp["proposal_coord"]]
            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                Global.PROPOSAL_BBOX_COLOR, Global.PROPOSAL_BBOX_THICKNESS)
            img = cv2.putText(
                img,
                pp["proposal_class"],
                (x1, y1-10),
                Global.IMG_TEXT_FONT,
                Global.IMG_TEXT_FONT_SCALE,
                Global.PROPOSAL_BBOX_COLOR,
                Global.IMG_TEXT_THICKNESS,
                Global.IMG_TEXT_LINE_TYPE)
            if count == 0:
                gts = pp["gts"]
                for gt in gts:
                    xmin, ymin, xmax, ymax, class_id = [int(i) for i in gt]
                    img = cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax), Global.BBOX_COLOR, Global.BBOX_THICKNESS)
                    img = cv2.putText(
                        img,
                        str(class_id),
                        (xmin, ymin-10),
                        Global.IMG_TEXT_FONT,
                        Global.IMG_TEXT_FONT_SCALE,
                        Global.BBOX_COLOR,
                        Global.IMG_TEXT_THICKNESS,
                        Global.IMG_TEXT_LINE_TYPE)
                count = 1

        cv2.imwrite(os.path.join(Global.OUTPUT_DIR,
                    "region_proposals/", img_name + "_pp.png"), img)

        img = cv2.imread(os.path.join(Global.FINETUNE_DATA_DIR,
                         selected_type, "JPEGImages", img_name + ".jpg"))

        count = 0
        for np in negative_proposals:
            x1, y1, x2, y2 = [int(i) for i in np["proposal_coord"]]
            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                Global.PROPOSAL_BBOX_COLOR, Global.PROPOSAL_BBOX_THICKNESS)
            img = cv2.putText(
                img,
                np["proposal_class"],
                (x1, y1-10),
                Global.IMG_TEXT_FONT,
                Global.IMG_TEXT_FONT_SCALE,
                Global.PROPOSAL_BBOX_COLOR,
                Global.IMG_TEXT_THICKNESS,
                Global.IMG_TEXT_LINE_TYPE)
            if count == 0:
                gts = np["gts"]
                for gt in gts:
                    xmin, ymin, xmax, ymax, class_id = [int(i) for i in gt]
                    img = cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax), Global.BBOX_COLOR, Global.BBOX_THICKNESS)
                    img = cv2.putText(
                        img,
                        str(class_id),
                        (xmin, ymin-10),
                        Global.IMG_TEXT_FONT,
                        Global.IMG_TEXT_FONT_SCALE,
                        Global.BBOX_COLOR,
                        Global.IMG_TEXT_THICKNESS,
                        Global.IMG_TEXT_LINE_TYPE)
                count = 1

        cv2.imwrite(os.path.join(Global.OUTPUT_DIR,
                    "region_proposals/", img_name + "_np.png"), img)
        print(f"Saved image {img_name} examples from {selected_type}")


def image_grid(train_images, preds):

  figure = plt.figure(figsize=(10,10))
  for i in range(25):

    plt.subplot(5, 5, i + 1, title=Global.LABEL_TYPE[preds[i]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)

  return figure