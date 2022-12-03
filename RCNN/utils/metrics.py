import numpy as np


def iou(pred_box, target_boxes):
    "Compute the IoU of candidate proposal and labeled bounding boxes"

    if len(target_boxes.shape) == 1:
        target_boxes = target_boxes[np.newaxis, :]
    
    xA = np.maximum(pred_box[0], target_boxes[:, 0])
    yA = np.maximum(pred_box[1], target_boxes[:, 1])
    xB = np.minimum(pred_box[2], target_boxes[:, 2])
    yB = np.minimum(pred_box[3], target_boxes[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)

    return scores