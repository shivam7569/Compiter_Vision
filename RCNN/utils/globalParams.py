import cv2
import torch


class Global:

    MODEL_PATH = "./RCNN/models/"
    OUTPUT_DIR = "./RCNN/res/"
    DATA_DIR = "./RCNN/data/coco2017/"
    FINETUNE_DATA_DIR = "./RCNN/data/finetune/"
    CLASSIFIER_DATA_DIR = "./RCNN/data/classifier/"
    RESOURCE_DIR = "./RCNN/resources/"
    RCNN_TENSORBOARD_LOG_DIR = OUTPUT_DIR + "tensorboard/RCNN/"
    IMAGENET_TENSORBOARD_LOG_DIR = OUTPUT_DIR + "tensorboard/IMAGENET/"
    CHECKPOINT_DIR = MODEL_PATH + "./checkpoints/"

    CLASS_LABELS = {
        0: "none",
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
        18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
        24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
        32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
        37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
        41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
        46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
        51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
        56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
        61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
        67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
        75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
        80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
        86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
    }

    MAPPED_CLASS_LABELS = {
        0: 0,
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
        18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22,
        24: 23, 25: 24, 27: 25, 28: 26, 31: 27,
        32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
        37: 33, 38: 34, 39: 35, 40: 36,
        41: 37, 42: 38, 43: 39, 44: 40,
        46: 41, 47: 42, 48: 43, 49: 44, 50: 45,
        51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
        56: 51, 57: 52, 58: 53, 59: 54, 60: 55,
        61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
        67: 61, 70: 62, 72: 63, 73: 64, 74: 65,
        75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
        80: 71, 81: 72, 82: 73, 84: 74, 85: 75,
        86: 76, 87: 77, 88: 78, 89: 79, 90: 80
    }

    NUM_CLASSES = len(CLASS_LABELS.keys())

    NUM_PROPOSALS = 3000
    PROPOSAL_BBOX_COLOR = (0, 255, 0)
    BBOX_THICKNESS = 3
    PROPOSAL_BBOX_THICKNESS = 1
    BBOX_COLOR = (0, 0, 255)
    IMG_TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    IMG_TEXT_THICKNESS = 1
    IMG_TEXT_LINE_TYPE = 1
    IMG_TEXT_FONT_SCALE = 1
    IMG_TEXT_COLOR = (0, 0, 0)

    FINETUNE_IMAGE_SIZE = (224, 224)
    FINETUNE_BATCH_SIZE = 128
    FINETUNE_POSITIVE_SAMPLES = 32
    FINETUNE_NEGATIVE_SAMPLES = 96

    GPU_ID = 6
    TORCH_DEVICE = torch.device(
        f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ALEXNET_WEIGHTS = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/alexnet_weights/alexnet.pth"
    BEST_FINETUNE_MODEL = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/checkpoints/epoch_1_val_acc_0.6107.pt"
