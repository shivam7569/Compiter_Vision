import cv2
import torch


class Global:

    MODEL_PATH = "./RCNN/models/"
    OUTPUT_DIR = "./RCNN/res/"
    DATA_DIR = "./RCNN/data/"
    FINETUNE_DATA_DIR = "./RCNN/data/finetune/"
    CLASSIFIER_DATA_DIR = "./RCNN/data/classifier/"
    RESOURCE_DIR = "./RCNN/resources/"
    RCNN_TENSORBOARD_LOG_DIR = OUTPUT_DIR + "tensorboard/RCNN/"
    IMAGENET_TENSORBOARD_LOG_DIR = OUTPUT_DIR + "tensorboard/IMAGENET/"
    CHECKPOINT_DIR = MODEL_PATH + "./checkpoints/"

    LABEL_TYPE = [i.lower() for i in
                  ["none", "aeroplane", "Bicycle", "bird", "Boat", "Bottle", "Bus", "Car",
                   "Cat", "Chair", "cow", "Diningtable", "Dog", "Horse", "Motorbike", "person",
                   "Pottedplant", "sheep", "Sofa", "Train", "TVmonitor"]
                  ]

    NUM_CLASSES = len(LABEL_TYPE)

    CLASS_LABELS = {}

    for idx, x, in enumerate(LABEL_TYPE):
        CLASS_LABELS[x.lower()] = idx

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

    FINETUNE_IMAGE_SIZE = (227, 227)
    FINETUNE_BATCH_SIZE = 128
    FINETUNE_POSITIVE_SAMPLES = 64
    FINETUNE_NEGATIVE_SAMPLES = 64

    GPU_ID = 6
    TORCH_DEVICE = torch.device(
        f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ALEXNET_WEIGHTS = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/alexnet_weights/alexnet.pth"
    BEST_FINETUNE_MODEL = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/checkpoints/epoch_8_val_acc_0.6606.pt"
