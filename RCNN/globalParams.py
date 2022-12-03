import cv2
import torch


class Global:

    MODEL_PATH = "./RCNN/models/"
    OUTPUT_DIR = "./RCNN/res/"
    DATA_DIR = "./RCNN/data/"
    FINETUNE_DATA_DIR = "./RCNN/data/finetune/"
    RESOURCE_DIR = "./RCNN/resources/"
    TENSORBOARD_LOG_DIR = OUTPUT_DIR + "tensorboard/"
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

    NUM_PROPOSALS = 2000
    PROPOSAL_BBOX_COLOR = (0, 255, 0)
    BBOX_THICKNESS = 3
    PROPOSAL_BBOX_THICKNESS = 1
    BBOX_COLOR = (0, 0, 255)
    IMG_TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
    IMG_TEXT_THICKNESS = 1
    IMG_TEXT_LINE_TYPE = 1
    IMG_TEXT_FONT_SCALE = 1

    FINETUNE_IMAGE_SIZE = (224, 224)

    GPU_ID = 6
    TORCH_DEVICE = torch.device(
        f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    ALEXNET_WEIGHTS = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/alexnet_weights/alexnet.pth"
    BEST_FINETUNE_MODEL = "/home/hqh2kor/handsOn/ObjectDetection/RCNN/models/checkpoints/epoch_2_val_acc_0.7727.pt"
