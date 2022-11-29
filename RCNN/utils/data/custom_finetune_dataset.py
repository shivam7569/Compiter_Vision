import json
from logging import root
import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from RCNN.globalParams import Global
import warnings
warnings.filterwarnings("error")


class FineTuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.transform = transform
        self.root_dir = root_dir

        samples = os.listdir(os.path.join(root_dir, "JPEGImages"))

        self.jpeg_images = [os.path.join(
            root_dir, "JPEGImages", i) for i in samples]
        positive_annot = [os.path.join(
            root_dir, "Annotations", i[:-4] + "_1" + ".json") for i in samples]
        negative_annot = [os.path.join(
            root_dir, "Annotations", i[:-4] + "_0" + ".json") for i in samples]

        self.positive_sizes = []
        self.negative_sizes = []
        self.positive_rects = []
        self.negative_rects = []

        self.positive_labels = []
        self.negative_labels = []

        for annot_path in positive_annot:
            with open(annot_path, "r") as f:
                annot_data = json.load(f)

            rects = np.array(
                [
                    [
                        int(i["proposal_coord"][0]),
                        int(i["proposal_coord"][1]),
                        int(i["proposal_coord"][2]),
                        int(i["proposal_coord"][3])
                    ] for i in annot_data
                ]
            )

            labels = np.array(
                [
                    [int(i["proposal_class"])] for i in annot_data
                ]
            )

            if len(rects.shape) == 1:  # Covers the cases when there is no or only one proposal rectangle
                # Confirms the case when there is only one proposal rectangle => shape === (4,)
                if rects.shape[0] == 4:
                    self.positive_rects.append(rects)
                    self.positive_labels.append(labels)
                    self.positive_sizes.append(1)
                # Confirms the case when there is no proposal rectangle => shape === (0,)
                else:
                    self.positive_sizes.append(0)

            else:
                self.positive_rects.extend(rects)
                self.positive_labels.extend(labels)
                self.positive_sizes.append(len(rects))

        for annot_path in negative_annot:

            with open(annot_path, "r") as f:
                annot_data = json.load(f)

            rects = np.array(
                [
                    [
                        int(i["proposal_coord"][0]),
                        int(i["proposal_coord"][1]),
                        int(i["proposal_coord"][2]),
                        int(i["proposal_coord"][3])
                    ] for i in annot_data
                ]
            )

            labels = np.array(
                [
                    [int(i["proposal_class"])] for i in annot_data
                ]
            )

            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    self.negative_rects.append(rects)
                    self.negative_labels.append(labels)
                    self.negative_sizes.append(1)
                else:
                    self.negative_sizes.append(0)

            else:
                self.negative_rects.extend(rects)
                self.negative_labels.extend(labels)
                self.negative_sizes.append(len(rects))

        self.total_positive_num = int(np.sum(self.positive_sizes))
        self.total_negative_num = int(np.sum(self.negative_sizes))

        assert len(self.jpeg_images) == len(self.positive_sizes)
        assert len(self.jpeg_images) == len(self.negative_sizes)
        assert len(self.positive_rects) == len(self.positive_labels)
        assert len(self.negative_rects) == len(self.negative_labels)

    def __getitem__(self, index):
        image_id = len(self.jpeg_images) - 1

        if index < self.total_positive_num:
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            target = self.positive_labels[index][0]

            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
                    break

            image_path = self.jpeg_images[image_id]
            img = cv2.imread(image_path)
            proposal = img[ymin: ymax, xmin: xmax]

        else:
            index = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[index]
            target = self.negative_labels[index][0]

            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= index < np.sum(self.negative_sizes[:(i + 1)]):
                    image_id = i
                    break

            image_path = self.jpeg_images[image_id]
            img = cv2.imread(image_path)
            proposal = img[ymin: ymax, xmin: xmax]

        if self.transform:
            proposal = self.transform(proposal)\

        return proposal, target

    def __len__(self):
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self):
        return self.total_positive_num

    def get_negative_num(self):
        return self.total_negative_num


def test(idx):
    root_dir = Global.FINETUNE_DATA_DIR + "/train/"
    train_data_set = FineTuneDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    image, target = train_data_set.__getitem__(idx)
    print(f'ID: {idx} target: {Global.LABEL_TYPE[target]}')

    cv2.imwrite(f"./{idx}.png", image)

test(159622)
test(155891)
test(155890)
test(4051)