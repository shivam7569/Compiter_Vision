import json
import random
import numpy as np
import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from RCNN.utils.globalParams import Global
import warnings
warnings.filterwarnings("error")


class FineTuneDataset(Dataset):

    def __init__(self, root_dir, transform=None, phase="train", debug=False):
        super().__init__()

        self.transform = transform
        self.root_dir = root_dir
        self.phase = phase

        samples = list(set([i.split(
            "_")[0] + ".jpg" for i in os.listdir(os.path.join(root_dir, "Annotations"))]))

        if debug:
            samples = random.sample(samples, 200)

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

        self.positive_image_names = []
        self.negative_image_names = []

        print()
        pbar = tqdm(positive_annot, desc=f"Loading Positive Annotations ({phase})", total=len(
            positive_annot))
        for annot_path in pbar:
            with open(annot_path, "r") as f:
                annot_data = json.load(f)

            data = np.array(annot_data["proposal_coord"], dtype=np.int32)

            if data.shape == (0,):
                continue

            rects = data[:, :-1]
            labels = np.reshape(data[:, -1], newshape=(-1, 1))
            img_names = np.array([[annot_data["image_name"]]
                                 for _ in range(labels.shape[0])])

            if len(rects.shape) == 1:  # Covers the cases when there is no or only one proposal rectangle
                # Confirms the case when there is only one proposal rectangle => shape === (4,)
                if rects.shape[0] == 4:
                    self.positive_rects.append(rects)
                    self.positive_labels.append(labels)
                    self.positive_image_names.append(img_names)
                    self.positive_sizes.append(1)
                # Confirms the case when there is no proposal rectangle => shape === (0,)
                else:
                    self.positive_sizes.append(0)

            else:
                self.positive_rects.extend(rects)
                self.positive_labels.extend(labels)
                self.positive_image_names.extend(img_names)
                self.positive_sizes.append(len(rects))

        pbar = tqdm(negative_annot, desc=f"Loading Negative Annotations ({phase})", total=len(
            negative_annot))
        for annot_path in pbar:

            with open(annot_path, "r") as f:
                annot_data = json.load(f)

            data = np.array(annot_data["proposal_coord"], dtype=np.int32)

            if data.shape == (0,):
                continue

            rects = data[:, :-1]
            labels = np.reshape(data[:, -1], newshape=(-1, 1))
            img_names = np.array([[annot_data["image_name"]]
                                 for _ in range(labels.shape[0])])

            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    self.negative_rects.append(rects)
                    self.negative_labels.append(labels)
                    self.negative_image_names.append(img_names)
                    self.negative_sizes.append(1)
                else:
                    self.negative_sizes.append(0)

            else:
                self.negative_rects.extend(rects)
                self.negative_labels.extend(labels)
                self.negative_image_names.extend(img_names)
                self.negative_sizes.append(len(rects))

        self.total_positive_num = int(np.sum(self.positive_sizes))
        self.total_negative_num = int(np.sum(self.negative_sizes))

        assert len(self.positive_rects) == len(self.positive_labels)
        assert len(self.negative_rects) == len(self.negative_labels)
        assert len(self.positive_image_names) == len(self.positive_rects)
        assert len(self.negative_image_names) == len(self.negative_rects)

    def __getitem__(self, index):

        if index < self.total_positive_num:
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            target = self.positive_labels[index][0]
            img_name = self.positive_image_names[index][0]

            image_path = os.path.join(
                Global.DATA_DIR, self.phase, img_name + ".jpg")
            img = cv2.imread(image_path)
            proposal = img[ymin: ymax, xmin: xmax]

        else:
            index = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[index]
            target = self.negative_labels[index][0]
            img_name = self.negative_image_names[index][0]

            image_path = os.path.join(
                Global.DATA_DIR, self.phase, img_name + ".jpg")
            img = cv2.imread(image_path)
            proposal = img[ymin: ymax, xmin: xmax]

        if self.transform:
            transformed_proposal = self.transform(image=proposal)
            proposal = transformed_proposal["image"]
        else:
            proposal = cv2.resize(proposal, Global.FINETUNE_IMAGE_SIZE)

        return proposal, Global.MAPPED_CLASS_LABELS[target]

    def __len__(self):
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self):
        return self.total_positive_num

    def get_negative_num(self):
        return self.total_negative_num
