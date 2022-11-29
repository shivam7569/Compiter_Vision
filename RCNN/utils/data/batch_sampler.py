import random
from cv2 import randn
import cv2
import numpy as np
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from RCNN.globalParams import Global

from RCNN.utils.data.finetune_dataset import FineTuneDataset


class FineTuneSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative, shuffle):

        self.shuffle = shuffle
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = range(length)

        self.batch = batch_positive + batch_negative

        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = []

        for _ in range(self.num_iter):

            iter_indices = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            if self.shuffle:
                random.shuffle(iter_indices)

            sampler_list.extend(iter_indices)

        return iter(sampler_list)

    def __len__(self):
        return self.num_iter * self.batch

    def get_num_batches(self):
        return self.num_iter


def verifyDataLoader(num_samples):

    for _ in range(num_samples):

        selected_type = random.choice(["train/", "val/", "test/"])

        root_dir = Global.FINETUNE_DATA_DIR + selected_type

        transformation = A.Compose(
            [
                A.RGBShift(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.ChannelShuffle(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(height=224, width=224, always_apply=True),
                ToTensorV2()
            ]
        )

        data_set = FineTuneDataset(
            root_dir, transform=transformation, mode=selected_type.split("/")[0], debug=True)
        sampler = FineTuneSampler(data_set.get_positive_num(
        ), data_set.get_negative_num(), 32, 96, shuffle=True)
        data_loader = DataLoader(
            data_set, batch_size=128, sampler=sampler, num_workers=56, drop_last=True)

        for inputs, targets in data_loader:

            randNum = random.choice(range(32))
            t = targets[randNum].numpy()
            i = inputs[randNum]
            i = i.numpy()
            i = np.transpose(i, (1, 2, 0))

            if t == 0:
                continue
            else:
                name = Global.OUTPUT_DIR + \
                    f"data_loader_output/{Global.LABEL_TYPE[t]}_{selected_type[:-1]}.png"
                cv2.imwrite(name, i)
                print(
                    f"Saved example for {Global.LABEL_TYPE[t]} from {selected_type}")
                break
