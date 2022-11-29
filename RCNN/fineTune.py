from RCNN.utils.data.batch_sampler import FineTuneSampler
from RCNN.utils.data.finetune_dataset import FineTuneDataset
from RCNN.models.models import VGG19, AlexNet
from RCNN.globalParams import Global
import os
import copy
from time import sleep
import torch
import torch.nn as nn
import albumentations as A
from time import time
import multiprocessing as mp
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class FineTune:

    def __init__(self, debug=False):

        self.debug = debug
        self.tbWriter = SummaryWriter(Global.TENSORBOARD_LOG_DIR)

    def load_data(self, transformation):

        data_loaders = {}
        data_sizes = {}
        train_dataset = None
        train_sampler = None

        for phase in ["train", "val"]:
            data_dir = os.path.join(Global.FINETUNE_DATA_DIR, phase)
            dataset = FineTuneDataset(
                data_dir, transformation, phase, self.debug)
            data_sampler = FineTuneSampler(
                dataset.get_positive_num(),
                dataset.get_negative_num(),
                batch_positive=32, batch_negative=96, shuffle=True
            )
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=128,
                sampler=data_sampler,
                num_workers=8,
                drop_last=True
            )

            # if phase == "train":
            #     train_dataset = dataset
            #     train_sampler = data_sampler

            data_loaders[phase] = data_loader
            data_sizes[phase] = data_sampler.__len__()

        self.data_loaders = data_loaders
        self.data_sizes = data_sizes

        # print("\nComputing optimal number of workers for train data loading...")
        # self._get_optimal_workers(
        #     {"dataset": train_dataset, "sampler": train_sampler}
        # )

    def set_device(self):
        self.device = Global.TORCH_DEVICE

    def _get_optimal_workers(self, data_dict):

        # TODO

        workers = range(16, mp.cpu_count(), 4)
        for num_workers in tqdm(workers, unit="num_workers", total=len(workers)):
            dataloader = DataLoader(
                data_dict["dataset"],
                batch_size=128,
                sampler=data_dict["sampler"],
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )

            start = time()
            for _ in tqdm(range(3), unit="iterations", total=3):
                for _, _ in tqdm(enumerate(dataloader), unit="batches", total=len(dataloader)):
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(
                end - start, num_workers))

    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

    def resume_training(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        return model, optimizer, checkpoint["epoch"]

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, model, criterion, optimizer, lr_scheduler, epochs):

        best_model_weights = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        model.to(self.device)

        batch_counter_train = 0
        batch_counter_val = 0
        epoch_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch: {epoch+1}/{epochs}\n{'*' * 12}")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                dtLoaderLen = len(self.data_loaders[phase])

                with tqdm(self.data_loaders[phase], unit="batch", total=dtLoaderLen) as tepoch:
                    for imgs, labels in tepoch:

                        tepoch.set_description(f"Epoch: {epoch+1}/{phase}")

                        imgs = imgs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            model.zero_grad()
                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step()

                        step_loss = loss.item()
                        step_acc = torch.sum(
                            preds == labels.data) / imgs.shape[0]
                        running_loss += step_loss
                        running_corrects += step_acc

                        if phase == "train":
                            batch_counter = batch_counter_train
                        else:
                            batch_counter = batch_counter_val

                        current_lr = self._get_lr(optimizer)

                        if batch_counter % 5000 == 0:
                            if phase == "train":
                                self.tbWriter.add_scalar(f"a_train/Step Train Loss", round(
                                    step_loss, 3), batch_counter_train)
                                self.tbWriter.add_scalar(f"a_train/Step Train Acc", round(
                                    step_acc.item(), 3), batch_counter_train)
                                self.tbWriter.add_scalar(
                                    f"a_train/lr", current_lr, batch_counter_train)
                            else:
                                self.tbWriter.add_scalar(f"b_val/Step Val Loss", round(
                                    step_loss, 3), batch_counter_val)
                                self.tbWriter.add_scalar(f"b_val/Step Val Acc", round(
                                    step_acc.item(), 3), batch_counter_val)

                        if phase == "train":
                            batch_counter_train += 1
                        else:
                            batch_counter_val += 1

                        tepoch.set_postfix(lr=current_lr, loss=round(
                            step_loss, 3), accuracy=round(step_acc.item(), 3))

                        sleep(0.1)

                epoch_loss = running_loss / dtLoaderLen
                epoch_acc = running_corrects.item() / dtLoaderLen

                print(f"{phase} loss: {round(epoch_loss, 3)}")
                print(f"{phase} acc: {round(epoch_acc, 3)}")

                if phase == "train":
                    self.tbWriter.add_scalar(
                        f"a_train/Train Epoch Loss", round(epoch_loss, 3), epoch_counter)
                    self.tbWriter.add_scalar(
                        f"a_train/Train Epoch Acc", round(epoch_acc, 3), epoch_counter)
                else:
                    self.tbWriter.add_scalar(
                        f"b_val/Val Epoch Loss", round(epoch_loss, 3), epoch_counter)
                    self.tbWriter.add_scalar(
                        f"b_val/Val Epoch Acc", round(epoch_acc, 3), epoch_counter)

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }

                    checkpoint_name = f"epoch_{epoch}_val_acc_{round(best_acc, 4)}.pt"
                    torch.save(checkpoint, Global.CHECKPOINT_DIR +
                               checkpoint_name)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        avg_grad = torch.mean(param.grad)
                        self.tbWriter.add_scalar(
                            f"c_grad_avg/{name}", avg_grad.item(), epoch_counter)
                        self.tbWriter.add_histogram(
                            f"c_grad/{name}", param.grad.cpu().numpy(), epoch_counter)
                    if param.data is not None:
                        avg_weight = torch.mean(param.data)
                        self.tbWriter.add_scalar(
                            f"d_weight_avg/{name}", avg_weight.item(), epoch_counter)
                        self.tbWriter.add_histogram(
                            f"d_weight/{name}", param.data.cpu().numpy(), epoch_counter)

            epoch_counter += 1

        print(f"{'-' * 10}\nBest val acc: {round(best_acc, 3)}")
        print("Saving model...")

        best_model = {
            "state_dict": best_model_weights
        }
        best_model_path = Global.CHECKPOINT_DIR + f"best.pt"
        torch.save(best_model, best_model_path)


def performFineTuning(epochs=25, debug=False):

    model = AlexNet(pretrained=True)

    transformation = A.Compose(
        [
            A.AdvancedBlur(p=0.20),
            A.CLAHE(p=0.20),
            A.ChannelShuffle(p=0.15),
            A.FancyPCA(p=0.25),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Sharpen(p=0.3),
            A.Superpixels(p=0.15),
            A.RGBShift(p=0.15),
            A.Resize(
                height=Global.FINETUNE_IMAGE_SIZE[0], width=Global.FINETUNE_IMAGE_SIZE[0], always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ]
    )

    fineTune = FineTune(debug=debug)
    fineTune.load_data(transformation=transformation)
    fineTune.set_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=150000, mode="exp_range")

    fineTune.train(model, criterion, optimizer, lr_scheduler, epochs)


def loadBestFineTuneModel():
    fineTune = FineTune(debug=False)
    alexnet = AlexNet(pretrained=False)
    fineTune.load_model(alexnet, Global.BEST_FINETUNE_MODEL)

    return alexnet
