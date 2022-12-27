import os
import multiprocessing
from multiprocessing import Pool
import sys
from pycocotools.coco import COCO
from RCNN.utils.globalParams import Global
from RCNN.utils.data.finetune_data import process_data
from RCNN.utils.util import check_dir

##### Preparing fine tuning data #####


def createFineTuneData():
    check_dir(Global.FINETUNE_DATA_DIR)

    train_annot_path = Global.DATA_DIR + "annotations/instances_train2017.json"
    val_annot_path = Global.DATA_DIR + "annotations/instances_val2017.json"

    train_coco = COCO(train_annot_path)
    val_coco = COCO(val_annot_path)

    train_samples = train_coco.getImgIds()
    val_samples = val_coco.getImgIds()

    args_iter_train = [("train", train_coco, s) for s in train_samples]
    args_iter_val = [("val", val_coco, s) for s in val_samples]

    args_iter = args_iter_train + args_iter_val
    num_tasks = len(args_iter)

    multiprocessing.set_start_method("fork")
    cpu_cores = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_cores, maxtasksperchild=100)

    _ = list(pool.imap_unordered(process_data,
             args_iter, chunksize=num_tasks // cpu_cores))
             