import os
import multiprocessing
from multiprocessing import Pool
from RCNN.globalParams import Global
from RCNN.utils.data.finetune_data import process_data
from RCNN.utils.util import check_dir

##### Preparing fine tuning data #####


def createFineTuneData():
    check_dir(Global.FINETUNE_DATA_DIR)

    train_samples_path = os.path.join(Global.DATA_DIR, "train/JPEGImages/")
    test_samples_path = os.path.join(Global.DATA_DIR, "test/JPEGImages/")
    val_samples_path = os.path.join(Global.DATA_DIR, "val/JPEGImages/")

    train_samples = [s.split("/")[-1].split(".")[0]
                     for s in os.listdir(train_samples_path)]
    test_samples = [s.split("/")[-1].split(".")[0]
                    for s in os.listdir(test_samples_path)]
    val_samples = [s.split("/")[-1].split(".")[0]
                   for s in os.listdir(val_samples_path)]

    args_iter_train = [("train", s) for s in train_samples]
    args_iter_test = [("test", s) for s in test_samples]
    args_iter_val = [("val", s) for s in val_samples]

    args_iter = args_iter_train + args_iter_test + args_iter_val
    num_tasks = len(args_iter)

    multiprocessing.set_start_method("fork")
    cpu_cores = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_cores, maxtasksperchild=100)

    _ = list(pool.imap_unordered(process_data,
             args_iter, chunksize=num_tasks // 8))
