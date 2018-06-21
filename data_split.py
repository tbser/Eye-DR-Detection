# coding:utf-8
import pandas as pd
import os
import shutil
import pdb
import settings
import glob, random

logger = settings.getlogger("data_split")


def split_traindata():
    data_read = pd.read_csv(settings.DR_SRC_DIR + 'trainLabels.csv')
    for row in data_read.itertuples():
        ok = os.path.isfile(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg')

        logger.info("row:{0}".format(row))    # row:Pandas(Index=0, image='10_left', level=0)
        # pdb.set_trace()
        for i in range(5):    # 0 1 2 3 4
            filepath = settings.WORKING_DIR + 'dataset/train/' + str(i) + '/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)

        if ok:
            if row.level == 0:
                shutil.copy2(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg',
                             settings.WORKING_DIR + 'dataset/train/0/' + str(row.image) + '.jpeg')
            elif row.level == 1:
                shutil.copy2(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg',
                             settings.WORKING_DIR + 'dataset/train/1/' + str(row.image) + '.jpeg')
            elif row.level == 2:
                shutil.copy2(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg',
                             settings.WORKING_DIR + 'dataset/train/2/' + str(row.image) + '.jpeg')
            elif row.level == 3:
                shutil.copy2(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg',
                             settings.WORKING_DIR + 'dataset/train/3/' + str(row.image) + '.jpeg')
            else:
                shutil.copy2(settings.DR_SRC_DIR + 'train/' + str(row.image) + '.jpeg',
                             settings.WORKING_DIR + 'dataset/train/4/' + str(row.image) + '.jpeg')
        else:
            logger.info("not found")


# train data and validation data
def split_train_val(train_percentage=80):
    source_dir = settings.WORKING_DIR + "dataset/resized_512x512_300_train/"
    des_dir = settings.WORKING_DIR + "dataset_pre/"
    for i in range(5):
        filepath = source_dir + str(i) + "/"
        despath_train = des_dir + "train/" + str(i) + "/"
        despath_val = des_dir + "val/" + str(i) + "/"
        if not os.path.exists(despath_train):
            os.makedirs(despath_train)
        if not os.path.exists(despath_val):
            os.makedirs(despath_val)
        samples = glob.glob(filepath + "*.jpeg")
        logger.info("samples: {}".format(len(samples)))
        random.shuffle(samples)
        train_count = int((len(samples) * train_percentage) / 100)
        samples_train = samples[:train_count]
        samples_val = samples[train_count:]

        for file in samples_train:
            fpath, fname = os.path.split(file)
            shutil.copy(file, despath_train + fname)

        for file in samples_val:
            fpath, fname = os.path.split(file)
            shutil.copy(file, despath_val + fname)


if __name__ == '__main__':
    # split_traindata()
    split_train_val(train_percentage=90)
