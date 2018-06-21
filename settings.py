import os
import logging
import sys

BASE_DIR = "/media/mdt/data/deeplearning/"
WORKING_DIR = os.getcwd() + "/"
print("Current working directory is ", WORKING_DIR)

DR_SRC_DIR = BASE_DIR + "kaggle_DiabeticRetinopathy/"      # kaggle_DiabeticRetinopathy/train  test


def getlogger(name):
    mylogger = logging.getLogger(name)

    formatter = logging.Formatter('%(thread)d - %(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not mylogger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)

        filehandler = logging.FileHandler(name + '.log')
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.DEBUG)

        mylogger.addHandler(handler)
        mylogger.addHandler(filehandler)
        mylogger.setLevel(logging.DEBUG)
        mylogger.propagate = False
    return mylogger


def cleanlogger(logger):
    x = list(logger.handlers)
    for i in x:
        logger.removeHandler(i)
        i.flush()
        i.close()