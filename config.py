import os

ROOT_DIR = os.environ["HOME"]

CURRENT_DIR = os.path.abspath(os.getcwd())

DATASET_FOLDER = "/datasets"

_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER

DATASET_BIOVID = 'BIOVID'
BIOVID_PATH = _FER_DATASET_PATH + '/Biovid'
BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_img_red_classes'
BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_two_labels.txt'

TRAIN_SOURCE_AND_TARGET = 'Train both source and target'
TRAIN_ONLY_TARGET = 'Train only target'


# ---------------------  xxx  xxxx --------------------- #

