import os
from easydict import EasyDict

CONF = EasyDict()

# Main Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/home/jiachen/TOD2D'  # TODO: change this
#CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, 'data')

#Temp
CONF.PATH.DATA = os.path.join('/media/jiachen/LJC-2/TOD2D', 'data')
# DTLD
CONF.PATH.DATA_DTLD = os.path.join(CONF.PATH.DATA, 'data_DTLD')
CONF.PATH.LABELS_DTLD = os.path.join(CONF.PATH.DATA_DTLD, 'DTLD_Labels_v2.0/v2.0/DTLD_all.json') # TODO: change this if use different data
# BSLTD
CONF.PATH.DATA_BSLTD = os.path.join(CONF.PATH.DATA, 'data_BSLTD')

CONF.PATH.DATASET_YOLO = os.path.join(CONF.PATH.BASE, 'dataset_yolo')
CONF.PATH.DATASET_CLS = os.path.join(CONF.PATH.BASE, 'dataset_cls')
CONF.PATH.DEMO = os.path.join(CONF.PATH.BASE, 'demo')

# Data
CONF.data = EasyDict()

# Dataset_Yolo
CONF.dataset_yolo = EasyDict()
CONF.dataset_yolo.images_train = os.path.join(CONF.PATH.DATASET_YOLO, 'dtld/images/train')
CONF.dataset_yolo.images_val = os.path.join(CONF.PATH.DATASET_YOLO, 'dtld/images/val')
CONF.dataset_yolo.labels_train = os.path.join(CONF.PATH.DATASET_YOLO, 'dtld/labels/train')
CONF.dataset_yolo.labels_val = os.path.join(CONF.PATH.DATASET_YOLO, 'dtld/labels/val')
CONF.dataset_yolo.split_ratio = 0.9

# Dataset_Tlc_Classifier
CONF.dataset_tlc_classifier = EasyDict()
CONF.dataset_tlc_classifier.images = os.path.join(CONF.PATH.DATASET_CLS, 'images')
CONF.dataset_tlc_classifier.labels = os.path.join(CONF.PATH.DATASET_CLS, 'labels')
CONF.dataset_tlc_classifier.labels_txt_path = os.path.join(CONF.dataset_tlc_classifier.labels, 'cls.txt')

# Datamodule_Tlc_Classifier
CONF.datamodule_tlc_classifier = EasyDict()
CONF.datamodule_tlc_classifier.split_ratio = 0.9
CONF.datamodule_tlc_classifier.batch_size = 25

# Model_Tlc_Classifier
CONF.model_tlc_classifier = EasyDict()
CONF.model_tlc_classifier.freeze_layers = False
CONF.model_tlc_classifier.best_model = os.path.join(CONF.PATH.BASE, 'TLC/saved_models/best_model.ckpt')

