# -*- coding: utf-8 -*-
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log
import coco


"""Directories to logs and pretrained models"""
ROOT_DIR = os.getcwd()

# TODO: Enter path where to save trained model
# Ether root dir or full path
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# MODEL_DIR = "D:\Eigene Dateien\Dokumente\mylogs" 

# TODO: Enter path to trained weights .h5 file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") # matterport model
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5") # superlee506 model
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose_lu.h5") # Lukas model

# TODO: Enter your path to COCO images and annotations
# coco/
# ├─test2017/
# ├─train2017/
# ├─val2017/
# ├─annotations/
COCO_DIR = "D:/coco"
#COCO_DIR = "D:/Eigene Dateien/Dokumente/coco"



"""Training parameters and loading annotations"""
# Hyperparameter settings
class TrainingConfig(coco.CocoConfig):
    USE_MINI_MASK = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 4000
    IMAGE_MAX_DIM = 704
    #TRAIN_ROIS_PER_IMAGE = 80
    #MAX_GT_INSTANCES = 64
    
    # Mask-R CNN paper config
    KEYPOINT_MASK_POOL_SIZE = 14
    RPN_NMS_THRESHOLD = 0.5

training_config = TrainingConfig()

# Load dataset
assert training_config.NAME == "coco"

# Training dataset
# load person keypoints dataset
train_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
train_dataset_keypoints.load_coco(COCO_DIR, "train")
train_dataset_keypoints.prepare()

# Validation dataset
# load person keypoints dataset
val_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
val_dataset_keypoints.load_coco(COCO_DIR, "val")
val_dataset_keypoints.prepare()

# Show number of training images
print("Train Keypoints Image Count: {}".format(len(train_dataset_keypoints.image_ids)))
print("Train Keypoints Class Count: {}".format(train_dataset_keypoints.num_classes))
for i, info in enumerate(train_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
# Show number of validations images
print("Val Keypoints Image Count: {}".format(len(val_dataset_keypoints.image_ids)))
print("Val Keypoints Class Count: {}".format(val_dataset_keypoints.num_classes))
for i, info in enumerate(val_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))



    
"""Create model object in training mode"""
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=training_config)

# Load weights trained on MS-COCO
# TODO: Select model

# COCO_MODEL_PATH = model.get_imagenet_weights()
# model.load_weights(COCO_MODEL_PATH, by_name=True)

# necessary when loading matterport model
# model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# necessary when loading Superlee506 model (new keypoint mask branch in model.py)
# model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_keypoint_mask_deconv"])

# necessary when loading LukasStu or Superlee506 (old keypoint branch in model.py) model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# loading last trained model in "mylogs"
# COCO_MODEL_PATH = model.find_last()[1]
# model.load_weights(COCO_MODEL_PATH, by_name=True)

print("Loading weights from ", COCO_MODEL_PATH)

# Show model layers in training mode
# model.keras_model.summary()

# Correction when continue training
x = 0

"""Train model -starting from heads"""
# Training - Stage 1
print("Train heads")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=training_config.LEARNING_RATE /5,
            epochs=100-x,
            layers='heads')
# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Training Resnet layer 4+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=training_config.LEARNING_RATE / 10,
            epochs=120-x,
            layers='4+')
# Training - Stage 3
# Finetune layers from ResNet stage 3 and up
print("Training Resnet layer 3+")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=training_config.LEARNING_RATE / 100,
            epochs=130-x,
            layers='all')