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
import model_new as modellib
from model_new import log


ROOT_DIR = os.getcwd()
#MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
MODEL_DIR = "D:\Eigene Dateien\Dokumente\mylogs"
# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") # matterport weights
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5") # superlee506 weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose_lu.h5") # Lukas weights

# MS COCO Dataset
import coco

class TrainingConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # changed to 1. GPU to small for 2 images
    IMAGE_MAX_DIM = 512 # was 1024
    # Mask-R CNN paper config
    KEYPOINT_MASK_POOL_SIZE = 14
    DETECTION_NMS_THRESHOLD = 0.5

training_config = TrainingConfig()

COCO_DIR = "D:/Eigene Dateien/Dokumente/coco"  # TODO: enter your own path here

# Load dataset
assert training_config.NAME == "coco"
# Training dataset
# load person keypoints dataset
train_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
train_dataset_keypoints.load_coco(COCO_DIR, "train")
train_dataset_keypoints.prepare()

#Validation dataset
val_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
val_dataset_keypoints.load_coco(COCO_DIR, "val")
val_dataset_keypoints.prepare()

print("Train Keypoints Image Count: {}".format(len(train_dataset_keypoints.image_ids)))
print("Train Keypoints Class Count: {}".format(train_dataset_keypoints.num_classes))
for i, info in enumerate(train_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Val Keypoints Image Count: {}".format(len(val_dataset_keypoints.image_ids)))
print("Val Keypoints Class Count: {}".format(val_dataset_keypoints.num_classes))
for i, info in enumerate(val_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
# Create model object in training mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=training_config)

# Load weights trained on MS-COCO

# only necessary when loading matterport weights
# model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# only necessary when loading Superlee weights with old keypoint mask branch
# model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_keypoint_mask_deconv"])

# only necessary when loading Lukas weights
model.load_weights(COCO_MODEL_PATH, by_name=True)

# print("Loading weights from ", COCO_MODEL_PATH)
# model.keras_model.summary()

# Training - Stage 1 #15 Epochs
# print("Train heads")
# model.train(train_dataset_keypoints, val_dataset_keypoints,
#             learning_rate=training_config.LEARNING_RATE,
#             epochs=15,
#             layers='heads')
# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
#print("Training Resnet layer 4+")
#model.train(train_dataset_keypoints, val_dataset_keypoints,
#            learning_rate=training_config.LEARNING_RATE / 10,
#            epochs=20,
#            layers='4+')
# Training - Stage 3
# Finetune layers from ResNet stage 3 and up
print("Training Resnet layer 3+") #100 Epochs
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=training_config.LEARNING_RATE / 100,
            epochs=98,
            layers='all')