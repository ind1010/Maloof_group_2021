"""
Mask R-CNN
Code to train a model on the ooid dataset.
Written by Adam Kelly, Modified by Hugh Shields

"""

import ooid
import time

# adjust config parameters for specific trainings
ooid.IMAGE_MIN_DIM = 1024
ooid.IMAGE_MAX_DIM = 1024
ooid.OoidConfig.STEPS_PER_EPOCH = 250

# insert training files and directories
TRAIN_FILE = 'Standard Annotation_Training.3'
TRAIN_JSON_DIR = '.../Ooid Mask R-CNN/datasets/train/' + TRAIN_FILE + '.json'
TRAIN_IMAGE_DIR = '.../Ooid Mask R-CNN/datasets/train/images'
dataset_train = ooid.OoidDataset()
dataset_train.load_ooid(TRAIN_JSON_DIR,TRAIN_IMAGE_DIR)
dataset_train.prepare()

# insert validation files and directories
VAL_FILE = 'Standard Annotation_Validation.2'
VAL_JSON_DIR = '.../Ooid Mask R-CNN/datasets/val/' + VAL_FILE +'.json'
VAL_IMAGE_DIR = '.../Ooid Mask R-CNN/datasets/val/images'
dataset_val = ooid.OoidDataset()
dataset_val.load_ooid(VAL_JSON_DIR, VAL_IMAGE_DIR)
dataset_val.prepare()

# Create model in training mode
model = ooid.modellib.MaskRCNN(mode="training", config=ooid.config,
                          model_dir=ooid.DEFAULT_LOGS_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(ooid.COCO_WEIGHTS_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val,
            learning_rate=ooid.config.LEARNING_RATE,
            epochs=5,
            layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')
