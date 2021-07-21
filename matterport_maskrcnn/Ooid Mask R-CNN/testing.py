"""
Mask R-CNN
Code to test a model on ooid images and calculate mAP.
Written by Adam Kelly, Modified by Hugh Shields

"""

import ooid
import numpy as np

# insert testing file and directories
TEST_FILE = 'Standard Annotation_Testing.3'
TEST_JSON_DIR = '.../Ooid Mask R-CNN/datasets/test/' + TEST_FILE + '.json'
TEST_IMAGE_DIR = '.../Ooid Mask R-CNN/datasets/test/images'
dataset_test = ooid.OoidDataset()
dataset_test.load_ooid(TEST_JSON_DIR, TEST_IMAGE_DIR)
dataset_test.prepare()

# set up testing config
class InferenceConfig(ooid.OoidConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048
    DETECTION_MIN_CONFIDENCE = 0.95

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = ooid.modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=ooid.DEFAULT_LOGS_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = ooid.os.path.join(ooid.ROOT_DIR, "C:/Users/hughs/Documents/Internships/HMEI Internship 2021/Ooid Mask R-CNN/saved logs/Edge Ooids_5.250/mask_rcnn_ooid_0005.h5")
model_path = model.find_last()
print(model_path)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

import skimage
real_test_dir = TEST_IMAGE_DIR
image_paths = []
for filename in ooid.os.listdir(real_test_dir):
    if ooid.os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
    # if os.path.splitext(filename)[1].lower() in ['.tif']:
        image_paths.append(ooid.os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    ooid.visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                dataset_test.class_names, r['scores'], figsize=(5,5))

# Compute VOC-Style mAP @ for IoU 0.50 to 0.95
# Running on 3 images in test diretory. Increase for better accuracy.
image_ids = dataset_test.image_ids
APs = []
iou_thresholds = np.arange(0.5, 1.0, 0.05)
for iou_threshold in iou_thresholds:
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            ooid.modellib.load_image_gt(dataset_test, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(ooid.modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            ooid.utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=iou_threshold)
        APs.append(AP)
    print("mAP " + str(np.round(iou_threshold*100)) + ": ", np.mean(APs))
