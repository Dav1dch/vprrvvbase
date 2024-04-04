# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
from misc.utils import MinkLocParams
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import cv2 as cv

from models.model_factory import model_factory


rgb_config = "./config/config_baseline_rgb.txt"
rgb_model_config = "./models/minklocrgb.txt"
rgb_weights = "./weights/model_MinkLocRGB_20240324_1942_fire_best.pth"


class ValRGBTransform:
    def __init__(self):
        # 1 is default mode, no transform
        t = [
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        e = self.transform(e)
        return e


def ts_from_filename(filename):
    # Extract timestamp (as integer) from the file path/name
    temp = os.path.split(filename)[1]
    lidar_ts = os.path.splitext(temp)[0]  # LiDAR timestamp
    return lidar_ts


def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    # lidar_ts, traversal = ts_from_filename(filename)
    lidar_ts = ts_from_filename(filename)
    image_file_path = os.path.join(image_path, filename)
    # image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'

    img = Image.open(image_file_path)
    # import cv2
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # query_img = Image.open(query_filename).convert('RGB')
    return img


def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(file_name)

    result = {}

    if params.use_rgb:
        img = Image.open(file_path)
        transform = ValRGBTransform()
        # Convert to tensor and normalize
        result["image"] = transform(img)

    return result


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device: {}".format(device))

rgb_params = MinkLocParams(rgb_config, rgb_model_config)


# # load minkloc
rgb_mink_model = model_factory(rgb_params)
rgb_mink_model.load_state_dict(torch.load(rgb_weights, map_location=device))
rgb_mink_model.eval()
img_dir = "/home/david/datasets/fire/color"
img_list = os.listdir(img_dir)
img_path = "/home/david/datasets/fire/color/frame-000109.color.png"
# img_path = "/home/david/datasets/kitti/color/002001.png"

# img_path_m = '/home/graham/datasets/dataset/color/frame-000832.color.png'
# img_path = '/home/graham/datasets/dataset/color/frame-000833.color.png'
# img_path_p = '/home/graham/datasets/dataset/color/frame-000834.color.png'
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    ClassifierOutputSoftmaxTarget,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

print(rgb_mink_model)
model = resnet50(pretrained=True)
# target_layers = [rgb_mink_model.image_fe.fh_conv1x1["4"]]
target_layers = [rgb_mink_model.image_fe.fh_conv1x1["4"]]
# target_layers = [rgb_mink_model.image_fe.fh_conv1x1['1'],rgb_mink_model.image_fe.fh_conv1x1['2'], rgb_mink_model.image_fe.fh_conv1x1['3'], rgb_mink_model.image_fe.fh_conv1x1['4'] ]
# target_layers = [model.layer4[-1]]

# img_path = os.path.join(img_dir, i)
img = load_data_item(img_path, rgb_params)
input_tensor = img["image"].unsqueeze(0).cuda()
# input_tensor = img['image'].unsqueeze(0).to('cuda')
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!
#
# Construct the CAM object once, and then re-use it on many images:
cam = EigenCAM(model=rgb_mink_model, target_layers=target_layers)
# cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
#
#
targets = [ClassifierOutputTarget(255)]
#
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#
# In this example grayscale_cam has only one image in the batch:
print(grayscale_cam.shape)
grayscale_cam = grayscale_cam[0, :]
img_ = cv2.imread(img_path)
img_ = np.float32(img_ / 255)
print(grayscale_cam.shape)
visualization = show_cam_on_image(img_, grayscale_cam)
cv2.imwrite("./viz_kitti.png", visualization)
