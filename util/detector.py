
import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
from PIL import Image
import PIL
import cv2
from util.cqrcode_attention_generator import CAM
import numpy as np
import torch.nn.functional as F
from models.experimental import attempt_load
from pdb import set_trace as st
import onnxruntime as ort
from util.detect import Detector

def simulate_underwater_effect_pil(pil_image):
    # 将 PIL 图像转换为 NumPy 数组
    np_img = np.array(pil_image)

    # 调用原先只支持 NumPy 数组的 simulate_underwater_effect
    # 模拟水下效果，返回 NumPy 数组
    np_underwater = simulate_underwater_effect(np_img)

    # 将处理后的 NumPy 数组转换回 PIL 图像
    return Image.fromarray(np_underwater)

def simulate_underwater_effect(image):
    # 读取图像


    # 以30%的概率不进行任何操作
    if np.random.rand() < 0.2:
        return image

    # 1. 模拟颜色失真
    if np.random.rand() < 0.5:
        color_matrix = np.diag(np.random.uniform(0.8, 1.2, 3))
        distorted_image = np.clip(np.dot(image / 255.0, color_matrix.T), 0, 1)
    else:
        distorted_image = image / 255.0

    # 2. 模拟模糊效果
    if np.random.rand() < 0.5:
        kernel_size = tuple(np.random.choice([ 1, 3,5], size=2))
        sigma = np.random.uniform(1, 5)
        blurred_image = cv2.GaussianBlur((distorted_image * 255).astype(np.uint8), kernel_size, sigma)
    else:
        blurred_image = (distorted_image * 255).astype(np.uint8)

    # 3. 降低对比度
    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.7, 1.2)  # 对比度缩减系数
        beta = np.random.randint(0, 30)      # 提升亮度
        low_contrast_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)
    else:
        low_contrast_image = blurred_image

    return low_contrast_image
def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


def preprocess_image(img, img_size):
    # Ensure img is a tensor
    if not isinstance(img, torch.Tensor):
        raise ValueError(f"Expected image to be a tensor, but got {type(img)}")
    
    # Resize the image
    img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    return img


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        # self.root = opt.dataroot
        # self.root = opt.dataroot
        self.src_dset=opt.src_dset
        self.tgt_dset=opt.tgt_dset
        assert os.path.exists(self.src_dset)
        assert os.path.exists(self.tgt_dset)
        self.dir_A = self.src_dset
        self.dir_B = self.tgt_dset

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        # self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        # self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # self.resize_transform = transforms.Resize((320, 320))
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

        self.transform = get_transform(opt)
        # model = attempt_load(opt.attention_model_path, map_location='cpu')
        # model.eval()
        # target_layer = model.model[-2]  # Change this to the layer you want to visualize
        # self.attention_generator = CAM(model, target_layer)
        self.detector=Detector()
        
        # Load the ONNX model
        # self.model_pb_path = "checkpoints/attention_model/best.onnx"
        # so = ort.SessionOptions()
        # self.net = ort.InferenceSession(self.model_pb_path, so)
        
        # # Define labels
        # self.dic_labels = {0: 'QRCode'}
        
        # # Model parameters
        # self.model_h = 320
        # self.model_w = 320
        # self.nl = 3
        # self.na = 3
        # self.stride = [8., 16., 32.]
        # anchors = [[10, 13, 16, 30, 33, 23],
        #            [30, 61, 62, 45, 59, 119],
        #            [116, 90, 156, 198, 373, 326]]
        # self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

    # def _make_grid(self, nx, ny):
    #     xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    #     return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    # def cal_outputs(self, outs):
    #     row_ind = 0
    #     grid = [np.zeros(1)] * self.nl
    #     for i in range(self.nl):
    #         h, w = int(self.model_h / self.stride[i]), int(self.model_w / self.stride[i])
    #         length = int(self.na * h * w)
    #         if grid[i].shape[0] != h * w:
    #             grid[i] = self._make_grid(w, h)
    #         outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
    #             grid[i], (self.na, 1))) * int(self.stride[i])
    #         outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
    #             self.anchor_grid[i], h * w, axis=0)
    #         row_ind += length
    #     return outs

    # def post_process_opencv(self, outputs, img_h, img_w, thred_nms=0.4, thred_cond=0.5):
    #     conf = outputs[:, 4].tolist()