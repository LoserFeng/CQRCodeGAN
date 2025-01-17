import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transforms #注意这里是get_transforms
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
from torchvision.transforms.functional import crop, resize

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


def crop_and_resize(img: torch.Tensor, box, fine_size):
    """
    对输入的 PyTorch 张量 img 进行裁剪和调整大小。

    Args:
        img (torch.Tensor): 输入图像，形状为 (C, H, W)。
        box (tuple): 裁剪框 (box_x1, box_y1, box_x2, box_y2)。
        fine_size (int): 调整后的目标大小。

    Returns:
        torch.Tensor: 裁剪并调整大小后的图像，形状为 (C, fine_size, fine_size)。
    """
    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    min_side = min(box_width, box_height)

    # 确定随机裁剪大小
    crop_size = random.randint(max(min_side // 2, 64), min(min_side, fine_size))

    # 调整裁剪边界，确保覆盖 QR 码位置且不超过图像尺寸
    crop_x1 = max(0, box_x1 - random.randint(0, crop_size // 4))
    crop_y1 = max(0, box_y1 - random.randint(0, crop_size // 4))
    crop_x2 = min(img.shape[2], crop_x1 + crop_size)  # width
    crop_y2 = min(img.shape[1], crop_y1 + crop_size)  # height

    # 确保裁剪区域仍覆盖 QR 码位置
    crop_x1 = min(crop_x1, box_x1)
    crop_y1 = min(crop_y1, box_y1)
    crop_x2 = max(crop_x2, box_x2)
    crop_y2 = max(crop_y2, box_y2)

    # 提取裁剪区域
    cropped_img = crop(img, top=crop_y1, left=crop_x1, height=crop_y2 - crop_y1, width=crop_x2 - crop_x1)

    # 调整大小
    resized_img = resize(cropped_img, (fine_size, fine_size))

    return resized_img

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
        self.fineSize=self.opt.fineSize
        
        # self.resize_transform = transforms.Resize((320, 320))
        self.normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])
        self.plain_transform= transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        
        self.transforms = get_transforms(opt)
        # model = attempt_load(opt.attention_model_path, map_location='cpu')
        # model.eval()
        # target_layer = model.model[-2]  # Change this to the layer you want to visualize
        # self.attention_generator = CAM(model, target_layer)
        self.detector=Detector()
        

    # def random_crop(self,img):
    #     # Random crop
    #     i, j, h, w = transforms.RandomCrop.get_params(
    #         img, output_size=(self.opt.fineSize, self.opt.fineSize))
    #     img = F.crop(img, i, j, h, w)
    #     return img
    
    def crop_img(self,img, box, num_patches, fine_size, edge_size=0):
        """
        对图像 img 的指定区域 box 进行 num_patches 等分（加上边缘 edge_size），
        返回所有裁剪的部分并调整为指定大小。

        Args:
            img (torch.Tensor): 输入图像，形状为 (C, H, W)。
            box (tuple): 裁剪区域的边界 (x_min, y_min, x_max, y_max)。
            num_patches (int): 裁剪为 num_patches x num_patches 的小块数量。
            fine_size (tuple): 每个裁剪块调整到的大小 (height, width)。
            edge_size (int, optional): box 外围扩展的像素大小。默认为 0。

        Returns:
            torch.Tensor: 调整大小后的裁剪图像块列表，形状为 (num_patches^2, C, fine_size[0], fine_size[1])。
        """
        # 确保输入是张量
        assert isinstance(img, torch.Tensor), "Input img must be a torch.Tensor"

        # 提取裁剪区域，扩展边缘
        x_min, y_min, x_max, y_max = box
        x_min = max(0, x_min - edge_size)
        y_min = max(0, y_min - edge_size)
        x_max = min(img.shape[2], x_max + edge_size)
        y_max = min(img.shape[1], y_max + edge_size)
        # 适当扩展 box 区域，使其尽可能接近正方形
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width > box_height:
            diff = box_width - box_height
            y_min = max(0, y_min - diff // 2)
            y_max = min(img.shape[1], y_max + (diff - diff // 2))
        elif box_height > box_width:
            diff = box_height - box_width
            x_min = max(0, x_min - diff // 2)
            x_max = min(img.shape[2], x_max + (diff - diff // 2))
        # 裁剪后的区域大小
        cropped_region = img[:, y_min:y_max, x_min:x_max]
        cropped_height, cropped_width = cropped_region.shape[1], cropped_region.shape[2]

        resize_size = (fine_size * num_patches, fine_size * num_patches)
        # Resize成正方形
        cropped_region_resized = resize(
        cropped_region, resize_size, interpolation=transforms.InterpolationMode.NEAREST
        )
        
        cropped_height, cropped_width = cropped_region_resized.shape[1], cropped_region_resized.shape[2]

        # 计算每个小块的大小
        patch_height = cropped_height // num_patches
        patch_width = cropped_width // num_patches

        crops = []
        for i in range(num_patches):
            for j in range(num_patches):
                y_start = i * patch_height
                x_start = j * patch_width

                # 确保最后一个块覆盖完整区域
                y_end = y_start + patch_height if i < num_patches - 1 else cropped_height
                x_end = x_start + patch_width if j < num_patches - 1 else cropped_width

                # 提取小块并调整大小
                patch = cropped_region_resized[:, y_start:y_end, x_start:x_end]  # 3 h w
                # patch_resized = resize(patch, fine_size,interpolation=transforms.InterpolationMode.NEAREST)
                crops.append(patch)

        # 堆叠为张量输出
        return torch.stack(crops), [x_min,y_min,x_max,y_max] # 还包括裁剪的区域的位置

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_crops = []
        B_crops = []

        if self.opt.simulate_underwater:
            A_img = simulate_underwater_effect_pil(A_img)

        # Perform QR code detection
        A_boxes, _, _ = self.detector.infer_img(cv2.cvtColor(np.array(A_img), cv2.COLOR_RGB2BGR))
        B_boxes, _, _ = self.detector.infer_img(cv2.cvtColor(np.array(B_img), cv2.COLOR_RGB2BGR))

        if len(A_boxes) == 0 or len(B_boxes) == 0:
            return self.__getitem__(index + 1)  # Skip image if no QR code is detected

        A_boxes = A_boxes.astype(int)  # Ensure boxes are integers
        B_boxes = B_boxes.astype(int)  # Ensure boxes are integers

        A_box=A_boxes[0]
        B_box=B_boxes[0]

        A_img=self.plain_transform(A_img)  #应该调整图片为fineSize* sqrt(n_patch)大小的图片
        B_img=self.plain_transform(B_img)

        if self.opt.random_crop:
            
            for i in range(self.opt.n_patch):
                A_crops.append(crop_and_resize(A_img, A_box, self.opt.fineSize))
                B_crops.append(crop_and_resize(B_img, B_box, self.opt.fineSize))

        else:
            A_crops,A_croped_region=self.crop_img(A_img,A_box,self.opt.n_patch,self.opt.fineSize,self.opt.edge_size)
            B_crops,B_croped_region=self.crop_img(B_img,B_box,self.opt.n_patch,self.opt.fineSize,self.opt.edge_size)
        
        return {
            'A_crops': A_crops,   #只有一张图像
            'B_crops': B_crops,
            'A_img': A_img,  
            'B_img': B_img,  
            'A_paths': A_path,
            'B_paths': B_path,
            'A_qrcode_position': torch.tensor(A_box),
            'A_croped_region': torch.tensor(A_croped_region),
            'B_qrcode_position': torch.tensor(B_box),
            'B_croped_region': torch.tensor(B_croped_region),
        }


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'



