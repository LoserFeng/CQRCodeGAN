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
        model = attempt_load(opt.attention_model_path, map_location='cpu')
        model.eval()
        target_layer = model.model[-2]  # Change this to the layer you want to visualize
        self.attention_generator = CAM(model, target_layer)
        
        # Load the ONNX model
        self.model_pb_path = "checkpoints/attention_model/best.onnx"
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(self.model_pb_path, so)
        
        # Define labels
        self.dic_labels = {0: 'QRCode'}
        
        # Model parameters
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        anchors = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def cal_outputs(self, outs):
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_h / self.stride[i]), int(self.model_w / self.stride[i])
            length = int(self.na * h * w)
            if grid[i].shape[0] != h * w:
                grid[i] = self._make_grid(w, h)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs

    def post_process_opencv(self, outputs, img_h, img_w, thred_nms=0.4, thred_cond=0.5):
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / self.model_w * img_w
        c_y = outputs[:, 1] / self.model_h * img_h
        w = outputs[:, 2] / self.model_w * img_w
        h = outputs[:, 3] / self.model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)

        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
        
        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
        if len(ids) > 0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        else:
            return [], [], []

    def infer_img(self, img0:torch.Tensor):
        # Preprocess image
        # img = cv2.resize(img0, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32) / 255.0
        # blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        img = F.interpolate(img0.unsqueeze(0), size=(self.model_h, self.model_w), mode='bilinear', align_corners=False).squeeze(0)
    
        blob=img.unsqueeze(0).numpy()  # 1 3 320 320
        # Model inference
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        # Adjust outputs
        outs = self.cal_outputs(outs)

        # Calculate detection boxes
        c,img_h, img_w = img0.shape
        boxes, confs, ids = self.post_process_opencv(outs, img_h, img_w)

        return boxes, confs, ids

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx) 
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
                
        #  获取QRCode Attention
        attention_input_img=preprocess_image(A_img,320)
        attention_img=self.attention_generator.generate(attention_input_img,0)
        attention_img = F.interpolate(attention_img.unsqueeze(0).unsqueeze(0), size=(A_img.shape[-2], A_img.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)
        attention_img = (attention_img - torch.min(attention_img))/(torch.max(attention_img) - torch.min(attention_img))

        # Perform QR code detection
        boxes, confs, ids = self.infer_img(A_img)
        if len(boxes)>0:  # 将 boxes 转换为整数类型
            boxes = boxes.astype(int) 
            boxes = np.clip(boxes, 0, 320) 
        
        # Prepare qrcode_position
        qrcode_position = torch.tensor(boxes[0]) if len(boxes) > 0 else torch.tensor([-1,-1,-1,-1])
        
        return {
            'A': A_img,
            'B': B_img,
            'A_gray': attention_img,
            'input_img': input_img,
            'A_paths': A_path,
            'B_paths': B_path,
            'qrcode_position': qrcode_position
        }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'



