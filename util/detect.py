

import cv2
import re
import numpy as np

import onnxruntime as ort
import torch.nn.functional as F
import random

# from utils import plot_one_box,_make_grid,cal_outputs,post_process_opencv,infer_img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def _make_grid( nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs



def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)

    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]
def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, [model_w,model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)

    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)

    return  boxes,confs,ids


def infer_img_tensor(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    img = F.interpolate(img0.unsqueeze(0), size=(model_h, model_w), mode='bilinear', align_corners=False).squeeze(0)
    
    blob=img.unsqueeze(0).numpy()  # 1 3 320 320
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)

    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)

    return  boxes,confs,ids

class Detector:
    def __init__(self,model_pb_path= "checkpoints/qrcode_detect/best.onnx") -> None:
        self.model_pb_path = model_pb_path
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(model_pb_path, so)
        
        # 标签字典
        self.dic_labels= {0:'QRCode'}
        
        # 模型参数
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride=[8.,16.,32.]
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        pass




    def identify(self,img,thred_cond=0.5):
        det_boxes,scores,ids = infer_img(img,self.net,self.model_h,self.model_w,self.nl,self.na,self.stride,self.anchor_grid,thred_nms=0.4,thred_cond=thred_cond)
        return ids
    
    def get_boxs(self,img):
        det_boxes,scores,ids = infer_img(img,self.net,self.model_h,self.model_w,self.nl,self.na,self.stride,self.anchor_grid,thred_nms=0.4,thred_cond=0.5)
        return det_boxes
    
    def infer_img(self,img,thred_cond=0.5):
        det_boxes,scores,ids = infer_img(img,self.net,self.model_h,self.model_w,self.nl,self.na,self.stride,self.anchor_grid,thred_nms=0.4,thred_cond=thred_cond)
        return det_boxes,scores,ids
    
    def infer_img_tensor(self,img,thred_cond=0.5):
        det_boxes,scores,ids = infer_img_tensor(img,self.net,self.model_h,self.model_w,self.nl,self.na,self.stride,self.anchor_grid,thred_nms=0.4,thred_cond=thred_cond)
        return det_boxes,scores,ids
