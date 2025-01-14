import torch.nn.functional as F
import cv2
import numpy as np
from models.experimental import attempt_load
from util.general import non_max_suppression, scale_coords
import torch


class CAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """注册前向传播的钩子函数，用于提取目标层的激活值。"""
        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_image, target_class):
        """
        生成目标类别的注意力图。
        Args:
            input_image (torch.Tensor): 输入图像张量。
            target_class (int): 目标类别索引。
        Returns:
            np.ndarray: 注意力图。
        """
        # 前向传播，获取输出
        output = self.model(input_image)[0]  # 获取模型输出 (batch_size, n_boxes, n_outputs)
        
        # 获取最后一层卷积层的权重
        last_conv_layer_weight = self.model.model[-1].m[-1].weight  # 假设 self.model.model[-1].m 是检测层
        num_anchors = self.model.model[-1].na  # 锚框数量
        per_anchor_outputs = self.model.model[-1].no  # 每个锚框的输出大小 (nc + 5)
        target_class_weight_index = target_class + 5  # 类别索引从第6位开始

        # 获取目标类别的权重
        target_class_weights = last_conv_layer_weight.view(
            num_anchors, per_anchor_outputs, -1, 
            last_conv_layer_weight.shape[-2], last_conv_layer_weight.shape[-1]
        )[:, target_class_weight_index, :, :, :]  # (na, in_channels, kernel_h, kernel_w)

        # 激活值 (batch_size, channels, height, width)
        activations = self.activations  # 通过 hook 提取到的激活值
        cam = torch.zeros_like(activations[0, 0])  # 初始化空的 CAM 图

        # 遍历每个通道，计算注意力图
        for i in range(activations.shape[1]):
            cam += (activations[0, i] * target_class_weights[:, i, :, :].sum()).cpu().data.numpy()

        # ReLU 去除负值
        cam = np.maximum(cam, 0)

        return cam  # 10 10