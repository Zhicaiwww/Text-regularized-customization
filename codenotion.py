# 使用一个封装的类或者一个类内的函数记录需要正则的权重索引
import torch.nn as nn
import torch
class TextRegLDM(nn.Module):
    def __init__(self, backbone:nn.Module, neck, head):
        super(TextRegLDM, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.to_L1norm_name = {}
        self.to_textReg_name = {}
        self.to_L1norm_weight = 1

    def get_L1norm_loss(self):
        for key in self.to_L1norm_name:
            torch.norm(getattr(self.backbone, key).weight.data, p = 1) 
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x