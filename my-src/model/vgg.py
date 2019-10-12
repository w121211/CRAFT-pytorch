import torch
import torch.nn as nn
from torchvision import models


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True).features
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        N = x.shape[0]
        x = self.vgg(x)
        x = self.pool(x)
        return x.view(N, -1)
