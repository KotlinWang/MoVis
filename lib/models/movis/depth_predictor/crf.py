import torch
import torch.nn as nn
from torchvision.transforms import Resize


class CRFRefinement(nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(CRFRefinement, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_hidden + 3, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Sigmoid()

    def forward(self, src, image):
        img = Resize([(384 // 16), (1280 // 16)])(image)

        # 将深度概率和图像进行拼接
        # color_img = self.relu(self.color_trans(image))
        x = torch.cat((src, img), dim=1)

        # 使用卷积进行特征提取和空间平滑
        x = self.conv1(x)

        return self.softmax(x)
