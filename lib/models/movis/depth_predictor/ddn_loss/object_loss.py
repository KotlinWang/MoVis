import torch
import torch.nn as nn
import math

from .balancer import Balancer
from .focalloss import FocalLoss


# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class OBJLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        # self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        # self.bce_loss = nn.BCELoss()
        self.mse = nn.MSELoss()

    def build_target(self, object_probs, gt_boxes2d, gt_center_depth, num_gt_per_img):
        B, _, H, W = object_probs.shape
        depth_maps = torch.zeros((B, H, W), device=object_probs.device, dtype=object_probs.dtype) - 1.

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                # depth_maps[b, v1:v2, u1:u2] = n+1
                depth_maps[b, v1:v2, u1:u2] = (gt_boxes_per_batch.shape[0] - n) / gt_boxes_per_batch.shape[0]

        return depth_maps

    def forward(self, object_probs, gt_boxes2d, num_gt_per_img, gt_center_depth):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # Bin depth map to create target
        object_target = self.build_target(object_probs, gt_boxes2d, gt_center_depth, num_gt_per_img)
        # Compute loss
        loss = self.mse(object_probs, object_target.unsqueeze(1))
        # ipdb.set_trace()
        # Compute foreground/background balancing
        # loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return loss
