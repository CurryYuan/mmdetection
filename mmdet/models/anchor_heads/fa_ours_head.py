import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox,
                        multi_apply, multiclass_nms, force_fp32)
from mmdet.ops import DeformConv, MaskedConv2d
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.
    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 num_anchors,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            num_anchors * 4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module
class FAOursHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        self.deformable_groups = 4
        super(FAOursHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        self.feature_adaption = FeatureAdaption(
            self.num_anchors,
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

        self.feature_adaption.init_weights()

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_bbox_pred = self.rpn_reg(x)
        x = self.feature_adaption(x, rpn_bbox_pred)
        return x, rpn_bbox_pred

    def loss(self,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             proposals=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        if proposals:
            anchor_list = proposals

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_bbox, _ = multi_apply(
            self.loss_single,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos=num_total_pos,
            cfg=cfg)
        return dict(loss_rpn_bbox=losses_bbox)

    def loss_single(self, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_pos, cfg):
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_pos)
        return loss_bbox, None

    def get_refined_anchors_single(self, bbox_preds, mlvl_anchors, img_shape):
        mlvl_proposals = []
        for idx in range(len(bbox_preds)):
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            mlvl_proposals.append(proposals)
        return mlvl_proposals

    def get_refined_anchors(self, bbox_preds, img_metas, cfg,
                   refined_anchors, rescale=False):
        num_levels = len(bbox_preds)
        result_list = []

        if refined_anchors is None:
            mlvl_anchors = [
                self.anchor_generators[i].grid_anchors(bbox_preds[i].size()[-2:],
                                                       self.anchor_strides[i])
                for i in range(num_levels)
            ]

            for img_id in range(len(img_metas)):
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                proposals = self.get_refined_anchors_single(bbox_pred_list,
                                                   mlvl_anchors, img_shape)
                result_list.append(proposals)
        else:
            for img_id in range(len(img_metas)):
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                proposals = self.get_refined_anchors_single(bbox_pred_list,
                                                            refined_anchors[img_id], img_shape)
                result_list.append(proposals)
        return result_list