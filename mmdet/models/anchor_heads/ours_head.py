import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import AnchorGenerator, anchor_target, multi_apply
from .anchor_head import AnchorHead
from ..losses import smooth_l1_loss
from ..registry import HEADS
from ..losses import refinedet_multibox_loss

from math import sqrt as sqrt
from itertools import product as product


# TODO: add loss evaluator for SSD
@HEADS.register_module
class RefineDetHead(AnchorHead):
    mbox = {
        '320': [3, 3, 3, 3],  # number of boxes per feature map location
        '512': [3, 3, 3, 3],  # number of boxes per feature map location
    }

    def __init__(self,
                 input_size=320,
                 num_classes=81,
                 in_channels=(512, 512, 1024, 512),
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_base_sizes=None,
                 anchor_strides=(8, 16, 32, 64),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        self.anchor_strides = anchor_strides

        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        anchor_scales = 1

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(anchor_ratios) * anchor_scales

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

        self._init_layers()

    def _init_layers(self):
        # ARM
        reg_convs = []
        cls_convs = []
        for i in range(len(self.in_channels)):
            reg_convs.append(
                nn.Conv2d(self.in_channels[i], 1, kernel_size=1, padding=1))
            cls_convs.append(
                nn.Conv2d(self.in_channels[i], self.num_anchors * 2, kernel_size=1, padding=1))
        self.arm_reg = nn.ModuleList(reg_convs)
        self.arm_cls = nn.ModuleList(cls_convs)

        # TCB
        TCB = self.add_tcb(self.in_channels)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        # ODM
        reg_convs = []
        cls_convs = []
        for i in range(len(self.in_channels)):
            reg_convs.append(
                nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, padding=1))
            cls_convs.append(
                nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, padding=1))
        self.odm_reg = nn.ModuleList(reg_convs)
        self.odm_cls = nn.ModuleList(cls_convs)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):

        tcb_feats = list()
        arm_cls = list()
        arm_reg = list()
        odm_cls = list()
        odm_reg = list()

        # apply ARM to feats
        for feat, reg_conv, cls_conv in zip(feats, self.arm_reg, self.arm_cls):
            arm_cls.append(cls_conv(feat))
            arm_reg.append(reg_conv(feat))

        # calculate TCB features
        p = None
        for k, v in enumerate(feats[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3 - k) * 3 + i](s)
                # print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3 - k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3 - k) * 3 + i](s)
            p = s
            tcb_feats.append(s)

        tcb_feats.reverse()

        # apply ODM to feats
        for feat, reg_conv, cls_conv in zip(tcb_feats, self.odm_reg, self.odm_cls):
            odm_cls.append(cls_conv(feat))
            odm_reg.append(reg_conv(feat))

        return arm_cls, arm_reg, odm_cls, odm_reg

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def get_anchors(self, featmap_sizes, img_metas):

        steps = [8, 16, 32, 64]
        min_sizes = [32, 64, 128, 256]
        max_sizes = []
        aspect_ratios = [[2], [2], [2], [2]]
        clip = True

        mean = []
        for k, f in enumerate(featmap_sizes):
            f = f[0]
            for i, j in product(range(f), repeat=2):
                f_k = self.input_size / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = min_sizes[k] / self.input_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if max_sizes:
                    s_k_prime = sqrt(s_k * (max_sizes[k] / self.input_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if clip:
            output.clamp_(max=1, min=0)
        return output.cuda()

    def loss(self, arm_cls, arm_reg, odm_cls, odm_reg, gt_bboxes,
             gt_labels, img_metas, cfg, gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in arm_cls]
        assert len(featmap_sizes) == len(self.anchor_generators)

        assert len(arm_cls) == len(arm_reg)

        anchor_list = self.get_anchors(
            featmap_sizes, img_metas)

        # process predict
        arm_criterion = refinedet_multibox_loss(self.input_size, 2, 0.5, True, 0, True, 3, 0.5, False, self.target_stds)
        odm_criterion = refinedet_multibox_loss(self.input_size, self.num_classes, 0.5, True, 0, True, 3, 0.5,
                                                False, self.target_stds, use_ARM=True)

        arm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_cls], 1)
        arm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_reg], 1)

        odm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_cls], 1)
        odm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_reg], 1)

        predict = (
            arm_reg.view(arm_reg.size(0), -1, 4),
            arm_cls.view(arm_cls.size(0), -1, 2),
            odm_reg.view(odm_reg.size(0), -1, 4),
            odm_cls.view(odm_cls.size(0), -1, self.num_classes),
            anchor_list
        )

        targets = (gt_bboxes, gt_labels)

        arm_reg_loss, arm_cls_loss = arm_criterion(predict, targets)
        odm_reg_loss, odm_cls_loss = odm_criterion(predict, targets)

        return dict(arm_reg_loss=arm_reg_loss,
                    arm_cls_loss=arm_cls_loss,
                    odm_reg_loss=odm_reg_loss,
                    odm_cls_loss=odm_cls_loss)

    def add_tcb(self, in_channels):
        feature_scale_layers = []
        feature_upsample_layers = []
        feature_pred_layers = []
        for k, v in enumerate(in_channels):
            feature_scale_layers += [nn.Conv2d(in_channels[k], 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
                                     ]
            feature_pred_layers += [nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(inplace=True)
                                    ]
            if k != len(in_channels) - 1:
                feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
        return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


