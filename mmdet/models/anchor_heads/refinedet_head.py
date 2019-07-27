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

from mmdet.models.utils.refinedet_utils import decode, nms, center_size


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

        self.num_anchors = len(anchor_ratios) * anchor_scales

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

        self.conf_thresh = 0.01
        self.objectness_thre = 0.01
        self.top_k = 1000
        self.nms_thresh = 0.45
        self.keep_top_k = 500

        self._init_layers()

    def _init_layers(self):
        # ARM
        reg_convs = []
        cls_convs = []
        for i in range(len(self.in_channels)):
            reg_convs.append(
                nn.Conv2d(self.in_channels[i], self.num_anchors * 4, kernel_size=3, padding=1))
            cls_convs.append(
                nn.Conv2d(self.in_channels[i], self.num_anchors * 2, kernel_size=3, padding=1))
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

        self.softmax = nn.Softmax(dim=-1)

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

        assert len(arm_cls) == len(arm_reg)

        prior_data = self.get_anchors(featmap_sizes, img_metas)

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
            prior_data
        )

        targets = (gt_bboxes, gt_labels)

        arm_reg_loss, arm_cls_loss = arm_criterion(predict, targets)
        odm_reg_loss, odm_cls_loss = odm_criterion(predict, targets)

        return dict(arm_reg_loss=arm_reg_loss,
                    arm_cls_loss=arm_cls_loss,
                    odm_reg_loss=odm_reg_loss,
                    odm_cls_loss=odm_cls_loss)

    def get_bboxes(self, arm_cls, arm_reg, odm_cls, odm_reg, img_metas, cfg,
                   rescale=False):
        """
                Args:
                    loc_data: (tensor) Loc preds from loc layers
                        Shape: [batch,num_priors*4]
                    conf_data: (tensor) Shape: Conf preds from conf layers
                        Shape: [batch*num_priors,num_classes]
                    prior_data: (tensor) Prior boxes and variances from priorbox layers
                        Shape: [1,num_priors,4]
                """

        featmap_sizes = [featmap.size()[-2:] for featmap in arm_cls]
        prior_data = self.get_anchors(featmap_sizes, img_metas)
        num_priors = prior_data.size(0)

        arm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_cls], 1)
        arm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_reg], 1)

        odm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_cls], 1)
        odm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_reg], 1)

        arm_reg = arm_reg.view(arm_reg.size(0), -1, 4)  # arm loc preds
        arm_cls = self.softmax(arm_cls.view(arm_cls.size(0), -1, 2))  # arm conf preds
        odm_reg = odm_reg.view(odm_reg.size(0), -1, 4)  # odm loc preds
        odm_cls = self.softmax(odm_cls.view(odm_cls.size(0), -1, self.num_classes))

        loc_data = odm_reg
        conf_data = odm_cls

        arm_object_conf = arm_cls.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        det_bboxes = []
        det_labels = []
        result_list = []

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_reg[i], prior_data, self.target_stds)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.target_stds)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            # print(decoded_boxes, conf_scores)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # print(scores.dim())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # print(boxes, scores)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # for j in range(count):
                    # det_bboxes.append(torch.cat((scores[ids[j]].unsqueeze(0), boxes[ids[j]]), 0))
                    # det_labels.append(cl-1)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

        # det_bboxes = torch.cat([o.unsqueeze(0) for o in det_bboxes], 0)
        # det_labels = torch.tensor(det_labels)
        #
        # w = img_metas[0]['img_shape'][0]
        # h = img_metas[0]['img_shape'][1]
        #
        # det_bboxes[:, 0] *= w
        # det_bboxes[:, 2] *= w
        # det_bboxes[:, 1] *= h
        # det_bboxes[:, 3] *= h
        #
        # result_list.append((det_bboxes, det_labels))
        # return result_list

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


