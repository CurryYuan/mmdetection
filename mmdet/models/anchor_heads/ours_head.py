import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, giou_anchor_target, delta2bbox,
                        multi_apply, multiclass_nms, force_fp32, bbox_overlaps)
from mmdet.ops import DeformConv, MaskedConv2d
from mmdet.ops import nms, soft_nms
from .anchor_head import AnchorHead
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class OursHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        self.deformable_groups = 4
        super(OursHead, self).__init__(2, in_channels, **kwargs)
        self.loss_giou = build_loss(dict(
            type='SmoothL1Loss', loss_weight=1.0))

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
             self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        self.rpn_giou = nn.Conv2d(self.feat_channels, self.num_anchors, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

        normal_init(self.rpn_giou, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_giou_pred = self.rpn_giou(x)
        return x, rpn_cls_score, rpn_bbox_pred, rpn_giou_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             giou_preds,
             gt_bboxes,
             img_metas,
             cfg,
             proposals=None,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        gt_labels = None
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        if proposals:
            anchor_list = proposals

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = giou_anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg,
         mlvl_anchors) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox, losses_giou = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            giou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mlvl_anchors,
            img_metas,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_giou=losses_giou)

    def get_giou(self, pred, target, pos_inds, anchors, img_meta):
        assert pred.size() == target.size() and target.numel() > 0

        pred = delta2bbox(anchors[pos_inds], pred[pos_inds], self.target_means,
                                    self.target_stds, img_meta['img_shape'])
        target = delta2bbox(anchors[pos_inds], target[pos_inds], self.target_means,
                                      self.target_stds, img_meta['img_shape'])
        
        # iou
        # ious = bbox_overlaps(pred_proposals, target_proposals, is_aligned=True).clamp(min=1e-6)

        # giou
        area1 = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        area2 = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
        
        overlap_lt = torch.max(pred[:, :2], target[:, :2])  # [n, 2]
        overlap_rb = torch.min(pred[:, 2:], target[:, 2:])
        overlap_wh = (overlap_rb - overlap_lt + 1).clamp(min=0)
        overlap = overlap_wh[:, 0] * overlap_wh[:, 1]
        
        unions = area1 + area2 - overlap
        ious = overlap / unions
        
        convex_lt = torch.min(pred[:, :2], target[:, :2])
        convex_rb = torch.max(pred[:, 2:], target[:, 2:])
        convex_wh = (convex_rb - convex_lt + 1).clamp(min=0)
        convex = convex_wh[:, 0] * convex_wh[:, 1]
        
        gious = ious - (convex - unions) / convex.clamp(min=1e-5)  # [n]

        return gious  # (1 - gious).sum()

    def loss_single(self, cls_score, bbox_pred, giou_pred, labels, label_weights,
                    bbox_targets, bbox_weights, anchors, img_meta,
                    num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)

        # giou loss
        anchors = anchors.reshape(-1, 4)
        giou_weights = bbox_weights[:, 0].reshape(-1, 1)
        pos_inds = torch.nonzero(giou_weights)[:, 0]

        if len(pos_inds) == 0:
            loss_giou = torch.tensor(0, dtype=torch.float).cuda()
            loss_bbox = torch.tensor(0, dtype=torch.float).cuda()
        else:
            giou_preds = giou_pred.permute(0, 2, 3, 1).reshape(-1, 1)[pos_inds]
            giou_targets = self.get_giou(bbox_pred, bbox_targets, pos_inds,
                                         anchors, img_meta).reshape(-1, 1)
            loss_giou = self.loss_giou(giou_preds, giou_targets,
                                       avg_factor=len(pos_inds))
            loss_bbox = self.loss_bbox(
                bbox_pred[pos_inds],
                bbox_targets[pos_inds],
                giou_weights[pos_inds],
                avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_giou

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

    def get_refined_anchors_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            # rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            # rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            # if self.use_sigmoid_cls:
            #     rpn_cls_score = rpn_cls_score.reshape(-1)
            #     scores = rpn_cls_score.sigmoid()
            # else:
            #     rpn_cls_score = rpn_cls_score.reshape(-1, 2)
            #     scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
            #     _, topk_inds = scores.topk(cfg.nms_pre)
            #     rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
            #     anchors = anchors[topk_inds, :]
            #     scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            # if cfg.min_bbox_size > 0:
            #     w = proposals[:, 2] - proposals[:, 0] + 1
            #     h = proposals[:, 3] - proposals[:, 1] + 1
            #     valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
            #                                (h >= cfg.min_bbox_size)).squeeze()
            #     proposals = proposals[valid_inds, :]
            #     scores = scores[valid_inds]
            # proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            # proposals, _ = nms(proposals, cfg.nms_thr)
            # proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        # proposals = torch.cat(mlvl_proposals, 0)
        # if cfg.nms_across_levels:
        #     proposals, _ = nms(proposals, cfg.nms_thr)
        #     proposals = proposals[:cfg.max_num, :]
        # else:
        #     scores = proposals[:, 4]
        #     num = min(cfg.max_num, proposals.shape[0])
        #     _, topk_inds = scores.topk(num)
        #     proposals = proposals[topk_inds, :]
        return mlvl_proposals

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'giou_preds'))
    def get_bboxes(self, cls_scores, bbox_preds,
                   giou_preds,
                   img_metas, cfg,
                   refined_anchors, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        result_list = []

        if refined_anchors is None:
            mlvl_anchors = [
                self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                       self.anchor_strides[i])
                for i in range(num_levels)
            ]

            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                giou_pred_list = [
                    giou_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self.get_bboxes_single(giou_pred_list, bbox_pred_list,
                                                            mlvl_anchors, img_shape,
                                                            scale_factor, cfg, rescale)
                result_list.append(proposals)
        else:
            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                            refined_anchors[img_id], img_shape,
                                                            scale_factor, cfg, rescale)
                result_list.append(proposals)
        return result_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_refined_anchors(self, cls_scores, bbox_preds, img_metas, cfg,
                   refined_anchors, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        result_list = []

        if refined_anchors is None:
            mlvl_anchors = [
                self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                       self.anchor_strides[i])
                for i in range(num_levels)
            ]

            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self.get_refined_anchors_single(cls_score_list, bbox_pred_list,
                                                   mlvl_anchors, img_shape,
                                                   scale_factor, cfg, rescale)
                result_list.append(proposals)
        else:
            for img_id in range(len(img_metas)):
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor']
                proposals = self.get_refined_anchors_single(cls_score_list, bbox_pred_list,
                                                            refined_anchors[img_id], img_shape,
                                                            scale_factor, cfg, rescale)
                result_list.append(proposals)
        return result_list
