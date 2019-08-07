import mmcv

from mmdet.core import tensor2imgs, bbox_mapping
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
import copy
import torch


@DETECTORS.register_module
class MyFaRPN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(MyFaRPN, self).__init__()
        self.num_stages = num_stages

        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.rpn_head = builder.build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(MyFaRPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights()
        for i in range(self.num_stages):
            self.rpn_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)

        losses = dict()
        proposal_list = None

        for i in range(self.num_stages):
            # print("stage: ", i)
            lw = self.train_cfg.stage_loss_weights[i]
            x, rpn_cls_scores, rpn_bbox_preds, rpn_giou_preds = self.rpn_head[i](x)
            rpn_outs = (rpn_cls_scores, rpn_bbox_preds, rpn_giou_preds)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn[i], copy.deepcopy(proposal_list))
            rpn_losses = self.rpn_head[i].loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            for name, value in rpn_losses.items():
                for j in range(len(value)):
                    value[j] = value[j] * lw
                losses['s{}.{}'.format(i, name)] = value
            # losses.update(rpn_losses)

            with torch.no_grad():
                rpn_refined_inputs = (rpn_cls_scores, rpn_bbox_preds) + (img_meta, self.train_cfg.rpn[i], proposal_list)
                proposal_list = self.rpn_head[i].get_refined_anchors(*rpn_refined_inputs)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)

        proposal_list = None

        for i in range(self.num_stages):
            x, rpn_cls_scores, rpn_bbox_preds, rpn_giou_preds = self.rpn_head[i](x)
            rpn_outs = (rpn_cls_scores, rpn_bbox_preds, rpn_giou_preds)

            if i == self.num_stages - 1:
                proposal_cfg = self.test_cfg.get('rpn_proposal', self.test_cfg.rpn)
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg, proposal_list)
                proposal_list = self.rpn_head[i].get_bboxes(*proposal_inputs)
            else:
                rpn_refined_inputs = (rpn_cls_scores, rpn_bbox_preds) + (img_meta, self.test_cfg.rpn, proposal_list)
                proposal_list = self.rpn_head[i].get_refined_anchors(*rpn_refined_inputs)

        if rescale:
            for proposals, meta in zip(proposal_list, img_meta):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape,
                                                scale_factor, flip)
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()

    def show_result(self, data, result, img_norm_cfg, dataset=None, top_k=20):
        """Show RPN proposals on the image.

        Although we assume batch size is 1, this method supports arbitrary
        batch size.
        """
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            mmcv.imshow_det_bboxes(img_show, result, result[:, 4], score_thr=0.1)
            # mmcv.imshow_bboxes(img_show, result, top_k=top_k)
