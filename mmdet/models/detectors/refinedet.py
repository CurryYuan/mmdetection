from .single_stage import SingleStageDetector
from ..registry import DETECTORS
import torch
import numpy as np
from mmdet.core import bbox2result


@DETECTORS.register_module
class RefineDet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RefineDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        # bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]

        detections = self.bbox_head.get_bboxes(*bbox_inputs).data
        if rescale:
            h = img_meta[0]['ori_shape'][0]
            w = img_meta[0]['ori_shape'][1]
        else:
            h = img_meta[0]['img_shape'][0]
            w = img_meta[0]['img_shape'][1]
        bbox_results = []

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                bbox_results.append(np.zeros((0, 5), dtype=np.float32))
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            bbox_results.append(np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False))

        return bbox_results