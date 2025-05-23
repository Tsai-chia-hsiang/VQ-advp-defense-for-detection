from . import _CFG_DIR_
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.nn.tasks import DetectionModel

__all__ = [
    "DEFAULT_PB_FILE",
    "V8Detection_MaxProb_Loss",
    "Supervised_V8Detection_MaxProb_Loss",
    "TotalVariation",
    "NPSCalculator"
]

DEFAULT_PB_FILE =_CFG_DIR_/"30values.txt"

class Supervised_V8Detection_MaxProb_Loss(v8DetectionLoss):
    
    def __init__(self, model:DetectionModel, to_attack:torch.Tensor, conf:float=0.25, tal_topk=10):

        assert isinstance(model, DetectionModel)
        super().__init__(model, tal_topk)

        if isinstance(self.hyp, dict):
            self.hyp = IterableSimpleNamespace(**self.hyp)
        
        self.hyp.box = 7.5 # (float) box loss gain
        self.hyp.cls = 0.5 # (float) cls loss gain (scale with pixels)
        self.hyp.dfl = 1.5
        self.loss_w = torch.tensor([self.hyp.cls, self.hyp.box, self.hyp.dfl])
        self.to_attack = to_attack.detach().clone()
        self.conf = conf

    def __call__(self, preds, batch):

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2
        confident = pred_scores.detach().sigmoid()
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            confident,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        if self.to_attack.device != target_scores.device:
            self.to_attack = self.to_attack.to(target_scores.device)
        if self.loss_w.device != target_scores.device:
            self.loss_w = self.loss_w.to(target_scores.device)
         
        conf, cls_idx = confident.max(dim=-1)
        att_mask = (conf >= self.conf) & torch.isin(cls_idx, self.to_attack)
        fg_mask = fg_mask & att_mask

        loss = torch.tensor([0.0,0.0,0.0], device=pred_scores.device)
        
        if att_mask.sum():
            l,_ = torch.max(pred_scores.sigmoid()[att_mask], dim=-1)
            loss[0] = -l.mean()
        
        if fg_mask.sum():
            
            target_bboxes /= stride_tensor
            loss[1], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, 
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        
        loss = loss * self.loss_w

        return -loss
      
class V8Detection_MaxProb_Loss(nn.Module):

    def __init__(self, model:DetectionModel, to_attack:torch.Tensor, conf:float=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m =  model.model[-1]
        self.reg_max = m.reg_max
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.to_attack = to_attack.clone().detach()
        self.conf = conf
    
    def forward(self, preds, **kwargs) -> dict[str, torch.Tensor]:
        
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        if self.to_attack.device != feats[0].device:
            self.to_attack = self.to_attack.to(feats[0].device)
        
        
        _, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max*4, self.nc), 1
        )
       
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
        conf, cls_idx = pred_scores.max(dim=-1)
        attack_mask = (conf >= self.conf) & torch.isin(cls_idx, self.to_attack)
        
        if attack_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_scores.device)
        
        l,_ = torch.max(conf[attack_mask], dim=-1)
        
        return l.mean()
    


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self, w:float=2.5):
        super(TotalVariation, self).__init__()
        self.w = w

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)*self.w


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """
    def __init__(self, patch_side, printability_file:Path=DEFAULT_PB_FILE, w:float=0.01):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)
        self.w = w

    def forward(self, adv_patch:torch.Tensor):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array.to(adv_patch.device)+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)*self.w

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa