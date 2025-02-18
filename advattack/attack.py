import sys
from pathlib import Path
import os.path as osp
import gc
import math
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .augment import MedianPool2d

sys.path.append(osp.abspath(Path(__file__).parent.parent/"torchcvext"))
from torchcvext.box import scale_box, xywh2xyxy, draw_boxes
from torchcvext.convert import tensor2img

DEFAULT_ATTACKER_ARGS = {
    'contrast': (0.8,1.2),
    'brightness':(-0.1, 0.1),
    'angle':(-20, 20),
    'noise':(-1, 1),
    'noise_factor':0.1,
    'patch_box_scale':0.2
}

class PatchAttacker():

    def __init__(self, contrast = (0.8,1.2), brightness=(-0.1, 0.1), angle=(-20, 20), noise=(-1, 1), noise_factor=0.1, patch_box_scale=0.2):
        
        self.medianpooler = MedianPool2d(7,same=True)
        self.contrast = contrast
        self.brightness = brightness
        self.noise = noise
        self.noise_factor = noise_factor
        self.patch_box_scale = patch_box_scale
        self.y_shift = -0.05
        self.rotate_angle = (angle[0]/ 180 * math.pi, angle[1]/ 180 * math.pi)

    def box_geo_scale(self, xywh:torch.Tensor) -> torch.Tensor:
        return torch.sqrt((xywh[:, 2]*self.patch_box_scale)**2 + (xywh[:, 3]*self.patch_box_scale)**2)
    
    @staticmethod
    def uniform_mask(uniform_bound:tuple[float, float], psize:tuple, n_instantces:int, device:torch.device)-> torch.Tensor:
        mask = torch.empty(
            n_instantces, dtype=torch.float32, device=device
        ).uniform_(uniform_bound[0], uniform_bound[1]).view((n_instantces,1,1,1))  
        mask = mask.expand(n_instantces, *psize)
        return mask
        
    def patch_affine_transformation(self, patch:torch.Tensor, xywh:torch.Tensor, angle:torch.Tensor, scale:torch.Tensor) -> torch.Tensor:
        """
        [R|t]^(-1) = [R^(-1)|-t] 
        = [[
            cos -sin \\
            sin cos
        ]]^-1 | [-t] = [[
            cos sin \\
            -sin cos
        ]] | [-t]
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        tx = (-xywh[:, 0].to(device=patch.device)+0.5)*2
        ty = (-(xywh[:, 1].to(device=patch.device)+self.y_shift)+0.5)*2
        theta = torch.zeros(xywh.size(0), 2, 3, dtype=torch.float32, device=patch.device)
        
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = (tx*cos + ty*sin)/scale
        
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = (-tx*sin + ty*cos)/scale
        
        grid = F.affine_grid(theta, patch.size(), align_corners=False)
        
        return F.grid_sample(patch, grid, align_corners=False)
 
    def __call__(self, img:torch.Tensor, patch:torch.Tensor, bboxes:torch.Tensor, batch_idx:torch.Tensor, random_rotate_patch:bool=False):
        
        dev = img.device
        origin_box = scale_box(boxes=bboxes, imgsize=img.size()[::-1][:2], direction='back')
        patch = self.medianpooler(patch.unsqueeze(0))
        psize = patch.size()[1:] # CxHxW
        pad = (
            (img.size(-1) - patch.size(-1)) / 2, 
            (img.size(-2) - patch.size(-2)) / 2 
        )
        # W pad, H pad
        n_boxes = bboxes.size(0)
        
        contrast = PatchAttacker.uniform_mask(uniform_bound=self.contrast, psize=psize, n_instantces=n_boxes, device=dev)
        brightness = PatchAttacker.uniform_mask(uniform_bound=self.brightness, psize=psize, n_instantces=n_boxes, device=dev)
        noise = torch.empty(brightness.size(),dtype=torch.float32, device=dev).uniform_(self.noise[0], self.noise[1]) * self.noise_factor
        patch = patch.expand(n_boxes, *psize)
        patch = patch*contrast + brightness + noise
        patch = torch.clamp(patch, 0.000001, 0.99999)

        patch = F.pad(patch, (int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), mode='constant', value=0)
        angle =None
        if random_rotate_patch:
            angle = torch.empty(n_boxes, dtype=torch.float32, device=dev).uniform_(self.rotate_angle[0], self.rotate_angle[1])
        else:
            angle = torch.zeros(n_boxes, dtype=torch.float32, device=dev)
        
        geo_scale = (self.box_geo_scale(origin_box)/min(psize[1:])).to(device=dev)
        
        patch = self.patch_affine_transformation(patch=patch, xywh=bboxes, angle=angle, scale=geo_scale)

        # for each img, attacking it independently
        for i in range(img.size(0)):
            bidx = torch.where(batch_idx == i)[0]
            using_patch = patch[bidx]
            # each slice of the patch is for a bbox
            for obj_adv_patch in using_patch:
                img[i] = torch.where((obj_adv_patch == 0), img[i], obj_adv_patch)
        
        return img



def ultralytics_yolobatch_draw_boxes(batch, save_to:Path=None, return_img_lst:bool=False, need_scale=False) -> None|list[np.ndarray]:
    """
    special design for Ultraytics YOLODataset batch
    - mainly for debug purpose

    Arg:
    -- 
    - batch: (dict): A default batch ultrayltics YOLODataset dataloader
    - save_to: (Path, default `None`): Root for save those images for `batch`
    - return_img_lst: (bool, default `False`): Whether to return the batch of images with drawn bounding boxes. 
    
    Return:
    --
        `None` if `return_img_lst` is `False`, otherwise a list of cv2 image with drawn bounding boxes
    --
    """
    
    ret = []
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
    
    for i, timg in enumerate(batch['img']):
        idx_mask:torch.Tensor = torch.where(batch['batch_idx'] == i)[0]
        its_boxes:torch.Tensor = batch['bboxes'][idx_mask]
        img = tensor2img(timg, scale_back_f=lambda x:x*255 if need_scale else lambda x:x)
        xyxy = xywh2xyxy(xywh=its_boxes.cpu().detach().numpy())
        draw_boxes(img=img, xyxy=scale_box(xyxy, imgsize=np.asarray(img.shape[:2][::-1]), direction='back'))
        if save_to is not None:
            cv2.imwrite(save_to/Path(batch['im_file'][i]).name, img=img)
        if return_img_lst:
            ret.append(img)
    
    if return_img_lst:
        return ret
    else:
        gc.collect()


def preprocess_yolo_batch_with_attack(
    batch:dict, patch:torch.Tensor, attacker:PatchAttacker, device:torch.device,
    random_rotate_patch:bool=False, plot:bool=False
)->dict[str, ]:
    """
    Apply the transformation in place at the batch level, 
    similar to the preprocess() function in Ultralytics' Trainer and Validator. 
    Then, paste the patch onto the batch images.

    Args
    --
    - batch: dict
        - ultrayltics YOLODataset batch 
    - patch: torch Tensor
        - the patch for adversarial attacking 
    - attaker: PatchAttacker
        - a PatchAttacker
    - random_rotation_patch: bool, default False
        - Wether to apply random ratation
            - For attacker's argument
    - plot: bool, defualt False
        - Wether plotting the results of images under attack..
            - It will auto save (using cv2) the images to ${workdir}/view/
            - For debug

    Return
    --
    No return value (None)
    """
    batch["img"] = batch["img"].to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255
    if patch.device != device:
        patch.data = patch.data.to(device=device)

    batch["img"] = attacker(
        patch = patch, img=batch["img"],  
        bboxes=batch['bboxes'], 
        batch_idx=batch['batch_idx'], 
        random_rotate_patch=random_rotate_patch
    )
    for k in ["batch_idx", "cls", "bboxes"]:
        batch[k] = batch[k].to(device)
    
    if plot:
        ultralytics_yolobatch_draw_boxes(batch=batch, save_to=Path("view"), need_scale=True) # visaulization debug
    
    return batch
