import sys
from pathlib import Path
import os.path as osp
import math
import torch
import torch.nn.functional as F
from .ultralytics_utils import ultralytics_yolobatch_draw_boxes
from .augment import MedianPool2d
from .import _LOCAL_DIR_, _CFG_DIR_
sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from deepcvext.box import scale_box, box_geo_scale

DEFAULT_ATTACKER_CFG_FILE = _CFG_DIR_/"attacker.yaml"

class PatchAttacker():

    def __init__(
        self, 
        contrast:tuple[float, float] = (0.8,1.2), 
        brightness:tuple[float, float]=(-0.1, 0.1), 
        angle:tuple[float, float]=(-20, 20), 
        noise:tuple[float, float]=(-1, 1), 
        noise_factor:float=0.1, 
        patch_box_scale:float=0.2, 
        y_shift:float=-0.05,
        blur_ksize:int=7
    ):
        
        self.medianpooler = MedianPool2d(blur_ksize, same=True)
        self.contrast = contrast
        self.brightness = brightness
        self.noise = noise
        self.noise_factor = noise_factor
        self.patch_box_scale = patch_box_scale
        self.y_shift = y_shift
        self.rotate_angle = (angle[0]/ 180 * math.pi, angle[1]/ 180 * math.pi)

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
        
    def __call__(self, img:torch.Tensor, patch:torch.Tensor, bboxes:torch.Tensor, batch_idx:torch.Tensor, patch_random_rotate:bool=True, patch_blur:bool=True):
        
        assert img.device == patch.device
        dev = img.device
        origin_box = scale_box(boxes=bboxes, imgsize=img.size()[::-1][:2], direction='back')
        patch = patch.unsqueeze(0)
        if patch_blur:
            patch = self.medianpooler(patch)
        
        psize = patch.size()[1:] # CxHxW
        pad = ((img.size(-1) - patch.size(-1)) / 2, (img.size(-2) - patch.size(-2)) / 2 )# W pad, H pad
        n_boxes = bboxes.size(0)
        
        contrast = PatchAttacker.uniform_mask(uniform_bound=self.contrast, psize=psize, n_instantces=n_boxes, device=dev)
        brightness = PatchAttacker.uniform_mask(uniform_bound=self.brightness, psize=psize, n_instantces=n_boxes, device=dev)
        noise = torch.empty(brightness.size(),dtype=torch.float32, device=dev).uniform_(self.noise[0], self.noise[1]) * self.noise_factor
        patch = patch.expand(n_boxes, *psize)
        patch = patch*contrast + brightness + noise
        patch = torch.clamp(patch, 0.000001, 0.99999)

        patch = F.pad(patch, (int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), mode='constant', value=0)
        angle =None
        if patch_random_rotate:
            angle = torch.empty(n_boxes, dtype=torch.float32, device=dev).uniform_(self.rotate_angle[0], self.rotate_angle[1])
        else:
            angle = torch.zeros(n_boxes, dtype=torch.float32, device=dev)
        
        geo_scale = (box_geo_scale(origin_box, scale=self.patch_box_scale)/min(psize[1:])).to(device=dev)
        
        patch = self.patch_affine_transformation(patch=patch, xywh=bboxes, angle=angle, scale=geo_scale)

        # for each img, attacking it independently
        for i in range(img.size(0)):
            bidx = torch.where(batch_idx == i)[0]
            using_patch = patch[bidx]
            # each slice of the patch is for a bbox
            for obj_adv_patch in using_patch:
                img[i] = torch.where((obj_adv_patch == 0), img[i], obj_adv_patch)
        
        return img

    def attack_yolo_batch(self, batch, patch:torch.Tensor, patch_random_rotate:bool=False, patch_blur:bool=True, plot:bool=False, **kwargs) -> dict:
        """
        Apply the transformation in place at the batch level, 
        
        ## Please run the process inside `preprocess()` function in Ultralytics' Trainer and Validator first then call this function
            - img and label move to device, normalize the image
        
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
        batch, whos value of "imgs" is the attacked version of origin "img"
        """
        
        batch["img"] = self(
            patch = patch, img=batch["img"],  
            bboxes=batch['bboxes'], 
            batch_idx=batch['batch_idx'], 
            patch_random_rotate=patch_random_rotate,
            patch_blur=patch_blur
        )
        if plot:
            # visaulization debug
            ultralytics_yolobatch_draw_boxes(batch=batch, save_to=Path("view"), need_scale=True) 

        return batch