import torch
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
import shutil
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from ultralytics.utils import ops
from .attacker import DEFAULT_ATTACKER_CFG_FILE, PatchAttacker
from pathlib import Path
from ultralytics.utils.torch_utils import select_device
from . import _LOCAL_DIR_
sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from tools import load_yaml
from deepcvext import tensor2img, img2tensor
from deepcvext.box import scale_box


def pred_once(model:YOLO, batch, conf:float=0.25) -> list[np.ndarray]:
    preds = model.model(batch["img"])
    result = ops.non_max_suppression(
        prediction=preds,
        conf_thres=conf,
        iou_thres=DEFAULT_CFG.iou,
        classes=DEFAULT_CFG.classes,
        agnostic=DEFAULT_CFG.agnostic_nms,
        max_det=DEFAULT_CFG.max_det,
        nc=len(model.names),
    )
    for i in range(len(result)):
        # [x,y,x,y,conf,cls]
        result[i] = result[i].cpu().numpy()
    return result

def pad_to_square(img:np.ndarray)->tuple[np.ndarray, tuple[int]]:
    s = np.asarray(img.shape[:2])
    min_axi = np.argmin(s)
    to_pad = np.zeros_like(s)
    to_pad[min_axi] = s[1-min_axi] - s[min_axi]
    final_s = s + to_pad
    pad_img = np.zeros((*final_s, 3), dtype=img.dtype)
    pad_img[:s[0], :s[1]] = img
    return pad_img, img.shape[:2]
    
def attack(
    pretrained_patch:Path, save_dir:Path,  attack_cls:list[int], 
    data:Path, split:str='val', attacker:Path=DEFAULT_ATTACKER_CFG_FILE, device:str='0',
    square=True, patch_transform_args:dict=None, 
)->None:   
    
    dev = select_device(device=device, verbose=False)
    patch = torch.load(pretrained_patch, weights_only=False, map_location=dev)['patch']
    advp_attacker = PatchAttacker(**load_yaml(attacker))
    data_cfg:dict = load_yaml(data)

    dataroot = Path(data_cfg['path'])/f"{data_cfg[split]}"
    imgs_dir = dataroot/"images"
    labels_dir = dataroot/"labels"
    assert imgs_dir.is_dir(), FileNotFoundError(f"No {imgs_dir}")
    assert labels_dir.is_dir(), FileNotFoundError(f"No {labels_dir}")
    
    img_save_dir = save_dir/"attacked"/split/"images"
    
    assert img_save_dir != imgs_dir, FileExistsError(f"save dir: {img_save_dir} and src dir:{imgs_dir} are the same, this will overwrite original images")
    mask_save_dir = save_dir/"attacked"/f"{split}_mask"
    
    img_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)

    img_paths:list[Path] = [_ for _ in imgs_dir.iterdir()]
    att_cls = np.asarray(attack_cls)
    
    for i in tqdm(img_paths):
        
        label:np.ndarray = np.loadtxt(labels_dir/f"{i.stem}.txt")
        im0:np.ndarray = cv2.imread(i)
        s = im0.shape[:2]

        if len(label) == 0:
            # no sample
            shutil.copy(i, img_save_dir/i.name)
            cv2.imwrite(m_save/"union.png", np.zeros_like(im0)[:, :, 0:1])
            continue
        
        if label.ndim < 2:
            label = np.expand_dims(label, axis=0)
        
        label = label[np.isin(label[:, 0].astype(np.int32), att_cls), 1:]
        
        if len(label) == 0:
            # no positive sample
            shutil.copy(i, img_save_dir/i.name)
            cv2.imwrite(m_save/"union.png", np.zeros_like(im0)[:, :, 0:1])
            continue
        
        if square:
            im0, s = pad_to_square(img=im0)
            label = scale_box(boxes=label, imgsize=s[::-1], direction='back')
            label = scale_box(boxes=label, imgsize=im0.shape[:2][::-1])

        img = img2tensor(img=im0).to(device=dev)
        label = torch.from_numpy(label).to(device=dev)
        
        img, masks = advp_attacker(
            img=img, patch=patch, bboxes=label, 
            batch_idx=torch.zeros(len(label)),
            **patch_transform_args
        )
        union_mask = torch.sum(masks, dim=0, keepdim=True)
        union_mask = torch.where(union_mask>0, 1, 0).to(dtype=torch.int32)
        
        union_mask = tensor2img(union_mask)
        img = tensor2img(img)
        masks = tensor2img(masks)

        cv2.imwrite(img_save_dir/i.name, img=img[:s[0], :s[1]])
        
        m_save = mask_save_dir/i.stem
        m_save.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(m_save/"union.png", union_mask[:s[0], :s[1]])
        if isinstance(masks, np.ndarray):
            cv2.imwrite(m_save/f"0.png", img=masks[:s[0], :s[1]])
        elif isinstance(masks, list):
            for bi, mi in enumerate(masks):
                cv2.imwrite(m_save/f"{bi}.png", img=mi[:s[0], :s[1]])
        else:
            raise TypeError(f"{type(masks)} is not a list of ndarray or ndarray")

