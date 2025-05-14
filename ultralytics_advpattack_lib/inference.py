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
from ultralytics.engine.results import Results
from .ultralytics_utils import get_data_args
from .ultralytics_utils import ultralytics_yolobatch_det_draw_boxes
from .attacker import DEFAULT_ATTACKER_CFG_FILE, PatchAttacker
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from pathlib import Path
from ultralytics.utils.torch_utils import select_device, init_seeds
from .import _LOCAL_DIR_
sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from tools import load_yaml
from deepcvext import tensor2img, img2tensor


def make_img_np_dir(root:Path)->tuple[Path, Path]:
    arr = root/"arr"
    arr.mkdir(parents=True,exist_ok=True)
    img = root/"img"
    img.mkdir(parents=True, exist_ok=True)
    return (img, arr)


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
      
def attack(
    pretrained_patch:Path, save_dir:Path, 
    data:Path, split:str='val',
    attacker:Path=DEFAULT_ATTACKER_CFG_FILE, device:str='0',
    patch_transform_args:dict=None
)->None:   
    dev = select_device(device=device)
    patch = torch.load(pretrained_patch, weights_only=False, map_location=dev)['patch']
    advp_attacker = PatchAttacker(**load_yaml(attacker))
    data_cfg:dict = load_yaml(data)
    dataroot = Path(data_cfg['path'])/f"{data_cfg[split]}"
    imgs_dir = dataroot/"images"
    
    labels_dir = dataroot/"labels"
    assert imgs_dir.is_dir(), FileNotFoundError(f"No {imgs_dir}")
    assert labels_dir.is_dir(), FileNotFoundError(f"No {labels_dir}")
    
    img_save_dir = save_dir/split/"images"
    label_save_dir = save_dir/split/"labels"
    assert img_save_dir != imgs_dir
    mask_save_dir = save_dir/f"{split}_mask"

    label_save_dir.mkdir(parents=True, exist_ok=True)
    img_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)

    img_paths:list[Path] = [_ for _ in imgs_dir.iterdir()]
   
    for i in tqdm(img_paths):
        shutil.copy(labels_dir/f"{i.stem}.txt", label_save_dir/f"{i.stem}.txt")
        label:np.ndarray = np.loadtxt(labels_dir/f"{i.stem}.txt")
        if len(label) == 0:
            shutil.copy(i, img_save_dir/i.name)
            continue
        if label.ndim < 2:
            label = np.expand_dims(label, axis=0)
        
        
        img = img2tensor(cv2.imread(i)).to(device=dev)
        label = torch.from_numpy(label[:, 1:]).to(device=dev)
        
        img, masks = advp_attacker(
            img=img, patch=patch, bboxes=label, 
            batch_idx=torch.zeros(len(label)),
            **patch_transform_args
        )
    
        cv2.imwrite(img_save_dir/i.name, img=tensor2img(img))
        masks = tensor2img(masks)
        m_save = mask_save_dir/i.stem
        m_save.mkdir(parents=True, exist_ok=True)
        if isinstance(masks, np.ndarray):
            cv2.imwrite(m_save/f"0.png", img=masks)
        elif isinstance(masks, list):
            for bi, mi in enumerate(masks):
                cv2.imwrite(m_save/f"{bi}.png", img=mi)
        else:
            raise TypeError(f"{type(masks)} is not a list of ndarray or ndarray")

