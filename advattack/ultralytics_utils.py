from pathlib import Path
import gc
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.data.dataset import YOLODataset
import sys
import os.path as osp
from .import _LOCAL_DIR_
sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from torchcvext.box import scale_box, xywh2xyxy, draw_boxes
from torchcvext.convert import tensor2img

def display_yolodataset_arguments(data_args:IterableSimpleNamespace, dataset_args:IterableSimpleNamespace) -> None:

    print(f"Arguments for {YOLODataset}")
    print(f"-"*50)
    print(f"task:{data_args.task}, mode:{data_args.mode}")
    print(f"imgsz:{data_args.imgsz}, classes:{data_args.classes}")
    print(f"rect:{data_args.rect}, cache:{data_args.cache}")
    print(f"stride: {data_args.stride}, batchsize:{data_args.batch}, worker:{data_args.workers}")
    print(f"[For training mode only] augment:{data_args.augment}, fraction:{data_args.fraction}")
    print()
    print(f"Arguments for transform (Please note that `mask_ratio` and `overlap_mask` are NOT for detection, here is just checking)")
    print(f"-"*50)
    print(f"mask_ratio:{data_args.mask_ratio}, overlap_mask:{data_args.overlap_mask}, bgr:{data_args.bgr}")
    print()
    print(f"dataset argument")
    print(f"-"*50)
    for i in dataset_args:
        if i != "names":
            v = dataset_args.get(i)
            print(i,v, f"({type(v)})")

def get_data_args(model_args:dict, stride:int, dataset_cfgfile_path:Path, mode:str='val', **kwargs) -> tuple[IterableSimpleNamespace, IterableSimpleNamespace]:
    
    default_data_args = {
        k:DEFAULT_CFG.get(k) for k in [
            'rect', 'classes', 'augment', 'fraction', 'cache',
            'mask_ratio', 'overlap_mask','bgr', 'batch', 'workers'
        ]
    }
    args = model_args|default_data_args|{'mode':mode, 'data':dataset_cfgfile_path}
    
    for k in kwargs:
        args[k] = kwargs[k]
    
    args['stride'] = stride
    args['imgsz'] = check_imgsz(args['imgsz'], stride=stride, max_dim=1)

    args = IterableSimpleNamespace(**args)
    dataset_args = check_det_dataset(args.data)
    return args, dataset_args

def check_model_frozen(model:nn.Module):
    for k,v in model.named_parameters():
        if v.requires_grad:
            print(f"detector {k} is not freeze yet, freeze it")
            v.requires_grad = False

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