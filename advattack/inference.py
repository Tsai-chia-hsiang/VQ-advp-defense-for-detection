import torch
import sys
import cv2
import os.path as osp
from ultralytics import YOLO
from .ultralytics_utils import get_data_args
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from pathlib import Path
from ultralytics.utils.torch_utils import select_device
from .import _LOCAL_DIR_
sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from torchcvext.box import scale_box, xywh2xyxy, draw_boxes
from torchcvext.convert import tensor2img

def compare_visualization(detector:Path, data:Path, project:Path, name:str, batch:int=16, device:str='0', **kwargs)->None:
    
    save_dir = project/name/"images"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(detector)
    model = model.to(select_device(device=device, verbose=False))

    data_args, dataset_args = get_data_args(
        model_args=model.overrides, 
        stride=max(int(model.model.stride.max()), 32),
        dataset_cfgfile_path = data,
        batch=batch,
        mode='val'
    )
    loader = build_dataloader(
        dataset = build_yolo_dataset(
            cfg = data_args,
            img_path=dataset_args.get('val'),
            batch=data_args.batch, 
            data=dataset_args, 
            mode=data_args.mode, 
            stride = data_args.stride,
            rect=False
        ),
        batch=data_args.batch, shuffle=False, 
        workers=1
    )
    for batch in loader:
        timg = batch['img']
        imgs = tensor2img(timg=timg, scale_back_f=lambda x:x)
        for i, n in zip(imgs, batch['im_file']):
            cv2.imwrite(save_dir/Path(n).name, i)
        



