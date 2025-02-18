from pathlib import Path
import torch.nn as nn
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.data.dataset import YOLODataset

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

