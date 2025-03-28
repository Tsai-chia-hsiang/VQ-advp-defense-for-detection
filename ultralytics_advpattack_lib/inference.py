import torch
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
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
from tools import load_yaml, write_json
from deepcvext.draw import canvas
from deepcvext import tensor2img

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


def compare_visualization(
    detector:Path, pretrained_patch:Path, save_dir:Path,
    data:Path, attacker:Path=DEFAULT_ATTACKER_CFG_FILE, 
    batch:int=16, device:str='0',conf:float=0.25,
    seed:int=89112, deterministic:bool=True, imgsz:int=512,
    **pr_arg
)->None:
    
    init_seeds(seed=seed, deterministic=deterministic)

    save_dir = save_dir/"images"
    save_dir_tree = {k:make_img_np_dir(save_dir/k) for k in ["clean", "attacked"]}
    save_dir_tree['cmp'] = save_dir/"cmp"
    save_dir_tree['cmp'].mkdir(parents=True, exist_ok=True)


    model = YOLO(detector)
    model = model.to(select_device(device=device, verbose=False))
    patch = torch.load(pretrained_patch, weights_only=False, map_location='cpu')['patch'].to(device=model.device)
    advp_attacker = PatchAttacker(**load_yaml(attacker))
    
    data_args, dataset_args = get_data_args(
        model_args={**(model.overrides),**{'imgsz':imgsz}}, 
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
    log = {'less':[],'redundant':[]}

    for batch in tqdm(loader):

        batch["img"] = batch["img"].to(dtype=torch.float32, device=model.device)/255.0
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(model.device)
        
        clean_imgs = tensor2img(batch["img"])
        clean_arr = batch["img"].detach().permute(0, 2, 3, 1).cpu().numpy()
        clean_preds = pred_once(model=model, batch=batch, conf=conf)
        
        clean_pred_imgs = ultralytics_yolobatch_det_draw_boxes(
            batch=batch, preds=clean_preds, labels=model.names, 
            need_scale=True, return_img=True, wanted_cls=[0]
        )
        attack_batch = advp_attacker.attack_yolo_batch(batch=batch, patch=patch, plot=False, **pr_arg)
        dirty_arr = attack_batch["img"].detach().permute(0, 2, 3, 1).cpu().numpy()
        dirty_imgs = tensor2img(attack_batch["img"])
        dirty_pred = pred_once(model=model, batch=attack_batch, conf=conf)
        
        dirty_pred_imgs = ultralytics_yolobatch_det_draw_boxes(
            batch=batch, preds=dirty_pred, labels=model.names, 
            need_scale=True, return_img=True, wanted_cls=[0]
        )


        for idx, name in enumerate(batch["im_file"]):
    
            clean_pred_person = ((clean_preds[idx][:, -1].astype(np.int32)) == 0).astype(np.int32).sum()
            dirty_pred_person = ((dirty_pred[idx][:, -1].astype(np.int32)) == 0).astype(np.int32).sum()
    
            cv2.imwrite(
                save_dir_tree['cmp']/Path(name).name,
                canvas(
                    imlist=[clean_imgs[idx], clean_pred_imgs[idx], dirty_imgs[idx], dirty_pred_imgs[idx]],  
                )
            )
            its_path = Path(name)
            cv2.imwrite(save_dir_tree['clean'][0]/its_path.name, clean_imgs[idx])
            np.save(save_dir_tree['clean'][1]/its_path.stem, clean_arr[idx])

            cv2.imwrite(save_dir_tree['attacked'][0]/its_path.name, dirty_imgs[idx])
            np.save(save_dir_tree['attacked'][1]/its_path.stem,  dirty_arr[idx])
        
            if dirty_pred_person < clean_pred_person:
                log['less'].append({
                    'name':str(Path(name).name),
                    'clean':int(clean_pred_person),
                    'dirty':int(dirty_pred_person)
                })
            elif dirty_pred_person> clean_pred_person:
                log['redundant'].append({
                    'name':str(Path(name).name),
                    'clean':int(clean_pred_person),
                    'dirty':int(dirty_pred_person)
                })
    
    for k in log:
        print(f"{k}: {len(log[k])}/{len(loader.dataset)}")
    
    write_json(log, save_dir/f"cmp_person.json")
        
    