from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import torch
import numpy as np
import cv2
from ultralytics.data.build import InfiniteDataLoader
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import de_parallel
from .attacker import PatchAttacker, DEFAULT_ATTACKER_CFG_FILE
from .ultralytics_utils import get_data_args, get_dataloader
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent))
from torchcvext import tensor2img
from tools import load_yaml
from taming_transformers.ultray_do import ultralytics_batch_recons


class AdvPatchAttack_YOLODetector_Validator(DetectionValidator):
    
    def __init__(
        self, detector:Path|YOLO, save_dir:Path,
        data:Optional[Path]=None, loader:Optional[InfiniteDataLoader]=None,
        conf:float=0.25,
        attacker:Path|PatchAttacker=DEFAULT_ATTACKER_CFG_FILE, 
        imgsz:int=512, batchsz:int=16, 
        pbar=None, _callbacks=None, **kwargs
    ):
        assert any([data, loader])
        if loader is None:
            assert detector is not None, "no loader, need `data` to build loader"
        
        reference_model=YOLO(detector) if isinstance(detector, Path) else detector
        
        loader = AdvPatchAttack_YOLODetector_Validator.get_dataloader_according_YOLO(
            reference_model,
            dataset_cfgfile_path=data, 
            imgsz=imgsz, batch=batchsz
        ) if loader is None else loader
 
        args = {
            **(reference_model.overrides),
            **{'mode':'val', 'imgsz':loader.dataset.imgsz,'rect':loader.dataset.rect, 'batch':loader.batch_size}
        }
        save_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(dataloader=loader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)
        self.args.plots = False
        self.data = self.dataloader.dataset.data
        self.attacker = PatchAttacker(**load_yaml(attacker)) if isinstance(attacker, Path) else attacker
        self.training = False
        self.args.conf = conf
    
    @staticmethod
    def get_dataloader_according_YOLO(reference_model:YOLO, dataset_cfgfile_path:Path, imgsz:int=512, batch:int=16)->InfiniteDataLoader:
        data_args, dataset_args = get_data_args(
            model_args={**(reference_model.overrides),**{'imgsz':imgsz}}, 
            dataset_cfgfile_path=dataset_cfgfile_path,
            stride=max(int(reference_model.model.stride.max()), 32),
            batch=batch,
        )
        print(dataset_args)
        data_args.workers = 1
        assert 'val' in dataset_args

        return get_dataloader(data_args=data_args, dataset_args=dataset_args, split='val')
  
    def init_metrics(self, model:nn.Module):
        self.device = next(model.parameters()).device
        return super().init_metrics(model)
    
    def _save_img(self, batch)->None:
        imgs = tensor2img(batch["img"])
        array = batch["img"].detach().permute(0, 2, 3, 1).cpu().numpy()
        if not (self.save_dir/"img").is_dir():
            (self.save_dir/"img").mkdir(parents=True, exist_ok=True)
        if not (self.save_dir/"np").is_dir():
            (self.save_dir/"np").mkdir(parents=True, exist_ok=True)
        for im, a, name in zip(imgs, array, batch['im_file']):
            cv2.imwrite(self.save_dir/"img"/Path(name).name, im)
            np.save(self.save_dir/"img"/Path(name).name)
        
    def __call__(self, model:DetectionModel, adv_patch:Optional[torch.Tensor]=None, **pr_args):
                
        self.init_metrics(de_parallel(model))
        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch=batch, adv_patch=adv_patch, **pr_args)
            preds = model(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            
        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        if not self.training:
            self.args.verbose=True
            self.print_results()
        stats = {k.replace("metrics/","").replace("(B)", ""):float(v) for k,v in stats.items()}
        
        return stats
    
    def preprocess(self, batch:dict[str, Any], adv_patch:torch.Tensor=None, patch_random_rotate:bool=False, patch_blur:bool=False, debug:bool=False, vq:bool=False, **kwargs):
        batch = super().preprocess(batch)
        if adv_patch is not None:
            batch = self.attacker.attack_yolo_batch(
                patch=adv_patch, batch=batch,  
                patch_random_rotate=patch_random_rotate, patch_blur=patch_blur, 
                plot=debug
            )
        if vq:
            batch = ultralytics_batch_recons(batch=batch)
        
        return batch

    def comparsion(self, model:DetectionModel, adv_patch:torch.Tensor, vq:bool=False, **kwargs) -> dict[str, dict[str, float]]:
        """
        Compare the result of clean data with data under adversarial patch attack using adv_patch.  
        """ 
        
        clean_metrics = self(model=model, adv_patch=None, **kwargs)
        
        attack_metrics = {k:None for k in clean_metrics}
        vq_attack_metrics = {k:None for k in clean_metrics}

        if adv_patch is not None:
            attack_metrics = self(model=model, adv_patch=adv_patch, **kwargs)
        if vq:
            vq_attack_metrics = self(model=model, adv_patch=adv_patch, vq=vq, **kwargs)
        
        return {
            k:{
                'clean':clean_metrics[k],
                'attack':attack_metrics[k],
                'vq_fix':vq_attack_metrics[k]
            }
            for k in clean_metrics
        }

