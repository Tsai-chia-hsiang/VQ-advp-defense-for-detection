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
from deepcvext import tensor2img
from tools import load_yaml

class AdvPatchAttack_YOLODetector_Validator(DetectionValidator):
    
    def __init__(
        self, reference_model:YOLO, save_dir:Path,
        data:Optional[Path]=None, loader:Optional[InfiniteDataLoader]=None,
        conf:float=0.25, attacker:Path|PatchAttacker=DEFAULT_ATTACKER_CFG_FILE, 
        imgsz:int=640, batch:int=16, **kwargs
    ):
        assert any([data, loader])
        data_args, dataset_args = get_data_args(
            model_args={**reference_model.overrides,**{'imgsz':imgsz}}, 
            dataset_cfgfile_path=data,
            stride=max(int(reference_model.model.stride.max()), 32),
            batch=batch
        )
        if loader is None:
            loader = get_dataloader(
                data_args=data_args, dataset_args=dataset_args, split='val',
                rect=True
            )
            
        args = {
            **reference_model.overrides,
            'mode':'val', 'rect':kwargs.get('rect', True), 'batch':batch, 'data':data,
            'plots':False, 'conf':conf
        }
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(args=args, save_dir=save_dir, dataloader=loader)# , dataloader=loader, save_dir=save_dir, pbar=pbar, _callbacks=_callbacks)
        self.data = dataset_args
        self.training = False
        self.attacker = PatchAttacker(**load_yaml(attacker)) if isinstance(attacker, Path) else attacker
   
    """
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
    """
    
    @torch.no_grad()
    def __call__(self, model:DetectionModel, adv_patch:Optional[torch.Tensor]=None, defenser=None, **pr_args):
      
        def setup():
            dp = de_parallel(model)
            self.device = next(dp.parameters()).device
            self.init_metrics(dp)
        setup()

        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch=batch, adv_patch=adv_patch, defenser=defenser, **pr_args)
            preds = model(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        stats = {k.replace("metrics/","").replace("(B)", ""):float(v) for k,v in stats.items()}
        
        return stats
 
    def preprocess(self, batch:dict[str, Any], adv_patch:torch.Tensor=None, patch_random_rotate:bool=False, patch_blur:bool=False, debug:bool=False, defenser=None, **kwargs):
        batch = super().preprocess(batch)
        if adv_patch is not None:
            batch = self.attacker.attack_yolo_batch(
                patch=adv_patch, batch=batch,  
                patch_random_rotate=patch_random_rotate, patch_blur=patch_blur, 
                plot=debug
            )
        if defenser is not None:
            batch = defenser(batch=batch)
        
        return batch
    
    
    def comparsion(self, model:DetectionModel, adv_patch:torch.Tensor, defenser=None, **kwargs) -> dict[str, dict[str, float]]:
        """
        Compare the result of clean data with data under adversarial patch attack using adv_patch.  
        """ 
        print(f"clean image evaluation")
        clean_metrics = self(model=model, adv_patch=None, **kwargs)
        clean_defense_metrics = {k:None for k in clean_metrics}
        attack_metrics = {k:None for k in clean_metrics}
        attack_defense_metrics = {k:None for k in clean_metrics}

        if adv_patch is not None:
            print(f"attacked image evaluation")
            attack_metrics = self(model=model, adv_patch=adv_patch, **kwargs)
        if defenser is not None:
            print(f"clean image with {defenser}")
            clean_defense_metrics = self(model=model, adv_patch=None, defenser=defenser, **kwargs)
            print(f"attacked image with {defenser}")
            attack_defense_metrics = self(model=model, adv_patch=adv_patch, defenser=defenser, **kwargs)
        
        return {
            k:{
                'clean':clean_metrics[k],
                'clean_defense':clean_defense_metrics[k],
                'attack':attack_metrics[k],
                'attack_defense':attack_defense_metrics[k]
            }
            for k in clean_metrics
        }

