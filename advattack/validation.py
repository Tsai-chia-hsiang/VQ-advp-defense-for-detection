from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import torch
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils.torch_utils import de_parallel, smart_inference_mode
from .attacker import PatchAttacker
from .ultralytics_utils import get_data_args

class PatchAttack_DetValidator(DetectionValidator):
    
    def __init__(self, attacker:PatchAttacker, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.attacker = attacker

    def init_metrics(self, model:nn.Module):
        self.device = next(model.parameters()).device
        return super().init_metrics(model)
    
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
        
        stats = {k.replace("metrics/","").replace("(B)", ""):float(v) for k,v in stats.items()}
        
        return stats
    
    def preprocess(self, batch:dict[str, Any], adv_patch:torch.Tensor=None, patch_random_rotate:bool=False, patch_blur:bool=False, debug:bool=False, **kwargs):
        batch = super().preprocess(batch)
        if adv_patch is not None:
            batch = self.attacker.attack_yolo_batch(
                patch=adv_patch, batch=batch,  
                patch_random_rotate=patch_random_rotate, patch_blur=patch_blur, 
                plot=debug
            )
        return batch

    def comparsion(self, model:DetectionModel, adv_patch:torch.Tensor, **kwargs) -> dict[str, dict[str, float]]:
        """
        Compare the result of clean data with data under adversarial patch attack using adv_patch.  
        """ 
        clean_metrics = self(model=model, adv_patch=None,**kwargs)
        attack_metrics = self(model=model, adv_patch=adv_patch, **kwargs)
        return {
            k:{
                'clean':clean_metrics[k],
                'attack':attack_metrics[k]
            }
            for k in clean_metrics
        }

    @classmethod
    def build_according_YOLO_with_dataset(cls, data:Path, attacker:PatchAttacker, model:YOLO=None, model_args:dict=None, rect:bool=False, batch:int=16, save_dir:Optional[Path]=None)->"PatchAttack_DetValidator":
        """
        Need to pass at least 1 of `model`, `model_args`.
        - If both are pass, using model.overrides as default 
        """

        data_args, dataset_args = get_data_args(
            model_args=model.overrides if model is not None else model_args, 
            stride=max(int(model.model.stride.max()), 32),
            dataset_cfgfile_path = data,
            batch=batch,
            mode='val'
        )
        data_args.workers = 1
        loader = build_dataloader(
            dataset = build_yolo_dataset(
                cfg = data_args,
                img_path=dataset_args.get('val'),
                batch=data_args.batch, 
                data=dataset_args, 
                mode=data_args.mode, 
                stride = data_args.stride,
                rect=rect
            ),
            batch=data_args.batch, shuffle=False, 
            workers=data_args.workers
        )

        validator = cls( 
            dataloader=loader,
            attacker=attacker,
            save_dir=save_dir,
            args=model.overrides|{'mode':'val','rect':rect, 'batch':batch, 'data':data},
            _callbacks=model.callbacks
        )
        validator.data = dataset_args
        validator.args.plots = False
        return validator