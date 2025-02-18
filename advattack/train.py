from tqdm import tqdm
import torch
import gc
import sys
from copy import deepcopy
import cv2
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Literal, Any
from pathlib import Path
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import select_device, init_seeds
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader,  build_yolo_dataset, build_dataloader
from ultralytics.nn.tasks import DetectionModel
from .ultralytics_utils import check_model_frozen
from .ultralytics_utils import get_data_args
from .attack import PatchAttacker, preprocess_yolo_batch_with_attack, DEFAULT_ATTACKER_ARGS
from .build import generate_patch
from .loss import v8DetLoss, TotalVariation, NPSCalculator, DEFAULT_PB_FILE
from .writer import TrainingMetricsWriter
from .validation import PatchAttack_DetValidator
from .ultralytics_utils import get_data_args
from .import _LOCAL_DIR_
import os.path as osp

sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from torchcvext.convert import tensor2img
from tools import write_json, refresh_dir, write_yaml

class AdvPatchAttack_YOLODetector_Trainer():

    def __init__(
        self, pretrained_weight:Path, dataset_cfg:Path,
        save_dir:Path, loss_args:dict[str, Any]=None,
        device:int=0, scale_det_loss:bool=False,
        batchsz:int=16, 
        seed:int=891122, 
        attacker_args:dict[str, tuple[float, float]|float]=DEFAULT_ATTACKER_ARGS,
        psize:int=300, ptype:str='gray',
        tensorboard:bool=True
    ):
        """
        To much different from BaseTrainer of Ultralytics, Re-design it.

        Currently only for single GPU training
        TODO: multi-GPUs training

        Args
        --

        """

        init_seeds(seed=seed)
        self.pretrained_weight = pretrained_weight
        self.dataset_cfg = dataset_cfg
        self.device = select_device(device=f"{device}", verbose=False)
        self.batchsz = batchsz
        self.save_dir = refresh_dir(save_dir)
        self.model_args, self.model = self.setup_model()
        
        self.not_improve = 0

        self.data_args, self.dataset_args = get_data_args(
            model_args=self.model_args, 
            stride=max(int(self.model.stride.max()), 32),
            dataset_cfgfile_path = self.dataset_cfg,
            batch=self.batchsz,
            mode='val'
        )

        self.trainloader = self.get_dataloader(split='train')
        self.valloader = self.get_dataloader(split='val')

        self.attacker = PatchAttacker(**attacker_args)

        self.adv_patch = generate_patch(ptype=ptype, psize=psize, device=self.device)
        
        self.optimizer:Optimizer = None
        self.scheduler:LRScheduler = None
        self.method:str=None
        
        self.scale_det_loss = scale_det_loss

        self.loss_dict:dict[str, Any] = {   
            **self._init_det_loss(),
            **self._init_other_loss(**(loss_args if loss_args is not None else {}))
        }

        self.valloader = self.get_validator()
        
        self.epoch_metrics = {'mAP50':None, 'mAP50-95':None, 'fitness':None}
        self.consider = 'mAP50'
        self.not_improve = 0
        self.clean_metrcis = self.eval_clean()
        self.worst = self.clean_metrcis[self.consider] if self.clean_metrcis is not None else 1
        self.det_turn = ['box','cls','dfl']
        self.metrics_writer = TrainingMetricsWriter(
            loss_order = self.det_turn + list(i for i in self.loss_dict.keys() if i!= 'det'),
            metrics_order=list(self.epoch_metrics.keys()),
            file=self.save_dir/f"log.csv",
            tb_dir=self.save_dir if tensorboard else None
        )
    
    def eval_clean(self) -> dict[str, float]|None:
        if self.valloader is not None:
            print(f"evaluating clean data ..")
            M =  self.valloader(model=self.model)
            print(M)
            return M
        return None
    
    def get_validator(self)->PatchAttack_DetValidator|None:
        
        if self.valloader is not None:

            validator = PatchAttack_DetValidator( 
                attacker=self.attacker, 
                save_dir=self.save_dir, 
                args=self.model_args|{
                    'mode':'val','rect':False, 'batch':self.batchsz, 
                    'data':self.dataset_cfg
                }
            ) 
            validator.data = self.dataset_args
            validator.args.plots = False
            
            validator.dataloader = self.valloader

            return validator
        
        return None

    def _init_det_loss(self):
        return {'det':v8DetLoss(model=self.model, **{k:DEFAULT_CFG.get(k) for k in ['box','cls','dfl']})}
    
    def _init_other_loss(self, w_tv:float=2.5, printability_file=DEFAULT_PB_FILE, **kwargs):
        return {
            'tv':TotalVariation(w=w_tv),
            'nps':NPSCalculator(patch_side=self.adv_patch.size(-1), printability_file=printability_file)
        }
    
    def cal_det_loss(self, preds, batch) -> tuple[str, torch.Tensor]:
        
        det:torch.Tensor = self.loss_dict['det'](preds=preds, batch=batch)
        bsz = len(batch["img"])
        
        if self.scale_det_loss:
            det *= bsz
        if self.method == "gd":
            det *= -1
    
        return {'box':det[0], 'cls':det[1], 'dfl':det[2]}
    
    def cal_other_loss(self, preds, batch)->dict[str, torch.Tensor]:
        
        return {
            'tv': self.loss_dict['tv'] (self.adv_patch),
            'nps': self.loss_dict['nps'](self.adv_patch)
        }
    
    def model_forward(self, batch):
        return self.model(batch['img'])

    def train(
        self, method:Literal['gd', 'pgd']='gd', lr:float=2e-2, epoch:int=1000, 
        preprocess_args:dict[str, Any]=None, patience=None
    ):
        pr_args = preprocess_args if preprocess_args is not None else {}
        if patience is None:
            patience = epoch
        
        self.method = method
        if self.method == 'gd':
            self.optimizer = optim.Adam([self.adv_patch], lr=lr, amsgrad=True)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , 'min', patience=50)
         
        loss_log = {k:0 for k in self.metrics_writer.loss_order}

        for _e in range(epoch):
            e = _e+1
            pbar = tqdm(self.trainloader)
            pbar.set_postfix(ordered_dict={'epoch':f"{e}/{epoch}", 'patience':f"{self.not_improve}/{patience}"})
        
            for k in loss_log:
                loss_log[k] = 0
        
            for batch in pbar:
                
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                
                batch = self.preprocess(batch=batch, **pr_args)
                preds = self.model_forward(batch=batch)
                
                tloss = {
                    **self.cal_det_loss(preds=preds, batch=batch), 
                    **self.cal_other_loss(preds=preds, batch=batch)
                }

                total_loss = sum([v for _,v in tloss.items()]) 
                total_loss.backward()
                self.optimizer_step()

                for l in loss_log:
                    loss_log[l] += float(tloss[l].detach())

            
            loss_log = {k:loss_log[k]/len(self.trainloader) for k in loss_log}
            
            self.scheduler_step(loss_log=loss_log, epoch=e, iterations=e*len(self.trainloader))
        
            if self.valloader is not None:
                # TODO: using with torch.no_grad() will cause inference_mode RE, 
                # Try to debug currently
                self.adv_patch.requires_grad = False
                self.epoch_metrics = self.valloader(model=self.model, adv_patch=self.adv_patch, **pr_args)
                self.adv_patch.requires_grad = True
            
            loss_log = {
                k:v*(-1) if  (k in self.det_turn and self.method == 'gd') else v 
                for k, v in loss_log.items()
            }
            self.metrics_writer(epoch=e, loss=loss_log, metrics=self.epoch_metrics)
        
            if self.check_if_update(epoch=e):
                self.save_model(name='worst', epoch=e)
            
            self.save_model(name='last', epoch=e)
            if self.not_improve == patience:
                break
        
        self.metrics_writer.close()
        
        return self.final_eval(preprocess_args=pr_args)
    
    def final_eval(self, preprocess_args:dict=None) -> dict[str, dict[str, float]]|None:
        
        to_test = self.save_dir/"worst.pt"
        
        if self.valloader is None:
            print("No validation set to evaluation, return None")
            return None
        
        if not to_test.is_file():
            print(f"attack fail! non of those epoch {self.consider} is worse than {self.worst}, return None")
            return None

        patch = torch.load(to_test, weights_only=False)['patch']
        FM = self.valloader(model=self.model, adv_patch=patch, **preprocess_args)
        
        cmp = {
            k:{
                'clean':self.clean_metrcis[k],
                'attack':FM[k]
            }
            for k in FM
        }
        eva_file = self.save_dir/"final_eval.json"
        print(f"final result is saved at {eva_file}, please check")
        write_json(o=cmp, f=eva_file)
        
        self._clear_memory()
        #print(cmp)
        return cmp

    def check_if_update(self, epoch:int) -> bool:
    
        if self.epoch_metrics[self.consider] is None:
            return False
    
        if self.epoch_metrics[self.consider] <= self.worst:
            print(f"Get worse adversaral patch that makes mAP from {self.worst} to {self.epoch_metrics[self.consider]} at {epoch}")
            self.worst = self.epoch_metrics[self.consider]
            self.not_improve = 0
            return True
        
        self.not_improve += 1
        return False

    def setup_model(self) -> tuple[dict, DetectionModel]:
        model = YOLO(self.pretrained_weight)
        model.train
        model = model.to(device=self.device) #TODO: multi-GPUs training
        check_model_frozen(model.model)
        return model.overrides, model.model
    
    def save_model(self, name:str, epoch:int):
    
        assert self.adv_patch.max() <= 1 and self.adv_patch.min() >= 0
    
        to_save = {'patch':self.adv_patch.detach().cpu(), 'epoch':epoch}
    
        if self.epoch_metrics['mAP50'] is not None:
            to_save = to_save|self.epoch_metrics
        
        torch.save(to_save, self.save_dir/f"{name}.pt")
        cv2.imwrite(self.save_dir/f"{name}.png", tensor2img(self.adv_patch))
    
    def get_dataloader(self, split:Literal['train', 'val']) -> InfiniteDataLoader|None:
    
        if not self.dataset_args.get(split, False):
            return None

        return build_dataloader(
            dataset = build_yolo_dataset(
                cfg=self.data_args,  
                img_path=self.dataset_args.get(split), 
                batch=self.data_args.batch, 
                data=self.dataset_args, 
                mode=self.data_args.mode, 
                stride = self.data_args.stride
            ),
            batch=self.data_args.batch, 
            shuffle= split == 'train', 
            workers=self.data_args.workers
        )

    def preprocess(self, batch, random_rotate_patch:bool=True, debug:bool=False, **kwargs)-> dict[str, Any]:
        
        return preprocess_yolo_batch_with_attack(
            attacker=self.attacker, 
            patch=self.adv_patch, 
            device=self.device,
            batch=batch,  
            random_rotate_patch=random_rotate_patch,
            plot=debug
        )

    def optimizer_step(self, lr:float=None)->None:
        match self.method:
            case 'gd':
                self.optimizer.step()
            case 'pgd':
                with torch.no_grad():
                    self.adv_patch.data = self.adv_patch.data + lr * self.adv_patch.grad.sign()
        
        self.adv_patch.data.clamp_(0, 1)

    def scheduler_step(self, loss_log:dict[str, float], epoch:int, iterations:int)->None:
        if self.scheduler is not None:
            self.scheduler.step(sum([v for _,v in loss_log.items()]))

    def _clear_memory(self):
        """Clear accelerator memory on different platforms., from ultralytics"""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

