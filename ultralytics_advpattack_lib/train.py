from tqdm import tqdm
import torch
import gc
import sys
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Literal, Any, Optional
from pathlib import Path
from ultralytics.utils.torch_utils import select_device, init_seeds, EarlyStopping
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.nn.tasks import DetectionModel
from .ultralytics_utils import ( 
    get_data_args, 
    get_dataloader,
    setup_YOLOdetection_model
)
from .attacker import PatchAttacker, DEFAULT_ATTACKER_CFG_FILE
from .loss import *
from .writer import TrainingMetricsWriter
from .validation import AdvPatchAttack_YOLODetector_Validator
from .ultralytics_utils import get_data_args
from .import _LOCAL_DIR_
import os.path as osp

sys.path.append(osp.abspath(_LOCAL_DIR_.parent))
from deepcvext.convert import tensor2img
from tools import write_json, refresh_dir, load_yaml, write_yaml

def generate_patch(ptype:Literal['gray', 'random']="gray", psize:int=300, init_patch:Path=None, device:torch.device = torch.device("cpu")) -> nn.Parameter:
    """
    Generate a random patch as a starting point for optimization.
    """
    
    adv_patch:torch.Tensor=None
    if init_patch is not None:
        adv_patch = torch.load(init_patch, map_location=device, weights_only=False)['patch']
    else:
        match ptype:
            case 'gray':
                adv_patch = torch.full((3, psize, psize), 0.5, device=device)
            case 'random':
                adv_patch = torch.rand((3, psize, psize), device=device)
        
    return nn.Parameter(adv_patch)


class AdvTrain_EarlyStopping(EarlyStopping):
    
    def __init__(self, current_worst:float, patience:Optional[int]=None):
        super().__init__(patience)
        self.best_fitness = current_worst
    
    def __call__(self, epoch:int, fitness:float)->tuple[bool, bool]:
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
            (bool): True if validation result is better
        """
        is_improve = False
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False, is_improve

        if fitness <= self.best_fitness:  # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
            is_improve = True
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded

        return stop, is_improve


class AdvPatchAttack_YOLODetector_Trainer():

    def __init__(
        self, detector:Path, attack_cls:Path, data:Path,
        save_dir:Path, loss_args:dict[str, Any]=None,
        device:str='0', batch:int=16, seed:int=891122, deterministic:bool=True, 
        imgsz:int=640, psize:int=300, ptype:str='random',
        init_patch:Path=None,
        conf:float=0.25, attacker:Path=DEFAULT_ATTACKER_CFG_FILE, 
        objective:Literal['cls', 'obj', 'obj-cls']='obj-cls',
        sup_prob_loss:bool=False, tensorboard:bool=True,
        **kwargs
    ):
        """
        To much different from BaseTrainer of Ultralytics, Re-design it.

        Currently only for single GPU training
        TODO: multi-GPUs training

        Args
        --

        """
        
        init_seeds(seed=seed, deterministic=deterministic)

        self.detector = detector
        self.save_dir = refresh_dir(save_dir)
        self.attack_cls = torch.tensor(load_yaml(attack_cls)).to(dtype=torch.long)
        self.device = select_device(device=f"{device}", verbose=False)
        self.batchsz = batch
        self.imgsz = imgsz
        self.conf=conf
        self.data=data
        self.objective = objective

        M = setup_YOLOdetection_model(model=self.detector, imgsz=self.imgsz, device=self.device)
        
        self.attacker = PatchAttacker(**load_yaml(attacker))
        self.adv_patch = generate_patch(ptype=ptype, psize=psize, device=self.device, init_patch=init_patch)
        
        self.optimizer:Optimizer = None
        self.scheduler:LRScheduler = None
        self.method:str=None

        
        self.trainloader, self.valloader = self.get_train_val_loaders(model=M)
        self.valloader = self.get_validator(model=M)

        self.model:DetectionModel = M.model
        self.loss_dict:dict[str, Any] = {   
           **self._init_prob_loss(supervised=sup_prob_loss), 
            **self._init_other_loss(**(loss_args if loss_args is not None else {}))
        }
        
        self.epoch_metrics = {'mAP50':None, 'mAP50-95':None, 'fitness':None}
        self.consider = 'mAP50'
        self.clean_metrcis = self.eval_clean()
        self.stoper:AdvTrain_EarlyStopping = None
        
        self.metrics_writer = TrainingMetricsWriter(
            loss_order =  list(self.loss_dict.keys()),
            metrics_order= list(self.epoch_metrics.keys()),
            file=self.save_dir/f"log.csv",
            tb_dir=self.save_dir if tensorboard else None
        )
        
        self.args_settting = {
            'data':str(data),
            'imgsz': self.imgsz, 
            'batch':batch,
            'init_path':init_patch,
            'objective':self.objective,
            'pretrained_weight':str(self.detector),
            'psize':psize, 'ptype':ptype,
            'seed':seed, 'deterministics':deterministic,
            'prob_supervised':sup_prob_loss,
            'conf':conf
        }
    
    def get_train_val_loaders(self, model:YOLO) -> tuple[InfiniteDataLoader, InfiniteDataLoader|None]:
            
        data_args, dataset_args = get_data_args(
            model_args=model.overrides,
            stride=max(int(model.model.stride.max()), 32),
            dataset_cfgfile_path = self.data,
            batch=self.batchsz,
            mode='val'
        )

        return( 
            get_dataloader(
                data_args=data_args, 
                dataset_args=dataset_args, 
                split='train'
            ), 
            get_dataloader(
                data_args=data_args, 
                dataset_args=dataset_args,
                split='val'
            )
        )   

    def get_validator(self, model:YOLO)->AdvPatchAttack_YOLODetector_Validator|None:
        
        if self.valloader is not None:
            return AdvPatchAttack_YOLODetector_Validator( 
                reference_model=model, 
                attacker=self.attacker, 
                save_dir=self.save_dir, 
                conf=self.conf, 
                data=self.data,
                loader=self.valloader,
                rect=False
            ) 

    def _init_prob_loss(self, supervised=False)->V8Detection_MaxProb_Loss|Supervised_V8Detection_MaxProb_Loss:
        L = V8Detection_MaxProb_Loss if not supervised else Supervised_V8Detection_MaxProb_Loss
        return {'prob':L(model=self.model, to_attack=self.attack_cls, conf=self.conf)}
           
    def _init_other_loss(self, w_tv:float=2.5, printability_file=DEFAULT_PB_FILE, **kwargs):
        return { 
            'tv':TotalVariation(w=w_tv),
            'nps':NPSCalculator(patch_side=self.adv_patch.size(-1), printability_file=printability_file)
        }

    def cal_prob_loss(self, preds, batch)->dict[str, torch.Tensor]:
        l:torch.Tensor = self.loss_dict['prob'](preds=preds, batch=batch)
        if l.numel() > 1:
            match self.objective:
                case 'cls': 
                    l = l[0]
                case 'obj':
                    l = l[[1, 2]].sum()
                case 'obj-cls':
                    l = l.sum()
        return {'prob': l}
    
    def cal_other_loss(self, **kwargs)->dict[str, torch.Tensor]:
        
        return {
            'tv': self.loss_dict['tv'] (self.adv_patch),
            'nps': self.loss_dict['nps'](self.adv_patch)
        }
      
    def preprocess(self, batch, patch_random_rotate:bool=False, patch_blur:bool=False, debug:bool=False, **kwargs)-> dict[str, Any]:
        
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].float() / 255
        
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        
        return self.attacker.attack_yolo_batch(
            patch=self.adv_patch, batch=batch,  
            patch_random_rotate=patch_random_rotate,
            patch_blur=patch_blur,
            plot=debug
        )
    
    def model_forward(self, batch):
        return self.model(batch['img'])

    def train(self, lr:float=2e-2, epochs:int=100, patience:Optional[int]=None,method:Literal['gd', 'pgd']='gd', preprocess_args:dict[str, Any]=None):
        
        pr_args = preprocess_args if preprocess_args is not None else {}
        self.method = method
        print(f"write training args to {self.save_dir/'args.yaml'}")
        write_yaml({**self.args_settting, **pr_args, **{'lr':lr, 'epochs':epochs,'method':method}}, self.save_dir/f"train_args.yaml")
        
        self.stoper = AdvTrain_EarlyStopping(patience=patience, current_worst=self.clean_metrcis[self.consider])
        
        if self.method == 'gd':
            self.optimizer = optim.Adam([self.adv_patch], lr=lr, amsgrad=True)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , 'min', patience=50)
    
        loss_log = {k:0 for k in self.metrics_writer.loss_order}
        self.valloader.training = True
        
        for _e in range(epochs):
            e = _e+1
            pbar = tqdm(self.trainloader)
            pbar.set_postfix(ordered_dict={'epoch':f"{e}/{epochs}", "metrics":self.epoch_metrics[self.consider], 'patience':f"{_e - self.stoper.best_epoch}/{self.stoper.patience}"})
        
            for k in loss_log:
                loss_log[k] = 0
            
            self.adv_patch.requires_grad = True

            for batch in pbar:
                
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                
                batch = self.preprocess(batch=batch, **pr_args)
                preds = self.model_forward(batch=batch)
                
                tloss:dict[str, torch.Tensor] = {
                    **self.cal_prob_loss(preds=preds, batch=batch), 
                    **self.cal_other_loss(preds=preds, batch=batch)
                }
                total_loss = self._agg_loss(tloss=tloss)
                total_loss.backward()
                self.optimizer_step()

                for l in loss_log:
                    loss_log[l] += float(tloss[l].detach())
            
            loss_log = {k:loss_log[k]/len(self.trainloader) for k in loss_log}
            
            self.scheduler_step(loss_log=loss_log, epoch=e, iterations=e*len(self.trainloader))
            
            
            if self.valloader is not None:   
                self.adv_patch.requires_grad = False          
                self.epoch_metrics = self.valloader(model=self.model, adv_patch=self.adv_patch, **pr_args)


            self.metrics_writer(epoch=e, loss=loss_log, metrics=self.epoch_metrics)
            stop, update = self.stoper(epoch=e, fitness=self.epoch_metrics[self.consider])
            if update:
                print(f"New worst {self.consider}: {self.stoper.best_fitness}")
                self.save_model(name='worst', epoch=e)
            
            self.save_model(name='last', epoch=e) 
            if stop:
                break

        self.metrics_writer.close()

        return self.final_eval(preprocess_args=pr_args)
    
    def _agg_loss(self, tloss:dict[str, torch.Tensor], **kwargs)->torch.Tensor:
        return sum([v for _, v in tloss.items()])
    
    def eval_clean(self) -> dict[str, float]|None:
        if self.valloader is not None:
            print(f"evaluating clean data ..")
            M =  self.valloader(model=self.model)
            print(M)
            return M
        return None 
    
    def final_eval(self, preprocess_args:dict=None) -> dict[str, dict[str, float]]|None:
        
        to_test = self.save_dir/"worst.pt"
        
        if self.valloader is None:
            print("No validation set to evaluation, return None")
            return None
        
        if not to_test.is_file():
            print(f"attack fail! non of those epoch {self.consider} is worse than {self.worst}, return None")
            return None

        patch = torch.load(to_test, weights_only=False)['patch'].to(self.device)
        self.valloader.training = False
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

    def save_model(self, name:str, epoch:int):
    
        assert self.adv_patch.max() <= 1 and self.adv_patch.min() >= 0
    
        to_save = {'patch':self.adv_patch.detach().cpu(), 'epoch':epoch}
    
        if self.epoch_metrics['mAP50'] is not None:
            to_save = to_save|self.epoch_metrics
        
        torch.save(to_save, self.save_dir/f"{name}.pt")
        cv2.imwrite(self.save_dir/f"{name}.png", tensor2img(self.adv_patch))

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

