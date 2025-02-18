import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal
from pathlib import Path

from tqdm import tqdm
import cv2
import torch
import torch.optim as optim
from torch.optim import Adam

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset, build_dataloader

from tools import set_seed, a_device
from tools import write_yaml, write_json
from advattack.ultralytics_utils import get_data_args, check_model_frozen
from advattack.attack import PatchAttacker, preprocess_yolo_batch_with_attack
from advattack.loss import v8DetLoss, TotalVariation, NPSCalculator
from advattack.validation import PatchAttack_DetValidator
from advattack.optimize import optimize_adv_img
from advattack.build import generate_patch
from advattack.writer import TrainingMetricsWriter
from torchcvext.convert import tensor2img

def save_patch(patch:torch.Tensor, save_dir:Path, filename:str, epoch:int, metrics:dict=None):
    assert patch.max() <= 1 and patch.min() >= 0
    to_save = {
        'patch':patch.detach().cpu(), 
        'epoch':epoch
    }
    if metrics is not None:
        to_save = to_save|metrics
    
    torch.save(to_save, save_dir/f"{filename}.pt")

    cv2.imwrite(save_dir/f"{filename}.png", tensor2img(patch))

def validataion(
    pretrained_weight:Path, patch_path:Path, 
    dataset_cfg:Path, proj:Path, seed:int, 
    device:torch.device=torch.device("cuda:0"), batchsz:int=16,
    random_rotation_patch=False, debug=False, **kwargs
):
    proj.mkdir(parents=True, exist_ok=True)
    model = YOLO(pretrained_weight)
    model = model.to(device=device)
    print(f"attacking with {patch_path}")
    patch = torch.load(patch_path, map_location='cpu')['patch']
    validator = PatchAttack_DetValidator.build_according_YOLO_with_dataset(
        model=model, dataset_cfg=dataset_cfg, 
        batchsz=batchsz, save_dir=proj, attacker=PatchAttacker()
    )
    metrics = validator(
        model=model.model,
        adv_patch=patch, 
        random_rotate_patch=random_rotation_patch,
        plot_attacked_img=debug
    )
    print(metrics)
    write_json(
        {'data':str(dataset_cfg), 'seed':seed, 'metrics':metrics},
        proj/"val_metrics.json"
    )

def adversarial_train(
    pretrained_weight:Path, dataset_cfg:Path, proj:Path, seed:int, 
    device:torch.device=torch.device('cuda', index=0),
    method:Literal['gd', 'pdg']='gd', lr:float=2e-2,
    epochs:int=1000, batchsz:int=16, psize:int=300, 
    det_loss_scale:bool=False, 
    random_rotation_patch:bool=False,
    tensorboard:bool=True, debug:bool=False, **kwargs
):
    if proj.is_dir():
        msg = f"{proj} already exist, If you don't want to overwrite it, stop the program (Ctrl+C), passing another proj direction and then run it again"
        print(msg)
        msg = "otherwise, it will just overwrite it"
        _ = input(msg)
        shutil.rmtree(proj)
    
    proj.mkdir(parents=True, exist_ok=False)
    
    write_yaml(
        o={
            'data':str(dataset_cfg),
            'pretrained_weight':str(pretrained_weight),
            'lr':lr,
            'epochs':epochs,
            "batchsz":batchsz,
            'psize':psize,
            "det_loss_scale":det_loss_scale,
            "random_rotation_patch":random_rotation_patch,
            'seed':seed
        }, 
        f=proj/"args.yaml"
    )
    
    set_seed(seed=seed)

    # YOLOv8 Detection Model
    model = YOLO(pretrained_weight)
    model = model.to(device=device)
    v8detection_model = model.model
    check_model_frozen(model=v8detection_model)

    # Loss functions : V8Detection loss, Variation loss, Non Printablitiy loss
    det_loss = v8DetLoss(model=v8detection_model, **{k:DEFAULT_CFG.get(k) for k in ['box','cls','dfl']})
    tv_loss = TotalVariation()
    nps_loss = NPSCalculator(patch_side=psize)
    
    # Trainable adversarial patch and patch applier (PatchAttacker)
    adv_patch = generate_patch(psize=psize, device=model.device)
    attacker = PatchAttacker()

    # Optimizer and scheduler
    patch_opt = Adam([adv_patch], lr=lr, amsgrad=True)
    patch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(patch_opt, 'min', patience=50)

    data_args, dataset_args = get_data_args(
        model_args = model.overrides, 
        stride = max(int(v8detection_model.stride.max()), 32),
        dataset_cfgfile_path = dataset_cfg,
        batch=batchsz
    )      

    trainloader = build_dataloader(
        dataset = build_yolo_dataset(
            cfg=data_args,  
            img_path=dataset_args.get('train'), 
            batch=data_args.batch, 
            data=dataset_args, 
            mode=data_args.mode, 
            stride = data_args.stride
        ),
        batch=data_args.batch, shuffle=True, 
        workers=data_args.workers
    )

    validator = PatchAttack_DetValidator.build_according_YOLO_with_dataset(
        model=model, dataset_cfg=dataset_cfg, 
        batchsz=batchsz, save_dir=proj, 
        attacker=attacker
    )
    
    clean_metrics = validator(
        model=v8detection_model,
        adv_patch=None, 
        random_rotate_patch=random_rotation_patch,
        plot_attacked_img=debug
    )
    worst_mAP50 = clean_metrics['mAP50']
    print(f"clean mAP50: {worst_mAP50}")

    metrics_writer = TrainingMetricsWriter(
        loss_order=['box_loss', 'cls_loss', 'dfl', 'tv_loss', 'nps_loss'], 
        metrics_order= ["mAP50", "mAP50-95", "fitness"],
        file=proj/f"log.csv",
        tb_dir=proj if tensorboard else None
    )

    for e in range(epochs):
        
        eloss = 0
        bar = tqdm(trainloader)
        bar.set_postfix(ordered_dict={'epoch':e})
        for batch in bar:
            
            if patch_opt is not None:
                patch_opt.zero_grad()

            batch = preprocess_yolo_batch_with_attack(
                attacker=attacker, 
                patch=adv_patch, 
                device=model.device,
                batch=batch,  
                random_rotation_patch=random_rotation_patch,
                plot=debug
            )
            
            bsz=len(batch["img"])
            
            preds = v8detection_model(batch['img'])
            det = det_loss(preds=preds, batch=batch)
            
            if det_loss_scale:
                det *= bsz

            if method == "gd":
                det *= -1

            tv:torch.Tensor = tv_loss(adv_patch)
            nps:torch.Tensor = nps_loss(adv_patch)
            tloss = det.sum() + tv + nps
            optimize_adv_img(img=adv_patch, method=method, loss=tloss, opt=patch_opt, )
            
            det_loss_log = det.detach() if not det_loss_scale else det.detach()/bsz

            loss_log = torch.cat([det_loss_log, tv.detach().unsqueeze(0), nps.detach().unsqueeze(0)])
            eloss += loss_log
        
        eloss /= len(trainloader)
        patch_scheduler.step(eloss.sum()) 
        
        eloss = {metrics_writer.loss_order[j]: float(eloss[j]) for j in range(len(eloss))}
        eloss['box_loss'] *= -1
        eloss['cls_loss'] *= -1
        eloss['dfl'] *= -1
        
        adv_patch.requires_grad = False

        metrics = validator(
            model=v8detection_model,
            adv_patch=adv_patch, 
            random_rotate_patch=random_rotation_patch,
            plot_attacked_img=debug
        )
        
        metrics_writer(epoch=e, loss=eloss, metrics=metrics)

        if metrics['mAP50'] <= worst_mAP50:
            print(f"Get worse adversaral patch that makes mAP from {worst_mAP50} to {metrics['mAP50']} at {e}")
            save_patch(patch=adv_patch, save_dir=proj, metrics=metrics, filename='worst', epoch=e)
            worst_mAP50 = metrics['mAP50']
        
        save_patch(patch=adv_patch, save_dir=proj, filename='last', metrics=metrics, epoch=e)
        adv_patch.requires_grad=True

    metrics_writer.close()
    
    worst_patch = torch.load(proj/"worst.pt", weights_only=False)['patch']
    final_metrics = validator(model=v8detection_model, adv_patch=worst_patch, random_rotate_patch=random_rotation_patch)
    
    cmp_metrics = {
        k:{
            'clean':clean_metrics[k],
            'attack':final_metrics[k]
        } 
        for k in clean_metrics
    }
    print(cmp_metrics)
    write_json(cmp_metrics, proj/"final_val.json")
    # print(model.val(data=dataset_cfg).result_dict)
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--task", type=str, default='train')
    parser.add_argument("--seed", type=int, default=891122)
    parser.add_argument("--proj", type=Path)
    parser.add_argument("--method", type=str, default='gd')
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--pretrained_weight", type=Path, default=Path("pretrained")/"yolov8n.pt")
    parser.add_argument("--dataset_cfg", type=Path, default=Path("INRIAPerson")/"inria.yaml")
    parser.add_argument("--batchsz", type=int, default=16)
    parser.add_argument("--det_loss_scale", action='store_true')
    parser.add_argument("--psize", type=int, default=300)
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument("--random_rotation_patch", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--device", type=int, default=0)

    cli_args = vars(parser.parse_args()) 
    cli_args["tensorboard"] = not cli_args["no_tensorboard"]
    set_seed(cli_args["seed"])
    cli_args["device"] = a_device(cli_args["device"])
    
    match cli_args["task"]:
        case 'train':
            adversarial_train(**cli_args)
        case 'val':
            validataion(**cli_args)
        case _:
            raise KeyError(f"No such a task {cli_args['task']}.")

    