import torch
from ultralytics import YOLO
from advattack.attack import PatchAttacker
from advattack.train import AdvPatchAttack_YOLODetector_Trainer
from advattack.validation import PatchAttack_DetValidator
from pathlib import Path
from typing import Literal
from argparse import ArgumentParser
from tools import write_yaml, write_json

def validataion(
    pretrained_weight:Path, patch_path:Path, 
    dataset_cfg:Path, proj:Path, seed:int, 
    device:torch.device=torch.device("cuda:0"), batchsz:int=16,
    random_rotate_patch=False, debug=False, **kwargs
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
    metrics = validator.comparsion(
        model=model.model,
        adv_patch=patch, 
        random_rotate_patch=random_rotate_patch,
        plot_attacked_img=debug
    )
    print(metrics)
    write_json(
        {'data':str(dataset_cfg), 'seed':seed, 'metrics':metrics},
        proj/"val_metrics.json"
    )

def train_advpatch(
    pretrained_weight:Path, dataset_cfg:Path, project:Path, seed:int,  
    device:str='0',
    method:Literal['gd', 'pdg']='gd', lr:float=2e-2,
    epochs:int=1000, patience:int=None, 
    batchsz:int=16, psize:int=300, 
    scale_det_loss:bool=False, 
    random_rotate_patch:bool=False,
    debug:bool=False, **kwargs
):

    advpatch_trainer = AdvPatchAttack_YOLODetector_Trainer(
        pretrained_weight=pretrained_weight, 
        dataset_cfg=dataset_cfg,
        seed=seed,
        device=device,
        save_dir=project, 
        scale_det_loss=scale_det_loss,
        batchsz=batchsz, psize=psize
    )
    write_yaml(
        o={
            'data':str(dataset_cfg),
            'pretrained_weight':str(pretrained_weight),
            'lr':lr,
            'epochs':epochs,
            "batchsz":batchsz,
            'psize':psize,
            "det_loss_scale":scale_det_loss,
            "random_rotate_patch":random_rotate_patch,
            'seed':seed,
            'patience':patience
        }, 
        f=project/"args.yaml"
    )
    train_result = advpatch_trainer.train(
        method=method, 
        lr=lr,
        epoch=epochs,
        patience=patience,
        preprocess_args={
            'random_rotate_patch':random_rotate_patch, 
            'debug':debug
        }
    )
    print(train_result)


def main():
    pass

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--task", type=str, default='train')
    parser.add_argument("--seed", type=int, default=891122)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--method", type=str, default='gd')
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--pretrained_weight", type=Path, default=Path("pretrained")/"yolov8n.pt")
    parser.add_argument("--patch_path", type=Path)
    parser.add_argument("--dataset_cfg", type=Path, default=Path("INRIAPerson")/"inria.yaml")
    parser.add_argument("--batchsz", type=int, default=16)
    parser.add_argument("--det_loss_scale", action='store_true')
    parser.add_argument("--psize", type=int, default=300)
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument("--not_rotate", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--device", type=str, default='0')

    cli_args = vars(parser.parse_args()) 
    cli_args["tensorboard"] = not cli_args["no_tensorboard"]
    cli_args["random_rotate_patch"] = not cli_args["not_rotate"]

    match cli_args["task"]:
        case 'train':
            train_advpatch(**cli_args)
        case 'val':
            pass
            validataion(**cli_args)
        case _:
            raise KeyError(f"No such a task {cli_args['task']}.")
