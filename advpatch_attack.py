import os
import warnings
import logging
logging.basicConfig(level=logging.ERROR)
warnings.simplefilter("ignore")
import os
import json
import torch
from ultralytics import YOLO
from ultralytics_advpattack_lib.train import AdvPatchAttack_YOLODetector_Trainer
from ultralytics_advpattack_lib.validation import AdvPatchAttack_YOLODetector_Validator
from pathlib import Path
from argparse import ArgumentParser
from tools import write_json, args2dict, write_yaml, load_yaml
from ultralytics.utils.torch_utils import select_device, init_seeds
from ultralytics_advpattack_lib.attacker import DEFAULT_ATTACKER_CFG_FILE
from ultralytics_advpattack_lib.inference import attack
from ultralytics_advpattack_lib.ultralytics_utils import runtime_datacfg

def validataion(validator_args, patch_transform_args, **kwargs):
    
    init_seeds(validator_args['seed'], deterministic=validator_args['deterministic'])
    model = YOLO(validator_args['detector']).to(device=select_device(validator_args['device'], verbose=False))
    pre_attacked_data:Path = validator_args['save_dir']/"data.yaml"
    if pre_attacked_data.is_file():
        validator_args['data'] = pre_attacked_data
        validator_args['clean'] = True
        
    clean_only = validator_args['clean']
    adv_patch = None
    if not clean_only:
        adv_patch = torch.load(validator_args['pretrained_patch'], map_location=model.device)['patch']
    
    validator = AdvPatchAttack_YOLODetector_Validator(reference_model=model, **validator_args)
    metrics = validator.comparsion(model=model.model, adv_patch=adv_patch, **patch_transform_args)
    print(f"mAP50: {json.dumps(metrics['mAP50'],indent=4)}")
    write_json(
        {'data':str(validator_args['origin_data']), 'metrics':metrics}, 
        validator.save_dir/"val_metrics.json"
    )
    write_yaml(
        {**validator_args, **patch_transform_args},
        validator.save_dir/"args.yaml"
    )

def train_advpatch(trainer_args, patch_transform_args, hyp, **kwargs):
    advpatch_trainer = AdvPatchAttack_YOLODetector_Trainer(**trainer_args)
    train_result = advpatch_trainer.train(**hyp, preprocess_args=patch_transform_args)
    print(json.dumps(train_result, indent=4))

def inference(infer_args, patch_transform_args, **kwargs):
    init_seeds(infer_args['seed'], deterministic=infer_args['deterministic'])
    del infer_args['seed'], infer_args['deterministic']
    attack(**infer_args, patch_transform_args=patch_transform_args)

def lazy_arg_parsers():
    
    parser = ArgumentParser()

    parser.add_argument("--task", type=str, choices=['train','val', 'infer'], default='train')   
    
    cfg_keys = load_yaml("cmd_args.yaml")
    
    parser.add_argument("--detector", type=Path, default=Path("pretrained")/"yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--sup_prob_loss", action='store_true')
    parser.add_argument("--objective", type=str, default='cls', choices=['cls', 'obj', 'obj-cls'])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--data", type=Path, default=Path("datasets")/"INRIAPerson"/"inria.yaml")
    parser.add_argument("--project", type=Path, default=Path("exps"))
    parser.add_argument("--name", type=str, default="advp_attack")
    parser.add_argument("--psize", type=int, default=300)
    parser.add_argument("--ptype", type=str,choices=['random', 'gray'], default='random')
    parser.add_argument("--attacker", type=Path, default=DEFAULT_ATTACKER_CFG_FILE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--seed", type=int, default=891122)
    parser.add_argument("--not_deterministic", action='store_true')
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument("--pretrained_patch", type=Path, default=None)
    parser.add_argument("--patch_random_rotate", action='store_true')
    parser.add_argument("--patch_blur", action='store_true')
    parser.add_argument("--attack_cls", type=Path, default=Path("attack_cls.yaml"))
    parser.add_argument("--clean", action='store_true')

    
    # training hyp
    import math
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=math.inf)
    parser.add_argument("--method", type=str, default='gd')
  
    parser.add_argument("--debug", action='store_true')
    cli_args = parser.parse_args()
    cli_args.save_dir = cli_args.project/cli_args.name
    cli_args.origin_data = cli_args.data
    cli_args.data = runtime_datacfg(cli_args.data)

    print(f"create a runtime temp data cfg file: {cli_args.data}")

    del cli_args.project, cli_args.name
    
    if cli_args.pretrained_patch is None and cli_args.task != 'train':
        if cli_args.task == 'infer' or not cli_args.clean:
            potential_patch_path = cli_args.save_dir/f"worst.pt"
            if not potential_patch_path.is_file():
                potential_patch_path = cli_args.save_dir/f"last.pt"
            assert potential_patch_path.is_file()
            print(f"using {potential_patch_path} according `project` and `name` arguments to choose pretrained patch")
            cli_args.pretrained_patch = potential_patch_path

    cli_args = args2dict(cli_args)
    
    task = cli_args['task']
    args = {k: {j:cli_args[j] for j in v} for k,v in cfg_keys.items() }
    
    return task, args, cli_args


task_map = {
    'train':train_advpatch,
    'val':validataion,
    'infer':inference
}

def main():
    task, args, cli_args = lazy_arg_parsers()
    
    task_map[task](**args)
    if task in ['train', 'val']:
        if os.path.exists(cli_args['data']):
            print(f"removing runtime created temp data cfg file : {cli_args['data']}")
            os.remove(cli_args['data'])

if __name__ == "__main__":
    main()
    