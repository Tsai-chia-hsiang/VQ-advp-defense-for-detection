import json
import torch
from ultralytics import YOLO
from advattack.attacker import PatchAttacker
from advattack.train import AdvPatchAttack_YOLODetector_Trainer
from advattack.validation import PatchAttack_DetValidator
from pathlib import Path
from argparse import ArgumentParser
from tools import write_json, args2dict, load_yaml
from ultralytics.utils.torch_utils import select_device, init_seeds
from advattack.attacker import DEFAULT_ATTACKER_CFG_FILE
from advattack.inference import compare_visualization


def validataion(
    data:Path, detector:Path, pretrained_patch:Path, 
    project:Path, name:str,  conf:float=0.25, 
    batch:int=16,  attacker=DEFAULT_ATTACKER_CFG_FILE,
    seed:int=89112, deterministic:bool=True, device:str='0',
    patch_random_rotate:bool=False, patch_blur:bool=False,
    debug:bool=False, clean=False, vq:bool=False,
    **kwargs
):
    
    init_seeds(seed=seed, deterministic=deterministic)
    device = select_device(device=device, verbose=False)

    (project/name).mkdir(parents=True, exist_ok=True)
    model = YOLO(detector)
    model = model.to(device=device)
    print(f"attacking with {pretrained_patch}")
    patch = torch.load(pretrained_patch, map_location=model.device)['patch']
    validator = PatchAttack_DetValidator.build_according_YOLO_with_dataset(
        model=model, data=data, 
        batch=batch, save_dir=project/name, 
        conf=conf,
        attacker=PatchAttacker(**load_yaml(attacker))
    )
    metrics = validator.comparsion(
        model=model.model,
        adv_patch=patch if not clean else None,
        vq=vq, 
        patch_random_rotate=patch_random_rotate,
        patch_blur=patch_blur,
        debug=debug
    )
    print(json.dumps(metrics,indent=4))
    write_json({'data':str(data), 'seed':seed, 'metrics':metrics},
        project/name/"val_metrics.json"
    )

def train_advpatch(trainer_args, patch_transform_args, hyp, **kwargs):
    advpatch_trainer = AdvPatchAttack_YOLODetector_Trainer(**trainer_args)
    train_result = advpatch_trainer.train(**hyp, preprocess_args=patch_transform_args)
    print(json.dumps(train_result, indent=4))

def lazy_arg_parsers():
    
    parser = ArgumentParser()

    parser.add_argument("--task", type=str, choices=['train','val', 'infer'], default='train')   
    parser.add_argument("--patch_path", type=Path)

    # patch transform

    cfg_keys = {
        "patch_transform_args":["patch_random_rotate", "patch_blur"],
        "trainer_args":[
            "detector", "data","to_attack", "logit_to_prob", "conf",
            "project", "name", "psize", "ptype", "attacker", "batch", "device",
            "sup_prob_loss", 
            "seed", "deterministic", "tensorboard"
        ],
        "validator_args":[
            "detector", "pretrained_patch", 
            "data","project", "attacker", "batch",
            "name", "seed", "deterministic", "device",
            "conf", "clean", "vq"
        ],
        "hyp":["lr", "epochs", "patience"]
    }
    
    
    parser.add_argument("--detector", type=Path, default=Path("pretrained")/"yolov8n.pt")
    parser.add_argument("--sup_prob_loss", action='store_true')
    parser.add_argument("--logit_to_prob", action='store_true')
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--data", type=Path, default=Path("INRIAPerson")/"inria.yaml")
    parser.add_argument("--project", type=Path, default=Path("adv_patch"))
    parser.add_argument("--name", type=str, default="advp_attack")
    parser.add_argument("--psize", type=int, default=300)
    parser.add_argument("--ptype", type=str,choices=['random', 'gray'], default='random')
    parser.add_argument("--attacker", type=Path, default=DEFAULT_ATTACKER_CFG_FILE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--seed", type=int, default=891122)
    parser.add_argument("--not_deterministic", action='store_true')
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument("--pretrained_patch", type=Path)
    parser.add_argument("--patch_random_rotate", action='store_true')
    parser.add_argument("--patch_blur", action='store_true')
    parser.add_argument("--to_attack", type=Path, default=Path("attack_cls.yaml"))
    parser.add_argument("--clean", action='store_true')
    parser.add_argument("--vq", action='store_true')
    
    # training hyp
    import math
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=math.inf)
    #parser.add_argument("--det_loss_scale", action='store_true')
    parser.add_argument("--method", type=str, default='gd')
    # parser.add_argument("--det_terms", nargs='+', type=str, default=['box', 'prob'])
    
    parser.add_argument("--debug", action='store_true')
    
    cli_args = args2dict(parser.parse_args())
    task = cli_args['task']
    args = {k: {j:cli_args[j] for j in v} for k,v in cfg_keys.items() }
    
    if task in ['val', 'infer']:
        # TODO: refactoring validator so that it can be like 
        # trainer directly takes hierarchical dict
        args = {**(args['validator_args']), **(args['patch_transform_args'])}
        if args['pretrained_patch'] is None:
            potential_patch_path = args['project']/args['name']/f"worst.pt"
            if not potential_patch_path.is_file():
                potential_patch_path = args['project']/args['name']/f"last.pt"
            
            assert potential_patch_path.is_file()
            print(f"using {potential_patch_path} according `project` and `name` arguments to choose pretrained patch")
            args['pretrained_patch'] = potential_patch_path
    
    return task, args


task_map = {
    'train':train_advpatch,
    'val':validataion,
    'infer':compare_visualization
}

def main():
    task, args = lazy_arg_parsers()
    task_map[task](**args)

if __name__ == "__main__":
    main()
    