from ultralytics_advpattack_lib.dummuy_validation import yolo_val
from numpy.random import Generator as nprng
from typing import Optional
from exps import advyolo_attacked_set
from deepcvext.seg import cv2_read_binary_map
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from vq import MOVQ
import torch
from deepcvext import tensor2img
from deepcvext.img_normalize import _IMG_DESTD_
from tools.path_utils import refresh_dir
from datasets.INRIAPerson import INRIA_DATA_CFG
from tools import write_json
from pytorch_fid import fid_score
from PAD.fuse_filter import heatmap_filter


def binary_mask_with_ratio(mask:np.ndarray, p:float, rng:Optional[nprng]=None)->np.ndarray:
    keep = None
    if rng is not None:
        keep = rng.random(mask.shape) <= p
    else:
        keep = np.random.rand(*(mask.shape)) <= p
     
    return (mask & keep)

@torch.no_grad()
def fragment_gt_token_mask_experiment():

    mask_rate = [0.3, 0.5, 0.7, 1.0]
    att_sets = advyolo_attacked_set(proj_prefix='ratio')
    dev = torch.device('cuda', 0)
    model = MOVQ.from_cfg(on_device=dev)
    result_dict = {}

    for target, pathes in att_sets.items():

        print(target, pathes['img_dir'])
        
        rng = np.random.default_rng(891122)
        ims = [_ for _ in pathes['img_dir'].glob("*.png")]
        gt_masks = [pathes['mask_dir']/_.stem/"union.png" for _ in ims]
        
        result_dict[target] = []

        im_save_dir = refresh_dir(pathes['proj']/"val"/"images", direct_remove=True)
        
        for p in mask_rate:
            print(p)
            for im, gt_mask_path in tqdm(zip(ims, gt_masks), total=len(ims)):
                i0 = cv2.imread(im)
                recons, diff, code_idxs, (top, bottom, left, right) = model.recons_cv2(
                    img=i0
                )    
                gt_mask = cv2_read_binary_map(gt_mask_path, target_hw=code_idxs.shape[1:])
                m = binary_mask_with_ratio(mask=gt_mask, p=p, rng=rng)
                defened = model.decode_code(code_b=code_idxs, mask=torch.from_numpy(1-m).squeeze().to(model.device))
                defened = tensor2img(timg=defened, scale_back_f=_IMG_DESTD_)
                cv2.imwrite(im_save_dir/im.name, defened)
            
            r = yolo_val(proj=pathes['proj'], default_data=INRIA_DATA_CFG, dev=dev)
            print(r['metrics/mAP50(B)'])
            result_dict[target].append(
                {
                    'p':p,
                    'mAP50': r['metrics/mAP50(B)']
                }
            )
        
    write_json(result_dict, Path("defense_eval")/"ratio.json")
    

def fragmented_gt_pixel_mask_experiment():
    
    att_sets = advyolo_attacked_set(proj_prefix='gt')
    result_dict = {}
    mask_rate = [0.3, 0.5, 0.7, 1.0]
    
    for target, pathes in att_sets.items():

        ims = [_ for _ in pathes['img_dir'].glob("*.png")]
        gt_masks = [pathes['mask_dir']/_.stem/"union.png" for _ in ims]
        im_save_dir = refresh_dir(pathes['proj']/"val"/"images", direct_remove=True)
        rng = np.random.default_rng(891122)
        result_dict[target] = []
        
        print(target)
        
        for p in mask_rate:    
            print(p)

            for im, gt_mask in tqdm(zip(ims, gt_masks), total=len(ims)):
                i0 = cv2.imread(im)
                gt_mask = cv2_read_binary_map(gt_mask, dtype=np.uint8)
                m = binary_mask_with_ratio(mask=gt_mask, p=p, rng=rng).astype(np.uint8)
                cv2.imwrite(im_save_dir/im.name, i0*(1-m))
            
            r = yolo_val(proj=pathes['proj'], default_data=INRIA_DATA_CFG)

            print(r['metrics/mAP50(B)'])
            
            result_dict[target].append(
                {    
                    'p':p,
                    'mAP50': r['metrics/mAP50(B)']
                }
            )
    
    
    write_json(result_dict, Path("defense_eval")/"gt_ratio_mask.json")


def FID(att_patch:str, img_path_pattern:Path=Path("val")/"images"):
    proj_root = Path("exps")/f"advyolo_{att_patch}"
    real = str(Path("/datasets/INRIAPerson/val/images"))
    aimg = str(proj_root/"attacked"/img_path_pattern)
    gimg = str(proj_root/"vq_tmask_recons"/img_path_pattern)
    mimg = str(proj_root/"cdfull"/img_path_pattern)

    dev = torch.device('cuda', 1)
    fid_value_a = fid_score.calculate_fid_given_paths(
        [real, aimg], 
        batch_size=1, dims=2048, device=dev
    )
    print(fid_value_a)
    
    
    fid_value_m = fid_score.calculate_fid_given_paths(
        [real, mimg],
        batch_size=1, dims=2048, device=dev
    )
    print(fid_value_m)

    fid_value_g = fid_score.calculate_fid_given_paths(
        [real, gimg],
        batch_size=1, dims=2048, device=dev
    )
    print(fid_value_g)


if __name__ == "__main__":
    # token_mask_fragment_experiment()
    # fragmented_gt_pixel_mask_experiment()
    FID(att_patch='obj')