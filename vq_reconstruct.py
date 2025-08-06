from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from tools.path_utils import refresh_dir
from tools.mask_utils import dbscan_mask, Diff2HeatmapProcessor
from tools import write_json
from PAD.fuse_filter import heatmap_filter
from vq import MOVQ
from ultralytics_advpattack_lib.dummuy_validation import yolo_val
from deepcvext.convert import tensor2img
from deepcvext.img_normalize import _IMG_DESTD_
from deepcvext.utils import cvtcolor
from deepcvext.draw import canvas
from datasets.INRIAPerson import INRIA_DATA_CFG
from exps import advyolo_attacked_set


@torch.no_grad()
def tokenspace_masking_and_decode(
    proj:Path, img_dir:Path, model:MOVQ, 
    heatmap_processor:Diff2HeatmapProcessor, 
    pad_morph:bool=False
):
    
    recons_img_dir = refresh_dir(proj/"val"/"images",  direct_remove=True)
    token_mask_dir = refresh_dir(proj/"tokenspace_mask", direct_remove=True)
    heatmap_dir = refresh_dir(proj/"heatmap", direct_remove=True)

    imgs = [_ for _ in img_dir.iterdir()]
    for i in tqdm(imgs):
        
        i0:np.ndarray = cv2.imread(i)
        
        recons, diff, code_idxs, (top, bottom, left, right) = model.recons_cv2(img=i0)
        heatmap, tmask = heatmap_processor.generate_binary_mask_from_diff(
            diff=diff, thr=80, 
            thr_method='percentile' if pad_morph else 'adp', 
            return_framework='numpy' if pad_morph else 'torch'
        )
        if pad_morph:
            h_t, h_t_o, h_t_o_c, heatmap_post, tmask = heatmap_filter(
                heatmap=heatmap, threshold=None, 
                kernel_param=80//model.encode_scaledown,
                height=heatmap.shape[0], 
                width=heatmap.shape[1]
            )
            tmask = torch.from_numpy(tmask).to(device=model.device)
        
        defened_img = model.decode_code(code_b=code_idxs, mask=1-tmask)
        defened_img = tensor2img(timg=defened_img, scale_back_f=_IMG_DESTD_)[top:bottom, left:right]
        
        assert defened_img.shape == i0.shape

        vis_heatmap = heatmap
        vis_mask =  cvtcolor(tmask.cpu().numpy()*255, 3)
        if pad_morph:
            vis_heatmap = canvas(
                [cvtcolor(heatmap, 3), cvtcolor(heatmap_post, 3)], 
                bar_color=(114,114,114)
            )
        else:
            vis_heatmap = cvtcolor(heatmap.cpu().numpy()*255)
            
        cv2.imwrite(heatmap_dir/i.name, vis_heatmap)
        cv2.imwrite(token_mask_dir/i.name, vis_mask)
        cv2.imwrite(recons_img_dir/i.name, defened_img)

@torch.no_grad()
def pixelspace_masking(
    proj:Path, img_dir:Path, model:MOVQ, 
    heatmap_processor:Diff2HeatmapProcessor,
    pad_morph:bool=False
):
    
    masked_img_dir = refresh_dir(proj/"val"/"images",  direct_remove=True)
    pixelspace_mask_dir = refresh_dir(proj/"pixelspace_mask", direct_remove=True)
    heatmap_dir = refresh_dir(proj/"heatmap", direct_remove=True)

    imgs = [_ for _ in img_dir.iterdir()]
    
    for i in tqdm(imgs):
        i0:np.ndarray = cv2.imread(i)
        recons, diff, _, (top, bottom, left, right) = model.recons_cv2(img=i0)

        diff = diff[:, :, top:bottom, left:right]
        heatmap, pmask = heatmap_processor.generate_binary_mask_from_diff(
            diff=diff, thr=80, thr_method='percentile'
        )
        if pad_morph:
            h_t, h_t_o, h_t_o_c, heatmap_post, pmask = heatmap_filter(
                heatmap=heatmap, threshold=None, 
                kernel_param=80,
                height=heatmap.shape[0], 
                width=heatmap.shape[1]
            )
            heatmap = canvas([cvtcolor(heatmap,3), cvtcolor(heatmap_post,3)], bar_color=(144,144,144))
        
        cv2.imwrite(heatmap_dir/i.name, heatmap)
        pmask = np.where(pmask > 0, 1, 0).astype(np.uint8)
        defened = i0*(1-np.expand_dims(pmask, -1))
        cv2.imwrite(masked_img_dir/i.name, defened)
        cv2.imwrite(pixelspace_mask_dir/i.name, cvtcolor(pmask*255))        


def tmasking_and_decode_advyolo(
    clean_dir:Path=Path("./datasets")/"INRIAPerson"/"yolodata",    
    proj_prefix:str="vq_tmask_recons", device:int=0,
    smooth_k:int=7, pad_morph:bool=False
):

    attacked_set = advyolo_attacked_set(
        proj_prefix=proj_prefix,
        clean_proj=clean_dir
    ) 
    result_dict = {}
    dev = torch.device('cuda', device)
    model = MOVQ.from_cfg(on_device=dev)

    heatmap_processor = Diff2HeatmapProcessor(
        smooth_k=smooth_k, 
        scale_down=model.encode_scaledown, 
        default_order=('smooth', 'channel_reduce', 'normalize', 'scale', 'normalize'),
        default_return='torch'
    )

    for att_set, pathes in attacked_set.items():
        proj = pathes['proj']
        proj = proj.parent/(proj.name+f"_k{smooth_k}")
        if pad_morph:
            proj = proj.parent/(proj.name+f"_morph")

        #_ = input
        print(f"{att_set}: {pathes['img_dir']} defense -> {proj}")
        
        tokenspace_masking_and_decode(
            img_dir=pathes['img_dir'], 
            proj=proj,
            model=model, 
            heatmap_processor=heatmap_processor,
            pad_morph=pad_morph
        )
        
        r = yolo_val(proj=proj, default_data=INRIA_DATA_CFG, dev=dev)
        print(r['metrics/mAP50(B)'])
        result_dict[att_set] = r
    
    report_name = f"{proj_prefix}_k{smooth_k}"
    if pad_morph:
        report_name=report_name + "_morph"
    write_json(result_dict, Path("defense_eval")/f"{report_name}.json")


def pmasking_advyolo(
    clean_dir:Path=Path("./datasets")/"INRIAPerson"/"yolodata",    
    proj_prefix:str="vq_pmask_recons_s", device:int=1
):

    attacked_set = advyolo_attacked_set(
        proj_prefix=proj_prefix,
        clean_proj=clean_dir
    ) 
    result_dict = {}
    dev = torch.device('cuda', device)
    model = MOVQ.from_cfg(on_device=dev)

    heatmap_processor = Diff2HeatmapProcessor(
        smooth_k=13,
        scale_down=8, 
        default_order=('smooth', 'channel_reduce', 'normalize', 'scale', 'scale_back', 'normalize'),
        default_return='numpy'
    )

    for att_set, pathes in attacked_set.items():
        print(f"{att_set}: {pathes['img_dir']} defense -> {pathes['proj']}")
        pixelspace_masking(
            img_dir=pathes['img_dir'], proj=pathes['proj'],
            model=model, heatmap_processor=heatmap_processor
        )
        r = yolo_val(proj=pathes['proj'], default_data=INRIA_DATA_CFG, dev=dev)
        print(r['metrics/mAP50(B)'])
        result_dict[att_set] = r
    
    write_json(result_dict, Path("defense_eval")/f"{proj_prefix}.json")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--setting", choices=['token','pixel'], type=str, default='token')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clean_dir", type=Path, default=Path("./datasets")/"INRIAPerson"/"yolodata")
    parser.add_argument("--smooth_k", type=int, default=7)
    parser.add_argument("--morph", action='store_true')
    args = parser.parse_args()

    match args.setting:
        case 'token':
            tmasking_and_decode_advyolo(
                clean_dir=args.clean_dir,
                device=args.device,
                smooth_k= args.smooth_k,
                pad_morph=args.morph
            )
        case 'pixel':
            pmasking_advyolo(
                clean_dir=args.clean_dir,
                device=args.device
            )