from pathlib import Path
from typing import Iterable

_EXP_DIR_ = Path(__file__).parent

def advyolo_attacked_set(
    proj_prefix:str, atts:Iterable[str],
    root:Path=_EXP_DIR_, advyolo_prefix:str='advyolo',
    clean_proj:Path = None 
)->dict[str, dict[str, Path]]:
    
    attacked_sets = {}
    
    if clean_proj is not None:
        img_dir = clean_proj/"val"/"images" 
        if img_dir.is_dir():
            attacked_sets['clean'] = {
                'img_dir':img_dir,
                'proj': clean_proj.parent/proj_prefix
            }
        
    for target in atts:
        img_dir = root/f"{advyolo_prefix}_{target}"/"attacked"/"val"/"images"
        mask_dir = root/f"{advyolo_prefix}_{target}"/"attacked"/"val_mask"
        if img_dir.is_dir():
            proj_dir = root/f"{advyolo_prefix}_{target}"/proj_prefix
            attacked_sets[target] = {
                'img_dir':img_dir, 
                'proj':proj_dir,
                'mask_dir':mask_dir
            }

    return attacked_sets