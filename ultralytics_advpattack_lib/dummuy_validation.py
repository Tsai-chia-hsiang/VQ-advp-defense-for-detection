from ultralytics import YOLO
import torch
from ultralytics.utils.metrics import DetMetrics
import shutil
from pathlib import Path
import numpy as np
from .ultralytics_utils import clone_ultralytics_proj

def yolo_val( 
    proj:Path, default_data:Path = None,
    dev:torch.device|str|int=torch.device('cuda', 0), 
    conf=0.25, pretrained:Path=Path("pretrained")/"yolov8n.pt"
)->dict[str, np.ndarray]:
    
    create_labels_link = False
    temp_data = False
    data = proj/"data.yaml"
    if not data.exists():
        (data, temp_data) ,(label_dir, create_labels_link), (_, _) = clone_ultralytics_proj(
            proj=proj, default_data=default_data
        )

    model = YOLO(pretrained).to(dev)
    r:DetMetrics = model.val(data=data, verbose=False, conf=conf)
    
    shutil.rmtree("runs")
    
    if create_labels_link:
        cached = label_dir.parent/"labels.cache"
        cached.unlink()
        label_dir.unlink()
    
    if temp_data:
        data.unlink()
        
    return r.results_dict


