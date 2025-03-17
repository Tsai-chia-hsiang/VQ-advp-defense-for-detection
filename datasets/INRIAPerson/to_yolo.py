import yaml
from pathlib import Path
import os
import json
import numpy as np
import cv2
import math

def load_yaml(yf):
    with open(yf, "r") as fp:
        c = yaml.safe_load(fp)
    return c

def swap_posfix(x:str|Path|os.PathLike, postfix:str) -> Path:
    return f"{Path(x).stem}.{postfix}"

def draw_box(img:np.ndarray, xyxy:list)->None:
    
    int_xyxy = [
        math.floor(xyxy[0]), math.floor(xyxy[1]),
        math.ceil(xyxy[2]), math.ceil(xyxy[3])
    ]
    
    #int_xyxy = list(map(lambda x:int(x), xyxy))
    cv2.rectangle(img, int_xyxy[:2], int_xyxy[2:], color=(0,0,255), thickness=1)


def xyxy2xywh(xyxy:list[list[float|int]], normalize_wh:list[int]=None) -> list[list[float]]:
    xyxy_arr = np.asarray(xyxy, dtype=np.float32)
    xywh_arr = np.zeros_like(xyxy_arr)
    xywh_arr[:, 0] = (xyxy_arr[:, 0] + xyxy_arr[:, 2])/2 #cx
    xywh_arr[:, 1] = (xyxy_arr[:, 1] + xyxy_arr[:, 3])/2 #cy
    xywh_arr[:, 2] = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) #w
    xywh_arr[:, 3] = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) #h
    if normalize_wh is not None:
        xywh_arr /= np.tile(np.asarray(normalize_wh), 2)
    return xywh_arr.tolist()

def extract_inria_label(label_file:Path) -> tuple[list[int, int], list[list[float]]]:
    labels = []
    imgsize = [None, None]
    offset = (0, 0)
    with open(label_file, "r", encoding='latin-1') as lf:
        for _ in lf.readlines():
            i = _.strip()
            if ":" not in i:
                continue
            content = i.split(":")
            content[1] = content[1].replace(" ",'')
            if 'Image size' in i:
                content[1] = content[1].split("x")
                imgsize[0] =  int(content[1][0])
                imgsize[1] =  int(content[1][1])
            elif 'Top left pixel' in i or 'Xmax' in i:
                content[1] = content[1].replace('(', '')
                content[1] = content[1].replace(')', '')
                content[1] = content[1].replace('-', ',')
                content[1] = content[1].split(",")
                if 'Top left pixel' in i:
                    offset = (float(content[1][0]), float(content[1][1]))
                elif 'Xmax' in i:
                    labels.append([float(li)+offset[j%2] for j, li in enumerate(content[1])])
    return labels, imgsize

def inria_img_label_pairs(root:Path = Path("."), mode:str="Test") -> list[tuple[Path, list[float]]]:
    
    r = root/mode
    annlist = r/"annotations.lst"
    with open(annlist, "r") as f:
        labeled_imgs = [i.strip() for i in f.readlines()]
    img_root = r/"pos"
    img_label_pair = [
        ((img_root/swap_posfix(i, "png")).absolute(), 
         extract_inria_label((root/i).absolute()))
         for i in labeled_imgs
    ]
    return img_label_pair


def main():
    imgs = inria_img_label_pairs(mode="Train")
    yolo_label_root = Path("Train")/"yolo_label"
    yolo_label_root.mkdir(parents=True, exist_ok=True)
    img_size = {}
    for p in imgs:
        img_name = p[0].stem
        img_size[p[0].name] = p[1][1] 
        label_file_name = f"{img_name}.txt"
        with open(yolo_label_root/label_file_name, "w+") as lf:
            coos = xyxy2xywh(xyxy=p[1][0], normalize_wh=p[1][1])
            for c in coos:
                lf.write(f"0 " + " ".join(list(map(lambda x:str(x), c)))+'\n')
        
if __name__ == "__main__":
    main()
