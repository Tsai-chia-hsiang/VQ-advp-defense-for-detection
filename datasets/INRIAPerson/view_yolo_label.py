import cv2
import sys
from pathlib import Path
import os
_FP_ = Path(__file__).parent
sys.path.append(os.path.abspath(_FP_.parent))
from deepcvext.box import scale_box, xywh2xyxy, xyxy2int
from deepcvext.draw import draw_boxes
if __name__ == "__main__":

    dataset = sys.argv[1].lower()
    target = sys.argv[2]
    
    img_root = Path("yolodata")/dataset/"images"
    label_root = Path("yolodata")/dataset/"labels"
    select = label_root/f"{target}.txt" 
    img = cv2.imread(img_root/f"{target}.png")

    ylabel = []
    cls = []
    
    with open(select, "r") as yl:
        for i in yl.readlines():
            ins = i.strip().split(" ")
            ylabel.append(list(map(lambda x:float(x),ins[1:])))
            cls.append(int(ins[0]))

    xyxy = scale_box(boxes=xywh2xyxy(ylabel), imgsize=list(img.shape[:2][::-1]), direction='back')
    xyxy = xyxy2int(xyxy=xyxy)
    print(xyxy)
    for xyxyi in xyxy:
        draw_boxes(img=img, xyxy=xyxyi)
    cv2.imwrite(f"{target}_see.png", img)
    

