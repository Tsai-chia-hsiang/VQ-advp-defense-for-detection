from pathlib import Path
import cv2
from deepcvext.draw import draw_boxes
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np

def main():
    sample = "crop001504"
    to_size = None
    cmap = np.loadtxt("colormap.csv").tolist()
    detector = YOLO("pretrained/yolov8n.pt").to(device=1)
    dst = Path("visualization/masks/")
    names = ["pixel_mask_det.png","token_mask_det.png"]
    imgs = [
        cv2.imread(
            Path(
                "exps/advyolo_obj/vq_img_att/val/images"
            )/f"{sample}.png"
        ),
        cv2.imread(
            Path(
                "exps/advyolo_obj/vq_tmask_recons/val/images"
            )/f"{sample}.png"
        )
    ]
   
    res:list[Results] = detector(imgs, verbose=False, conf=0.25, classes=[0], imgsz=640)
    for i, r in enumerate(res):
        boxes = r.cpu().numpy().boxes
        draw_boxes(
            imgs[i], xyxy=boxes.xyxy, color=cmap[:len(boxes)], 
            box_thickness=10
        )
        save_to = dst/sample/names[i]
        print(save_to)
        cv2.imwrite(save_to, imgs[i] if to_size is None else cv2.resize(imgs[i], to_size))

if __name__ == "__main__":
    main()