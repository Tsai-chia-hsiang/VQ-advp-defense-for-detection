# VQ for adversarial patch defense in YOLO series detection models

## Setup


## 1. Generate patch to perform adversarial patch attack:
Attack methods:
- [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf) (CVPRW2019):
    - reference code : [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git)
        - We do not simply run train_patch.py since it is primarily designed for YOLOv2 but not for V8. Instead, we integrate their patch appling method into [advattack/attack/PatchAttacker()](./advattack/attacker.py).
  
    - |Detector|Attack dataset|Command| Validation|
      |-|-|-|-|
      |YOLOv8n, pretrained on COCO, conduct by Ultrayltics|INRIA Person|sh scripts/advyolo.sh train|sh scripts/advyolo.sh val|