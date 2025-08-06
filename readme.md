# VQ for adversarial patch defense in YOLO series detection models

## Setup
1. Create a Python3.13 environment
2. Install torch with cuda 12.8
3. pip install -r requirements.txt
4. git clone https://github.com/Tsai-chia-hsiang/deepcvext.git

## Adversarial Path generation & attacking:
Attack methods:
- [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf) (CVPRW2019):
    - reference code : [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git)
        
        - We do not simply run `train_patch.py` since it is primarily designed for YOLOv2 but not for v8. 
        
            Instead, we integrate their patch appling method into [advattack/attack/PatchAttacker()](./advattack/attacker.py).
        
        - We use the patches from [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git) as initial patches and finetune them to attack YOLOv8n
  
    - |Detector|Attack dataset|Generate Adversarial Patch|Performing Attack
      |-|-|-|-|
      |YOLOv8n, pretrained on COCO, conduct by Ultrayltics|INRIA Person|`sh scripts/advyolo_{$PATCH_TYPE}_v8.sh train $DEVICE_ID`|`sh scripts/advyolo_{$PATCH_TYPE}_v8.sh infer $DEVICE_ID`|

    - `$PATCH_TYPE`:
        - __obj__ : maximizing postive sample's iou loss
            - `sh scripts/advyolo_obj_v8.sh train $DEVICE_ID`
            - finetune from `patches/object_score.png` from [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git)
        
        - __objcls__ : maximizing postive sample's iou loss while minimizing thire class probability
            - `sh scripts/advyolo_objcls_v8.sh train $DEVICE_ID`
            - finetune from `patches/class_detection.png` from from [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git)


## Defense Performance (Detection mAP50) $\uparrow$
Defense againsts adversarial patch attacked on INRIA Person validation set

- detector: YOLOv8n

|Defense method|No attack|OBJ|OBJ-CLS|Upper|
|-|-|-|-|-|
|no defense|96.42|56.74|75.30|65.77|
|LGS (WACV19)*|96.60|47.52|82.57|82.00|
|SAC (CVPR22)*|96.42|81.92|86.95|84.59|
|Jedi (CVPR23)*|96.64|57.63|64.35|58.13|
|PAD (CVPR2024)*|96.40|*87.53*|*87.69*|*88.92*|
||
|MoVQ reconstruction|__96.87__|73.60|86.69|77.04|
|MoVQ reconstruction with pixel mask|95.79|88.27|89.16|88.60|
|MoVQ reconstruction with code mask|95.20|__91.91__|__91.99__|__92.03__|


\* : reported from [PAD: Patch-Agnostic Defense against Adversarial Patch Attacks, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Jing_PAD_Patch-Agnostic_Defense_against_Adversarial_Patch_Attacks_CVPR_2024_paper.pdf)


## defense image FID $\downarrow$

|patch|pixel space masking|token space masking| 
|-|-|-|
|OBJ|84.74|__80.14__|
|OBJ-CLS|86.93|__80.87__|
|Upper|81.31|__75.51__|
## TODO
- Enhance the mAP50 while there is no adversarial patch attack is performed. (i.e. no attack)
