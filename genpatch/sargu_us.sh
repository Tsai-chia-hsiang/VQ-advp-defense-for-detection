name=advyolo_rotate_blur_us
LR=0.03
conf=0.25
python advpatch_attack.py --name $name \
    --lr $LR \
    --patch_random_rotate --patch_blur \
    --conf $conf