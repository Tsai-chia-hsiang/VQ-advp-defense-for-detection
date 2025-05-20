LR=0.01
conf=0.25
epochs=20
name=advyolo_objcls
python advpatch_attack.py \
    --task $1 --name $name \
    --imgsz 640 \
    --patch_random_rotate --patch_blur \
    --conf $conf  --lr $LR \
    --objective obj-cls\
    --device $2 --epochs $epochs\
    --sup_prob_loss \
    --init_patch exps/advyolo_oldobjcls/worst.pt
