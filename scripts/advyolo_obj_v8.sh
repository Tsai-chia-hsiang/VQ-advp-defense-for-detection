LR=0.01
conf=0.25
epochs=4
name=advyolo_oldobj
python advpatch_attack.py \
    --task $1 --name $name \
    --imgsz 640  \
    --patch_random_rotate --patch_blur \
    --conf $conf  --lr $LR \
    --objective obj\
    --device $2 --epochs $epochs\
    --sup_prob_loss \
    --init_patch exps/advyolo_oldobj/worst.pt
