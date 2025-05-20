LR=0.02
conf=0.25
epochs=100
name=advyolo_upper
python advpatch_attack.py \
    --task $1 --name $name \
    --imgsz 640 \
    --patch_random_rotate --patch_blur \
    --conf $conf  --lr $LR \
    --objective obj\
    --device $2 --epochs $epochs\
    --attacker ./ultralytics_advpattack_lib/cfg/attacker_upper.yaml \
    --sup_prob_loss \
    --init_patch exps/advyolo_oldupper/worst.pt