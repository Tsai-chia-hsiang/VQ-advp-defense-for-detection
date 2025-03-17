name=advyolo 
LR=0.03
conf=0.25
# --sup_prob_loss --logit_to_prob
python advpatch_attack.py \
    --task $1 --name $name \
    --patch_random_rotate --patch_blur \
    --conf $conf  --lr $LR  --vq