import sys
import os
from .import _LOCAL_DIR_
sys.path += [os.path.abspath(_LOCAL_DIR_.parent/"maskgit")]
import torch
import jax.numpy as jnp
from maskgit.maskgitlib.inference import ImageNet_class_conditional_generator
from torch2jax import t2j, j2t

MASKGIT_MODEL:ImageNet_class_conditional_generator = None
MASKGIT_MODEL_IMGSZ=512

def maskgit_reconstruct(batch:dict)->dict:
    """
    torch image shape: B x C X H X W
    jnp image shape : B x H x W X C
    """
    global MASKGIT_MODEL
    if MASKGIT_MODEL is None:
        MASKGIT_MODEL = ImageNet_class_conditional_generator(image_size=MASKGIT_MODEL_IMGSZ)
    
    batch['img'] = t2j(torch.permute(batch['img'], dims=(0, 2, 3, 1)))
    tokens = MASKGIT_MODEL.tokenizer_model.apply(
        MASKGIT_MODEL.tokenizer_variables,
        {"image": batch['img']},
        method=MASKGIT_MODEL.tokenizer_model.encode_to_indices,
        mutable=False
    )
    rimg = MASKGIT_MODEL.tokenizer_model.apply(
        MASKGIT_MODEL.tokenizer_variables,
        tokens,
        method=MASKGIT_MODEL.tokenizer_model.decode_from_indices,
        mutable=False
    )
    batch['img'] = j2t(jnp.clip(rimg, 0, 1)) # B x H x W x C
    batch['img'] = torch.permute(batch['img'], dims=(0,3,1,2))
    return batch
