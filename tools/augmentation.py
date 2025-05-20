import cv2
import numpy as np
import torch.nn.functional as F
import torch

def smooth(img:np.ndarray, ksize:int=7)->np.ndarray:
    return cv2.filter2D(
        img if img.ndim == 3 else np.expand_dims(img, axis=-1), -1, 
        np.ones((ksize, ksize)) / ksize**2
    )

def torch_smooth(img:torch.Tensor, kernel:torch.Tensor=None, ksize:int=7)->torch.Tensor:
    b, c, h, w = img.shape
    assert c in {1,3}
    if kernel is None:
        sk = torch.ones((c, 1, ksize, ksize), dtype=img.dtype, device=img.device)
        sk /= ksize ** 2
    else:
        assert kernel.shape[0] in {1,3}
        sk = kernel
        if sk.shape[0] != c:
            if c == 1:
                sk = sk[0:1]
            elif c == 3:
                sk = sk.repeat(3, 1, 1, 1)
    
    smoothed = F.conv2d(img, sk, padding=(sk.shape[-2]// 2, sk.shape[-1]//2), groups=c)
    return smoothed