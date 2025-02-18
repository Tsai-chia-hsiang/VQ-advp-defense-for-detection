import torch
from typing import Literal
from torch.optim.optimizer import Optimizer

def optimize_adv_img(img:torch.Tensor, loss:torch.Tensor, method:Literal['gd', 'pgd']='gd', opt:Optimizer=None, **kwargs):
    
    loss.backward()
    
    match method:
        case 'gd':
            opt.step()
        case 'pgd':
            with torch.no_grad():
                img = img + kwargs['alpha'] * img.grad.sign()
    
    img.data.clamp_(0, 1)