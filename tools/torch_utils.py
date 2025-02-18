import random
import numpy as np
import torch

def set_seed(seed=891122, loader=None):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True        
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

def a_device(device_id:int)->torch.device:
    """
    TODO: multi device case (for parallel forward)
    """
    if device_id < 0 or (not torch.cuda.is_available()):
        return torch.device('cpu')
    else:
        N_GPUs_available = torch.cuda.device_count()
        if device_id < N_GPUs_available:
            return  torch.device('cuda', index=device_id)
        else:
            return torch.device('cuda', index=N_GPUs_available-1)