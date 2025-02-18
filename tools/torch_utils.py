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
