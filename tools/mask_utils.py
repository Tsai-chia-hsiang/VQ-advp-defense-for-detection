import numpy as np
import cv2
import torch.nn.functional as F
from typing import Literal, Iterable, Callable, Optional
from sklearn.cluster import DBSCAN
import torch
from deepcvext.img_normalize import normalize_heatmap


def dbscan_mask(binary: np.ndarray, min_samples: int = 20, eps_rate=30, eps=4):
    coords = np.column_stack(np.where(binary > 0))

    #clean = np.zeros(binary.shape, dtype=np.int32)
    label_map = np.zeros(binary.shape, dtype=np.int32)
    min_axi = np.min(label_map.shape[:2])
    
    if len(coords) == 0:
        return label_map, None
    
    clustering = DBSCAN(
        eps=min_axi/eps_rate if eps is None else eps, 
        min_samples=min_samples
    ).fit(coords)
    
    # Unique cluster labels (ignore noise = -1)
    unique_labels = set(clustering.labels_)
    unique_labels.discard(-1)
 
    for g in unique_labels:
        # Get coordinates belonging to cluster g
        
        cluster_coords = coords[clustering.labels_ == g]
        xs = cluster_coords[:, 0]
        ys = cluster_coords[:, 1]

        label_map[xs, ys] = g+1
        """
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        # Calculate cluster area (number of pixels in cluster)
        
        #ratio = min((w,h))/max((w,h))
        # Filter condition
        #if ratio >= r :
            # Assign cluster pixels in 'clean' mask as 255 (white)
            # clean[xs, ys] = 1
        """
    clean = np.where(label_map > 0, 1, 0).astype(np.int32)
    return clean, label_map

def region_filter(binary:np.ndarray, k=5, a:float=400):
    
    kernel = np.ones((k,k),np.uint8)
    b = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations = 1) 
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b)
    clean = np.zeros_like(binary)   
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= a:
            clean[labels == i] = 255
    
    
    kernel = np.ones((k,k),np.uint8)
    b = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    
    return b

def filtout_not_patch(binary:np.ndarray, mina=500)->np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean= np.zeros_like(binary).astype(np.uint8)
    for l in range(1, num_labels):
        g = labels == l
        a = stats[l, cv2.CC_STAT_AREA]
        if a < mina:
            continue
        ys, xs = np.where(g)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min
        h = y_max - y_min
        if a/(w*h) >= 0.5:
            clean[g] = 255
    return clean

def thresholding(img:np.ndarray, thr_method:Literal['adp','percentile', 'fix']='adp', thr:float=None, binary:bool=True, t_value:bool=False)->np.ndarray:
    
    if thr_method != 'adp':
        assert thr is not None
    match thr_method:
        case 'adp':
            v, t = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        case 'percentile':
            v, t= cv2.threshold(img, np.percentile(img, thr), maxval=255, type=cv2.THRESH_TOZERO)
        case 'fix':
            v, t = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        case _:
            raise NotImplementedError(f"{thr_method} is not supported.")
    
    if binary and thr_method != 'percentile':
        t = t.astype(np.int32)//255
    if t_value:
        return v, t
    return t

class Diff2HeatmapProcessor(
    Callable[[torch.Tensor, Optional[Iterable[str]], bool], torch.Tensor]
):
    def __init__(
        self, 
        smooth_k:int|torch.Tensor=7, 
        channel_reduce:Literal['sum', 'mean']='mean',
        batch_reduce:Literal['sum', 'mean'] ='mean', 
        scale_down:float=None,
        default_order:Iterable[str] = (
            'smooth', 'channel_reduce', 'normalize', 
            'scale', 'batch_reduce', 'scale_back','normalize'
        ),
        default_return:Literal['numpy','torch']='numpy'
    )->None:
        """
        For processing images in torch order: B x C x H x W
        """
        self.old_hw:tuple[int, int] = None # to store runtime given heatmap's origin h,w
        
        self.smooth_kernel = smooth_k
        if not isinstance(self.smooth_kernel, torch.Tensor):
            self.smooth_kernel = torch.ones((3, 1, smooth_k, smooth_k), dtype=torch.float32)/(smooth_k ** 2)
        
        self.scale_f = 1/scale_down if scale_down is not None else None
        self.channel_r = channel_reduce
        self.batch_r = batch_reduce

        self.process_map = {
            'smooth':self.smooth,
            'channel_reduce':self.channel_reduce,
            'batch_reduce':self.batch_reduce,
            'scale':self.scale_resolution,
            'scale_back':self.descale_resolution,
            'normalize':self.normalize
        }
        self.default_order:Iterable[str] = default_order
        self.default_return = default_return
        
    def smooth(self, heatmap:torch.Tensor, **kwargs)->torch.Tensor:
        
        if self.smooth_kernel.device != heatmap.device:
            self.smooth_kernel = self.smooth_kernel.to(device=heatmap.device)

        b, c, h, w = heatmap.shape
        assert c in {1,3}
        
        if self.smooth_kernel.shape[0] != c:
            match c:
                case 1:
                    self.smooth_kernel = self.smooth_kernel[0:1]
                case 3:
                    self.smooth_kernel = self.smooth_kernel.repeat(3, 1, 1, 1)
        
        smoothed = F.conv2d(
            heatmap, self.smooth_kernel, 
            padding=(self.smooth_kernel.shape[-2]// 2, self.smooth_kernel.shape[-1]//2), 
            groups=c
        )
        return smoothed
    
    def batch_reduce(self, heatmap:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        heatmap: B x C x H x W
        """
        if heatmap.shape[0] == 1:
            return heatmap
        
        match self.batch_r:
            case 'mean':
                return torch.mean(heatmap, dim=0, keepdim=True)
            case 'sum':
                return torch.sum(heatmap, dim=0, keepdim=True)
    
    def channel_reduce(self, heatmap:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        heatmap: B x C x H x W
        """
        if heatmap.shape[1] == 1:
            return heatmap
        
        match self.channel_r:
            case 'mean':
                return torch.mean(heatmap, dim=1, keepdim=True)
            case 'sum':
                return torch.sum(heatmap, dim=1, keepdim=True)
    
    def scale_resolution(self, heatmap:torch.Tensor, **kwargs)->torch.Tensor:
        if self.scale_f is None:
            return heatmap 
        return F.interpolate(heatmap, scale_factor=self.scale_f , mode='bilinear')
    
    def descale_resolution(self, heatmap:torch.Tensor, origin_shape:tuple[int,int])->torch.Tensor:
        if heatmap.shape[-2:] != origin_shape:
            return F.interpolate(heatmap, origin_shape, mode='bilinear')
        return heatmap  

    def normalize(self, heatmap:torch.Tensor, **kwargs)->torch.Tensor:
        return normalize_heatmap(heatmap)
    
    @torch.no_grad()
    def __call__(self, diff:torch.Tensor, process_order:Iterable[str]=None, to_img:bool=True)->torch.Tensor:
        """
        diff: B x C x H x W
        """
        heatmap = diff
        origin_shape = heatmap.shape[-2:] # h,w
        order = process_order if process_order is not None else self.default_order

        for task in order:
            heatmap = self.process_map[task](heatmap=heatmap, old_hw=origin_shape)

        if to_img:
            if order[-1] != 'normalize':
                heatmap=self.normalize(heatmap=heatmap)
        
        heatmap = heatmap.squeeze()
        if to_img:
            heatmap = (heatmap*255).to(torch.uint8)

        return heatmap    
    
    def generate_binary_mask_from_diff(
        self, diff:torch.Tensor, process_order:Optional[Iterable[str]]=None, 
        thr_method:Literal['adp','percentile', 'fix']='adp', thr:int=None, 
        return_framework:Optional[Literal['torch', 'numpy']]=None
    )->tuple[np.ndarray|torch.Tensor, np.ndarray|torch.Tensor]:
        
        heatmap = self(diff, process_order, to_img=True)
        h = heatmap.detach().cpu().numpy()

        mask = thresholding(img=h, thr_method=thr_method, thr=thr, binary=True, t_value=False)

        rf = return_framework if return_framework is not None else self.default_return
        match rf:
            case 'torch':
                mask = torch.from_numpy(mask).to(device=heatmap.device)
                return heatmap, mask
            case 'numpy':
                return h, mask