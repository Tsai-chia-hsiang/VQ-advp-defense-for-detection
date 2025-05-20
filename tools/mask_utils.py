import numpy as np
import cv2
from typing import Literal
from sklearn.cluster import DBSCAN

def dbscan_mask(binary: np.ndarray, min_samples: int = 20, eps: float = 3, r=0.3):
    coords = np.column_stack(np.where(binary > 0))

    clean = np.zeros(binary.shape, dtype=np.int32)
    label_map = np.zeros(binary.shape, dtype=np.int32)

    if len(coords) == 0:
        return clean, label_map, None
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Unique cluster labels (ignore noise = -1)
    unique_labels = set(clustering.labels_)
    unique_labels.discard(-1)
 
    for g in unique_labels:
        # Get coordinates belonging to cluster g
        
        cluster_coords = coords[clustering.labels_ == g]
        xs = cluster_coords[:, 0]
        ys = cluster_coords[:, 1]
        
        
        label_map[xs, ys] = g+1
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        # Calculate cluster area (number of pixels in cluster)
        
        ratio = min((w,h))/max((w,h))
        # Filter condition
        if ratio >= r :
            # Assign cluster pixels in 'clean' mask as 255 (white)
            clean[xs, ys] = 1

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