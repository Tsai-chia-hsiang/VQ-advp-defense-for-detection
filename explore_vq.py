import torch
from vq import MOVQ
from pathlib import Path
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from deepcvext.draw import canvas
from deepcvext.img_normalize import _IMG_DESTD_
from deepcvext import tensor2img
from deepcvext.utils import cvtcolor
from deepcvext.seg import cv2_read_binary_map
from tools.mask_utils import dbscan_mask
from tqdm import tqdm
att_img_dir = Path("exps")/"advyolo_obj"/"attacked"/"val"/"images"
clean_img_dir = Path("/datasets")/"INRIAPerson"/"val"/"images"
mask_dir = att_img_dir.parent.parent/"val_mask"

@torch.no_grad()
def compare_token_idices():
    
    def cmp_entry_idx(m1:torch.Tensor, m2:torch.Tensor)->torch.Tensor:
        assert m1.shape == m2.shape
        diff_coors = torch.where(m1 != m2)
        diff_map = torch.zeros_like(m1)
        diff_map[diff_coors[0], diff_coors[1]] = 1
        return diff_coors, diff_map
    
    model = MOVQ.from_cfg(on_device=torch.device('cuda', 1))
    samples = [Path(_.name) for _ in att_img_dir.glob("*.png")]
    for sample in tqdm(samples): 
        clean = cv2.imread(clean_img_dir/sample)
        att = cv2.imread(att_img_dir/sample)
        
        _, _, clean_tok, (t,b,l,r) = model.recons_cv2(img=clean, degrad_levels=[0])
        clean_tok = clean_tok.squeeze()
        _, _, att_tok, (t,b,l,r) = model.recons_cv2(img=att, degrad_levels=[0])
        att_tok = att_tok.squeeze()
        diff_coors, diff_map = cmp_entry_idx(m1=clean_tok, m2=att_tok)
        
        diff_map = diff_map.cpu().numpy()
        diff_map_denoise, _ = dbscan_mask(binary=diff_map)
        
        mask = cv2_read_binary_map(pth=mask_dir/sample.stem/"union.png", target_hw=att_tok.shape,dtype=np.int32)
        
        cv2.imwrite(
            Path("visualization")/"entry_cmp"/sample, 
            canvas(
                [cvtcolor(mask*255, 3), cvtcolor(diff_map*255, 3), cvtcolor(diff_map_denoise*255, 3)], 
                bar_color=(255,255,255)
            )
        )
        
        cv2.imwrite(
            Path("visualization")/"img_cmp"/sample, 
            canvas([clean, att], bar_color=(255,255,255))
        )
    
@torch.no_grad()
def fragment_token_mask():
    model = MOVQ.from_cfg(on_device=torch.device('cuda', 1))
    sample =  "person_316.png" #"crop_000001.png"
    sample = Path(sample)
    att = cv2.imread(att_img_dir/sample)
    _, _, att_tok, (t,b,l,r) = model.recons_cv2(img=att, degrad_levels=[0])
    gt_mask = cv2_read_binary_map(mask_dir/sample.stem/"union.png", target_hw=att_tok.shape[1:])
    mask_rate = [-0.0001, 0.3, 0.5, 0.7, 1.0]
    ms = []
    rimgs = []
    for p in mask_rate:
        rng = np.random.default_rng(891122)
        keep = rng.random(gt_mask.shape) <= p
        m = (gt_mask & keep).astype(np.int32)
        rimg = model.decode_code(code_b=att_tok, mask=torch.from_numpy(1-m).squeeze().to(model.device))
        rimg = tensor2img(timg=rimg, scale_back_f=_IMG_DESTD_)
        rimgs.append(rimg[t:b, l:r])
        ms.append(cvtcolor(m.astype(np.uint8)*255, 3))
    cv2.imwrite(Path("visualization/fragmask/masks")/sample, canvas(ms, bar_color=(255,255,255)))
    cv2.imwrite(Path("visualization/fragmask/images")/sample, canvas(rimgs, bar_color=(255,255,255)))

@torch.no_grad() 
def semantic_manipulate():
    def mask_g():
        mask_generator = sam_model_registry["vit_l"](checkpoint="/datasets/ckpts/sam_vit_l_0b3195.pth")
        mask_generator.eval()
        mask_generator.to(device=torch.device('cuda', 1))
        mask_generator =  SamAutomaticMaskGenerator(mask_generator)
        return mask_generator
    
    src_path = clean_img_dir/"person_138.png"
    src_dir= Path("visualization/token_replacement/manipulate")/src_path.stem
    src_dir.mkdir(parents=True, exist_ok=True)

    src = cv2.imread(src_path)
    cat = cv2.imread("visualization/token_replacement/cat.png")
    cat = cv2.resize(cat, dsize=None, fx=0.15, fy=0.15)
    cat_mask = np.zeros_like(src)
    cat_mask[-cat.shape[0]:, :cat.shape[1]] = cat
    cv2.imwrite(src_dir/"catimg.png", cat_mask)
    m_g = mask_g()
    fg = m_g.generate(cat_mask)[1].get('segmentation').astype(np.uint8)*255
    fg = np.expand_dims(np.where(fg > 0, 255, 0), -1).astype(np.uint8)

    del m_g

    model = MOVQ.from_cfg(on_device=torch.device('cuda', 1))
    _, _, cat_tok, (t,b,l,r) = model.recons_cv2(cat_mask)
    _, _, clean_tok, (t,b,l,r) = model.recons_cv2(img=src)
    cat_tok = cat_tok.squeeze()
    cv2.imwrite(src_dir/"catmask_pxi.png", fg)
    fg = cv2.resize(fg, (cat_tok.shape[1], cat_tok.shape[0]))
    cv2.imwrite(src_dir/"catmask.png", fg)
    fg = torch.from_numpy(np.where(fg.squeeze() > 0, 1, 0)).to(cat_tok.device)
    manipulate = torch.where(fg>0, cat_tok, clean_tok)
    
    rimg = model.decode_code(manipulate)
    rimg = tensor2img(rimg, _IMG_DESTD_)
    cv2.imwrite(src_dir/'m.png', rimg)

@torch.no_grad()
def token_space_masking_n_decode():
    
    def random_patch(h, w, square_size=None, N=3, center_bias_strength=0.2):
        """
        Generate a binary mask with a single square of 1s at a random location.

        Parameters:
        - h, w: height and width of the full mask
        - square_size: side length of the square (in pixels)
        - seed: optional RNG seed for reproducibility

        Returns:
        - mask: (h, w) binary NumPy array with a square of 1s
        """
        if square_size is None:
            square_size = int(min(h,w)*0.15)
        if square_size > h or square_size > w:
            raise ValueError("square_size must be <= height and width")

        mask = np.zeros((h, w), dtype=np.uint8)
        center_y, center_x = h // 2, w // 2
        std_y = center_bias_strength * h
        std_x = center_bias_strength * w
        for _ in range(N): 
            top = int(np.random.normal(center_y, std_y)) - square_size // 2
            left = int(np.random.normal(center_x, std_x)) - square_size // 2

            # Clamp to valid bounds
            top = np.clip(top, 0, h - square_size)
            left = np.clip(left, 0, w - square_size)
            mask[top:top + square_size, left:left + square_size] = 1
     
        return mask

    src_path = clean_img_dir/"person_316.png"
    src_dir= Path("visualization/random_mask")/src_path.stem
    src_dir.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(src_path)
    model = MOVQ.from_cfg(on_device=torch.device('cuda', 1))
    _, _, clean_tok, (t,b,l,r) = model.recons_cv2(img=src)
    tok_mask = random_patch(
        h=clean_tok.shape[1], w=clean_tok.shape[2],
        N = np.random.choice([3, 4, 5, 6])
    )

    cv2.imwrite(src_dir/"rmask.png", cvtcolor(tok_mask*255, 3))
    tok_mask = torch.from_numpy(tok_mask).to(device=clean_tok.device)
    rimg = model.decode_code(code_b=clean_tok, mask=1-tok_mask)
    rimg = tensor2img(rimg, scale_back_f=_IMG_DESTD_)[t:b, l:r]
    cv2.imwrite(src_dir/"im.png", rimg)

@torch.no_grad()
def directly_reconst():
    src = cv2.imread(att_img_dir/"crop001573.png")
    model = MOVQ.from_cfg(on_device=torch.device('cuda', 1))
    rimg, _, clean_tok, (t,b,l,r) = model.recons_cv2(img=src)
    cv2.imwrite("att_recon.png", rimg[0])



if __name__ == "__main__":

    #fragment_token_mask()
    # token_space_masking_n_decode()
    # directly_reconst()
    semantic_manipulate()