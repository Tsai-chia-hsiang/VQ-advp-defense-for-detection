from PAD.fuse_filter import heatmap_filter
import cv2
import numpy as np
from deepcvext.draw import canvas
from deepcvext.utils import cvtcolor
from pathlib import Path
check = "crop001512"
raw_heat = Path(f"visualization/padcd_no_morph/{check}.png")
h_t = cv2.imread(raw_heat, cv2.IMREAD_GRAYSCALE)
h_t = h_t[:, :h_t.shape[1]//2-5]
base_kernel_size = int(min(h_t.shape[0], h_t.shape[1])/80)
kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
# kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
h_t_o=cv2.morphologyEx(h_t, cv2.MORPH_OPEN,kernel, iterations=1)

kernel=np.ones((base_kernel_size,base_kernel_size),np.uint8)
# kernel=np.ones((base_kernel_size*2,base_kernel_size*2),np.uint8)
h_t_o_c=cv2.morphologyEx(h_t_o,cv2.MORPH_CLOSE,kernel, iterations=2)

kernel=np.ones((base_kernel_size*3,base_kernel_size*3),np.uint8)
h_t_o_c_o=cv2.morphologyEx(h_t_o_c,cv2.MORPH_OPEN,kernel, iterations=2)
# cv2.imwrite(savefig_path+name+"_t_open_close_open.png", crosion3)
mask = np.where(h_t_o_c_o>1, 255, 0)

im0 = cv2.imread(f"exps/advyolo_obj/attacked/val/images/{check}.png")
operations = [
    ["thr", cvtcolor(h_t,3)],
    ["open", cvtcolor(h_t_o, 3)],
    ["close", cvtcolor(h_t_o_c,3)], 
    ["open1", cvtcolor(h_t_o_c_o, 3)],
    ["pixelmask", cvtcolor(mask, 3)]
]
dst_root = Path("visualization/masks")/check
print(dst_root)
dst_root.mkdir(parents=True, exist_ok=True)
for name, o in operations:
    cv2.imwrite(dst_root/f"{name}.png", o)
