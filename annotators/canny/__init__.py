import cv2
import einops
import numpy as np
import torch
from annotators.util import HWC3

class CannyDetector:
    def __call__(self, torch_img_batch, low_threshold=100, high_threshold=200):
        img_np = torch_img_batch.detach().cpu().permute(0,2,3,1).numpy() # (b, h, w, c)
        canny_torch = []
        for i in range(img_np.shape[0]):
            img = np.uint8((img_np[i] + 1.0) * 127.5) # [-1,1] --> [0, 255], shape: (h,w,c)
            canny_img_np = cv2.Canny(img, low_threshold, high_threshold)
            canny_img_np = HWC3(canny_img_np)
            canny_img_torch = (torch.from_numpy(canny_img_np)/127.5 - 1.0).float().unsqueeze(0) # [0,255] --> [-1, 1], shape: (b , h, w, c)
            
            canny_torch.append(canny_img_torch)
        canny_torch = torch.cat(canny_torch, dim=0).cuda()
        canny_torch = einops.rearrange(canny_torch, 'b w h c -> b c h w').clone()
        return canny_torch