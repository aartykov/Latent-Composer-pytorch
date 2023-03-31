import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from decimal import Decimal

class RadiateBase(Dataset):
    def __init__(self,
                 mode,
                 data_root,
                 height,
                 width,
                 resize_img=False,
                 crop_img=False,
                 gramme_encoder=False,
                 interpolation="bicubic",
                 ):
        
        self.data_root = data_root
        self.folders = sorted([os.path.join(data_root, f) for f in os.listdir(data_root) if f.split("_")[0]!="junction"])
        # Match each radar img with corresp. cam image
        if mode == "train":
            tr_folders = self.folders[1:]
            self.image_pairs = self.match_images(tr_folders)
        elif mode == "val":
            val_folders = [self.folders[1]]
            self.image_pairs = self.match_images(val_folders)
            

        #self.input_folders = [os.path.join(data_root, f, "zed_left") for f in os.listdir(data_root)]
        #self.input_image_paths = sorted([img for folder in self.input_folders for img in Path(f'{folder}').glob(f'**/*.png')])
        
        #self.cond_folders = [os.path.join(data_root, f, "Navtech_Cartesian") for f in os.listdir(data_root)]
        #self.cond_image_paths = sorted([img for folder in self.cond_folders for img in Path(f'{folder}').glob(f'**/*.png')])

        """
        # Get frame names and corresponding timestamps for input images
        self.input_img_ts_dict = {}
        for f in self.folders:
            with open(os.path.join(f, "zed_left.txt")) as file:
                lines = files.readlines()
                for l in lines:
                    line = l.split(" ")
                    frame = os.path.join(root_dir, f, "zed_left", line[1]) + ".png"
                    ts = line[3].split("\n")[0] 
                    self.input_img_ts_dict[frame] = ts

        # Get frame names and corres. timestamps for cond images
        self.cond_img_ts_dict = {}
        for f in self.folders:
            with open(os.path.join(f, "Navtech_Cartesian.txt")) as file:
                lines = files.readlines()
                for l in lines:
                    line = l.split(" ")
                    frame = os.path.join(root_dir, f, "Navtech_Cartesian", line[1]) + ".png"
                    ts = line[3].split("\n")[0] 
                    self.cond_img_ts_dict[frame] = ts
        """

        self._length = len(self.image_pairs)

        self.image_pairs_keys_iterator = iter(self.image_pairs.keys())

        #self.size = size
        self.height = height
        self.width = width
        self.resize_img = resize_img
        self.crop_img = crop_img
        self.gramme_encoder = gramme_encoder
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def match_images(self, folders):
        images = {}

        for folder in folders:
            # Get input image timestamps
            cam_imgs = os.listdir(folder + "/zed_left/")
            with open(folder + "/zed_left.txt") as file:
                lines = file.readlines()
                input_img_timestamps = []
                input_imgs = []
                for i in range(len(cam_imgs)):
                    l = lines[i]
                    line = l.split(" ")
                    ts = line[3].split("\n")[0]
                    input_img_timestamps.append(Decimal(ts))
                    input_imgs.append(line[1])
            input_img_timestamps_np = np.asarray(input_img_timestamps)
            cond_input_pairs = {}
            # Get cond image timestamps
            radar_imgs = os.listdir(folder + "/Navtech_Cartesian/")
            with open(folder + "/Navtech_Cartesian.txt") as file:
                lines = file.readlines()
                for r in range(len(radar_imgs)):
                    l = lines[r]
                    line = l.split(" ")
                    cond_img_path = os.path.join(folder, "Navtech_Cartesian", line[1] + ".png")
                    ts = line[3].split("\n")[0]
                    idx = np.abs(input_img_timestamps_np - Decimal(ts)).argmin()
                    matching_inp_img = input_imgs[idx]
                    corresp_img_pth = os.path.join(folder, "zed_left", matching_inp_img+".png")
                    if corresp_img_pth not in list(cond_input_pairs.values()):
                        cond_input_pairs[cond_img_path] = corresp_img_pth

                    
            images.update(cond_input_pairs)
        
        return images

            
        

    def __getitem__(self, i):
        example = {}
        cond_image_path = next(self.image_pairs_keys_iterator)
        cond_image = Image.open(cond_image_path) #NOTE: Reads radar images with shape of (h,w)
        input_image = Image.open(self.image_pairs[cond_image_path])
        
        
        #if not cond_image.mode == "RGB" and not self.gramme_encoder:
        #    cond_image = cond_image.convert("RGB")  
        #elif self.gramme_encoder:
        #    cond_image = cond_image.convert("L")
        
        if not input_image.mode == "RGB":
            input_image = input_image.convert("RGB")

        
        # Center crop the cond image by half
        cond_img = np.array(cond_image).astype(np.uint8)
        crop = 512 #cond_img.shape[0] // 2
        h, w = cond_img.shape[0], cond_img.shape[1]
        cond_image = cond_img[(h - crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2]

        #input_image.show()
        
        
        if self.resize_img == True:
            # Before resizing the image, first pad
            input_image = np.array(input_image).astype(np.uint8) # (376,672)->orig_size
            input_image = np.pad(input_image, ((148,148),(0,0), (0,0)), 'constant') # (672,672)
            input_image = Image.fromarray(input_image).resize((self.height, self.width), resample=self.interpolation)

            # Resize the cond image to (256,256) 
            cond_image = Image.fromarray(cond_image)
            cond_image = cond_image.resize((256, 256), resample=self.interpolation)
            cond_image = np.array(cond_image).astype(np.uint8)
            cond_image = cond_image[:, :, np.newaxis]
        
        #input_image = self.flip(input_image)
        input_image = np.array(input_image).astype(np.uint8)
        example["jpg"] = (input_image / 127.5 - 1.0).astype(np.float32) # (h,w,3)
        #cond_image = cond_image[:, :, np.newaxis] #np.array(cond_image).astype(np.uint8)
        example["hint"] = (cond_image / 255.0).astype(np.float32) # (h,w,1)
        #NOTE: What should be the right key for cond image? 
        example["txt"] = ""
        return example


class RadiateTrain(RadiateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RadiateValidation(RadiateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
                         *args, **kwargs)



"""
class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
"""