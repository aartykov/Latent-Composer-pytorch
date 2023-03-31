import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

class CustomDatasetBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 clip_version="openai/clip-vit-large-patch14",
                 text_max_length=77
                 ):
        #self.data_paths = txt_file
        self.data_root = data_root
        self.image_files = self.get_image_files()
        self.txt_labels = self.get_txt_labels()
        #with open(self.data_paths, "r") as f:
        #    self.image_paths = f.read().splitlines()
        
        self._length = len(self.image_files)
        

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        self.text_max_length = text_max_length


    def get_image_files(self):
        image_files = os.listdir(self.data_root)
        return image_files

    def get_txt_labels(self):
        with open(os.path.join(self.data_root, self.txt_label_file), 'r') as f:
            txt_labels = f.read().splitlines()
        return txt_labels 

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_root, image_name)
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        prompt = self.txt_labels[idx]
        # Tokenize the prompt
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=self.text_max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["input_image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["txt"] = prompt_tokenized ## (1,text_max_length)
        return example


class CustomDatasetTrain(CustomDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class CustomDatasetValidation(CustomDatasetBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)




