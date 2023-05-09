import os
import numpy as np
import cv2 
from scipy.ndimage import gaussian_filter
from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder, SpatialRescaler
import torch
from einops import rearrange
from PIL import Image
from transformers import CLIPTokenizer
from annotators.canny import CannyDetector
from annotators.midas import MidasDetector


rescaler = SpatialRescaler(n_stages=3, in_channels=3)
inp = torch.randn((8,3,512,512))
out = rescaler(inp)
print("Out SHAPE: ", out.shape)

"""
img = torch.randn((8,3,256,256))
clip = FrozenCLIPImageEmbedder()
out = clip(img)
print(out.shape)
print(clip.parameters())
"""

"""
pth = "/home/arslan/Downloads/living_room_images/0.png"
img = Image.open(pth)
print("Original Size: ", img.size)
width, height = img.size

new_width = width * 0.21
new_height = height * 0.5

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

# Crop the center of the image
img = img.crop((left, top, right, bottom))
#img.show()
print("Img Size: ", img.size)
print(20//3)
"""

"""
mask = [""]
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
encoding_mask = tokenizer(mask, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"]

text = ["bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz"]
encoding_text = tokenizer(text, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"]

#print(encoding_mask.shape, encoding_text.shape)

def prob_mask_like(shape, prob, device):
    #return (0 - 1) * torch.rand(shape) + 1 < prob
    return torch.zeros(shape, device=device).uniform_(0, 1) < prob

null_cond_prob = 0.8
batch = 8
keep_mask = prob_mask_like((batch,1), 1. - null_cond_prob, device = "cpu")
print(keep_mask)
#print(encoding_mask.shape)

text = ["bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz"] #torch.ones((1,77,768))
text_null = "" #torch.zeros((1,77,768))


#text_embeds = encoding_text * keep_mask #np.where(keep_mask, text, text_null)
text_embeds = torch.cat([o.unsqueeze(0) if b else encoding_mask for o, b in zip(encoding_text, list(keep_mask))])
print(text_embeds.shape)
print(text_embeds[1].shape)
print(text_embeds[2])
"""


#clip_embedder = FrozenCLIPEmbedderWithProjection(device="cpu")
#input_txt = "Hello World"
#last_hidden_state, text_embeds = clip_embedder(input_txt) # (1,77,768)
#print(text_embeds.shape)

"""
def smoothed_cielab_histogram(image):
    # Convert image to CIELab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Create an empty array to store the smoothed CIELab histogram
    hist = np.zeros((11, 5, 5))

    # Iterate over every pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Calculate CIELab values for the pixel
            l, a, b = lab_image[i, j]

            # Quantize CIELab values to 11 hue, 5 saturation, and 5 light values
            l_bin = int(round(l / 100 * 4))
            a_bin = int(round((a + 128) / 256 * 10))
            b_bin = int(round((b + 128) / 256 * 10))

            # Make sure a_bin and b_bin are within range [0, 10]
            if a_bin > 10:
                a_bin = 10
            if b_bin > 4:
                b_bin = 4

            if l_bin > 4:
                l_bin = 4
            # Increment the corresponding bin in the histogram
            hist[a_bin, b_bin, l_bin] += 1

    # Apply Gaussian filter to smooth the histogram
    sigma = 10
    hist_smoothed = gaussian_filter(hist, sigma=sigma)

    # Normalize the histogram
    hist_norm = hist_smoothed / np.sum(hist_smoothed)

    return hist_norm

img_pth = "/home/arslan/Downloads/room_images_created/1.jfif"
img = cv2.imread(img_pth)
hist = smoothed_cielab_histogram(img) 
print(hist.shape)
"""


"""
img_path = "/home/arslan/Downloads/room_images_created/tmp42oix8ud.png"

img = cv.imread(img_path)

# Convert the image from the RGB color space to the CIELab color space
lab_img = cv.cvtColor(img, cv.COLOR_BGR2Lab)

# Define the number of quantization levels for each dimension
hue_bins = 11
saturation_bins = 5
lightness_bins = 5

# Quantize the CIELab color space
quantized_lab_img = lab_img.copy()
quantized_lab_img[..., 0] = np.floor(quantized_lab_img[..., 0] / (100 / hue_bins))
quantized_lab_img[..., 1] = np.floor(quantized_lab_img[..., 1] / (128 / (saturation_bins-1)))
quantized_lab_img[..., 2] = np.floor(quantized_lab_img[..., 2] / (128 / (lightness_bins-1)))

# Calculate the histogram of the quantized image
hist = cv.calcHist([quantized_lab_img], [0, 1, 2], None, [hue_bins, saturation_bins, lightness_bins], [0, hue_bins, 0, saturation_bins, 0, lightness_bins])


#print(hist)
sigma = 10.0
# Smooth the histogram using Gaussian blur
hist = cv.GaussianBlur(hist, (5, 5), sigma)

#print(hist)

# Normalize the histogram
hist = hist / (img.shape[0] * img.shape[1])

print(hist)
# Flatten the histogram into a feature vector
feature_vector = hist.flatten()
print(feature_vector.shape)
#print(feature_vector)
"""




"""
# Load the image in the RGB color space
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Convert the image to the CIELab color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Define the quantized color space
hue_bins = 11
saturation_bins = 5
lightness_bins = 5
color_space = np.zeros((hue_bins, saturation_bins, lightness_bins), dtype=np.float32)

# Compute the size of each color bin
hue_range = np.linspace(0, 360, hue_bins+1)
saturation_range = np.linspace(0, 1, saturation_bins+1)
lightness_range = np.linspace(0, 100, lightness_bins+1)

# Compute the smoothed histogram of the image in the quantized color space
sigma = 10.0
histogram = np.zeros_like(color_space)
for i in range(hue_bins):
    for j in range(saturation_bins):
        for k in range(lightness_bins):
            hmin, hmax = hue_range[i], hue_range[i+1]
            smin, smax = saturation_range[j], saturation_range[j+1]
            lmin, lmax = lightness_range[k], lightness_range[k+1]
            mask = (lab[..., 0] >= hmin) & (lab[..., 0] < hmax) & (lab[..., 1] >= smin) & (lab[..., 1] < smax) & (lab[..., 2] >= lmin) & (lab[..., 2] < lmax)
            histogram[i, j, k] = np.sum(mask.astype(np.float32))
            histogram[i, j, k] = cv2.GaussianBlur(np.array(histogram[i, j, k]), (0, 0), sigma)
            
print(np.sum(histogram))
# Normalize the histogram to obtain a probability distribution
histogram /= np.sum(histogram)

# Flatten the histogram into a feature vector
feature_vector = histogram.flatten()

print(feature_vector.shape)
"""  


