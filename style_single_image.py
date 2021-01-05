from styler import StyleConfig, Styler
import util

import torch

import matplotlib.pyplot as plt

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

loader = util.ImageLoader(imsize, device)
content_img = loader.load_content_img(CONTENT_IMG_NAME)
style_img = loader.load_style_img(STYLE_IMG_NAME)

default_config = StyleConfig()
styler = Styler(default_config)

output = styler.style(
    content_img,
    style_img,
)

plt.figure()
loader.imsave(output, name=OUTPUT_NAME)
