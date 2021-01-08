import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from styler import StyleConfig, Styler
import util

import torch

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

loader = util.ImageLoader(imsize, device)
content_img, style_img = loader.load_content_style_imgs(CONTENT_IMG_NAME, STYLE_IMG_NAME)

debug_params = dict(
    num_iters=100,
)

default_config = StyleConfig()
default_config.update(**debug_params)
styler = Styler(device, default_config)

output = styler.style(
    content_img,
    style_img,
)

loader.imsave(output, name=OUTPUT_NAME)
