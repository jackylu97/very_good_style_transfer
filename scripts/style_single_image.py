import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from styler import StyleConfig, Styler
import util

import torch
from PIL import Image

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

default_config = StyleConfig()
styler = Styler(default_config, device)

content_image = Image.open(util.content_img_path(CONTENT_IMG_NAME))
style_image = Image.open(util.style_img_path(STYLE_IMG_NAME))

output = styler.style(
    content_image,
    style_image,
)

util.imsave(output, name=OUTPUT_NAME)
