import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from styler import StyleConfig, Styler
import util

import torch
from PIL import Image

device = torch.device("cuda")

NUM_OCTAVES = 3

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"
NUM_FRAMES = 24 * 12

custom_params = dict(
    max_dimsize=512,
    num_iters=100,
    content_weight=0.0,
    init_img_type='custom',
)

default_config = StyleConfig()
default_config.update(**custom_params)
styler = Styler(default_config, device)

def zoom(image, zoom_pixels):
    # PIL image -> remove borders -> resize -> PIL Image
    width, height = image.size
    left, right = zoom_pixels, width - zoom_pixels
    top, bottom = zoom_pixels, height - zoom_pixels
    return image.crop((left, top, right, bottom)).resize((width, height))

content_image = Image.open(util.content_img_path(CONTENT_IMG_NAME))
style_image = Image.open(util.style_img_path(STYLE_IMG_NAME))

last_frame_result = content_image
for frame in range(NUM_FRAMES):
    image = styler.style_multiscale(content_image,
                                    style_image,
                                    octaves=NUM_OCTAVES,
                                    init_img=last_frame_result)
    last_frame_result = zoom(image, 10)

    util.imsave(image, name=f"{OUTPUT_NAME}_f{frame}")