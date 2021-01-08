import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from styler import StyleConfig, Styler
import util

import torch

device = torch.device("cuda")

IMSIZE = 256 if torch.cuda.is_available() else 128  # use small size if no gpu
NUM_OCTAVES = 3

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

debug_params = dict(
    num_iters=100,
)

default_config = StyleConfig()
default_config.update(**debug_params)
styler = Styler(device, default_config)

output = None
for octave in range(NUM_OCTAVES):
    loader = util.ImageLoader(IMSIZE, device)
    content_img, style_img = loader.load_content_style_imgs(CONTENT_IMG_NAME,
                                                            STYLE_IMG_NAME,
                                                            octave)
    init_img = None
    if output != None:
        init_img = loader.resize_image_octave(output, octave)

    output = styler.style(
        content_img,
        style_img,
        init_img=init_img
    )
    loader.imsave(output, name=OUTPUT_NAME + f"_debug_{octave}")

loader.imsave(output, name=OUTPUT_NAME)
