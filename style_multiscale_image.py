from styler import StyleConfig, Styler
import util

import torch

device = torch.device("cuda")

IMSIZE = 256 if torch.cuda.is_available() else 128  # use small size if no gpu
NUM_OCTAVES = 3

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

output = None

default_config = StyleConfig()
styler = Styler(device, default_config)

for octave in range(NUM_OCTAVES):
    loader = util.ImageLoader(IMSIZE, device)
    content_img, style_img = loader.load_content_style_imgs(CONTENT_IMG_NAME, STYLE_IMG_NAME, octave)
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
