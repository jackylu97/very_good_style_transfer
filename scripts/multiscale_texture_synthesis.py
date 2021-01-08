import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from styler import StyleConfig, Styler
import util

import torch

device = torch.device("cuda")

IMSIZE = 256 # use small size if no gpu
NUM_OCTAVES = 3

CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"
NUM_FRAMES = 10

custom_params = dict(
    num_iters=100,
    content_weight=0.0,
    init_img_type='custom',
)

default_config = StyleConfig()
default_config.update(**custom_params)
styler = Styler(device, default_config)
loader = util.ImageLoader(IMSIZE, device)

def style_single_frame(init_img=None):
    for octave in range(NUM_OCTAVES):
        content_img, style_img = loader.load_content_style_imgs(CONTENT_IMG_NAME,
                                                                STYLE_IMG_NAME,
                                                                octave)

        # choose initialization for the octave
        if octave == 0 and init_img == None:
            init_img = content_img
        if octave > 0:
            init_img = loader.resize_image_octave(init_img, octave)

        init_img = styler.style(
            content_img,
            style_img,
            init_img=init_img
        )
        # loader.imsave(output, name=OUTPUT_NAME + f"_debug_{octave}")
    return init_img

def zoom(image, zoom_pixels):
    # PIL image -> remove borders -> resize -> PIL Image
    width, height = image.size
    left, right = zoom_pixels, width - zoom_pixels
    top, bottom = zoom_pixels, height - zoom_pixels
    return image.crop((left, top, right, bottom)).resize((width, height))

last_frame_result = None
for frame in range(NUM_FRAMES):
    output = style_single_frame(last_frame_result)
    image = loader.unload(output)
    image = zoom(image, 10)
    last_frame_result = loader.load(image, IMSIZE)

    loader.imsave(output, name=f"{OUTPUT_NAME}_f{frame}")