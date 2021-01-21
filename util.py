import torch
import torchvision.transforms as transforms

from PIL import Image
from dataclasses import is_dataclass
import os
import math

CONTENT_PATH = "./content/{}"
STYLE_PATH = "./style/{}"
OUTPUT_DIR = "output"
OUTPUT_PATH = f"./{OUTPUT_DIR}/{{}}"

def content_img_path(image_name):
    return CONTENT_PATH.format(image_name)

def style_img_path(image_name):
    return STYLE_PATH.format(image_name)

def output_path(image_name):
    return OUTPUT_PATH.format(image_name)

def calc_img_size(size, mx):
    # get height, width with same aspect ratio, setting the larger dimension to mx
    assert len(size) == 2, f"image should have dim=2 !! got {size}"
    h, w = size

    if h > w:
        w = (float(mx) / float(h)) * w
        return (int(w), mx)
    else:
        h = (float(mx) / float(w)) * h
        return (mx, int(h))

def calc_octave_resolution(final_resolution, octave, num_octaves):
    return int(final_resolution / (math.sqrt(2) ** (num_octaves - octave - 1)))

def imsave(image, name, filetype="png", config=None):
    '''
    Optionally supply a config to save corresponding StyleConfigs to a text file with
    the same name.
    '''
    img_name = OUTPUT_PATH.format(f"{name}.{filetype}")
    if not os.path.exists(os.path.abspath(OUTPUT_DIR)):
        os.mkdir(OUTPUT_DIR)
    if config:
        assert is_dataclass(config)
        with open(f'{name}.txt', 'w') as file:
            file.write(json.dumps(asdict(config)))
    image.save(img_name)

class ImageLoader:

    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''
        Arg max_dimsize corresponds to the largest pixel length allowed in any dimension of the
        image. If using 'octave' parameter, initialise max_dimsize to be the size of the first
        octave.
        '''
        self.device = device
        self.loader = lambda imsize: transforms.Compose(
            [transforms.Resize(imsize, interpolation=Image.BILINEAR), # scale imported image
             transforms.ToTensor()]  # transform it into a torch tensor
        )
        self.unloader = transforms.ToPILImage() # reconvert into PIL image

    def load(self, image, imsize):
        # PIL Image object to tensor, resized
        # resize -> tensor -> add batch dim -> mount to gpu
        return (self.loader(imsize)(image)).unsqueeze(0).to(self.device, torch.float)

    def unload(self, tensor):
        # tensor to PIL Image object
        # mount cpu -> remove batch dim -> reconvert to PIL image
        return self.unloader(tensor.cpu().clone().squeeze(0))

    def load_content_style_imgs(self, content_image, style_image, max_dimsize):
        '''
        Always couple loading content/style images, since they need to be the same size anyways
        '''
        # calculate image size, given max dimension size
        imsize = calc_img_size(content_image.size, max_dimsize)

        content_image = self.load(content_image, imsize)
        style_image = self.load(style_image, imsize)

        return content_image, style_image
