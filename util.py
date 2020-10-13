import torch
import torchvision.transforms as transforms

from PIL import Image
import os

CONTENT_PATH = "./content/{}"
STYLE_PATH = "./style/{}"
OUTPUT_DIR = "output"
OUTPUT_PATH = f"./{OUTPUT_DIR}/{{}}"

def calc_img_size(size, mx):
    # get height, width with same aspect ratio, given that no dimension can exceed mx
    assert len(size) == 2, "image should have dim=2 !!"
    h,w = size

    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        return (int(w), mx)
    if w > mx:
        h = (float(mx) / float(w)) * h
        return (mx, int(h))
    return (h, w)

class ImageLoader:

    def __init__(self, 
                 max_imsize,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.max_imsize = max_imsize
        self.device = device

    def init_loaders(self, imsize):
        self.imsize = imsize
        self.loader = transforms.Compose(
            [transforms.Resize(imsize, interpolation=Image.BILINEAR),
            transforms.ToTensor()]  # scale imported image
        )  # transform it into a torch tensor
        self.unloader = transforms.ToPILImage()  # reconvert into PIL image

    def load_content_img(self, image_name):
        image = Image.open(self.content_img_path(image_name))
        imsize = calc_img_size(image.size, self.max_imsize)
        self.init_loaders(imsize)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
        
    def load_style_img(self, image_name):
        # automatically resizes style image to the size of the content_img
        # this is done to run content / style losses on the same execution graph
        assert self.loader, "Need to initialize ImageLoader with content image first! uwu"
        image = Image.open(self.style_img_path(image_name))
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imsave(self, tensor, name, filetype="png"):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unloader(image)
        img_name = self.output_path(f"{name}.{filetype}")
        if not os.path.exists(os.path.abspath(OUTPUT_DIR)):
            os.mkdir(OUTPUT_DIR)
        image.save(img_name)

    def content_img_path(self, image_name):
        return CONTENT_PATH.format(image_name)

    def style_img_path(self, image_name):
        return STYLE_PATH.format(image_name)

    def output_path(self, image_name):
        return OUTPUT_PATH.format(image_name)
