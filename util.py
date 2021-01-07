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
    h, w = size

    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        return (int(w), mx)
    if w > mx:
        h = (float(mx) / float(w)) * h
        return (mx, int(h))
    return (h, w)

def calc_octave_resolution(initial_resolution, octave):
    '''
    Double in size every 2 octaves
    '''
    return int(initial_resolution * 1.415 ** (octave - 1))

class ImageLoader:

    def __init__(self,
                 max_dimsize,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''
        Arg max_dimsize corresponds to the largest pixel length allowed in any dimension of the
        image. If using 'octave' parameter, initialise max_dimsize to be the size of the first
        octave.
        '''
        self.max_dimsize = max_dimsize
        self.device = device

    def init_loaders(self, imsize):
        self.imsize = imsize
        self.loader = transforms.Compose(
            [transforms.Resize(imsize, interpolation=Image.BILINEAR),
             transforms.ToTensor()]  # scale imported image
        )  # transform it into a torch tensor
        self.unloader = transforms.ToPILImage()  # reconvert into PIL image

    def load_content_style_imgs(self, content_img_name, style_img_name, octave=1.0):
        '''
        Always couple loading content/style images, since they need to be the same size anyways
        '''
        content_image = Image.open(self.content_img_path(content_img_name))
        style_image = Image.open(self.style_img_path(style_img_name))

        # calculate octave resolution of max dimension size
        octave_resolution = calc_octave_resolution(self.max_dimsize, octave)

        # calculate image size, given max dimension size
        imsize = calc_img_size(content_image.size, octave_resolution)

        self.init_loaders(imsize)

        content_image = self.loader(content_image).unsqueeze(0)
        content_image = content_image.to(self.device, torch.float)

        # self.loader automatically resizes style image to the size of the content image
        # this is done to run content / style losses on the same execution graph
        style_image = self.loader(style_image).unsqueeze(0)
        style_image = style_image.to(self.device, torch.float)

        return content_image, style_image

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
