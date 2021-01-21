from __future__ import print_function

import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List
import copy
import os

from model import build_model_and_losses
import util

@dataclass
class StyleConfig:
    '''
    Style Transfer settings. One config per Styler instance.
    '''
    num_iters: int = 250

    # size of largest dimension in rendered output
    max_dimsize: int = 256

    # use average pooling in VGG-19 as opposed to max pooling
    use_avg_pool: bool = True

    # can take values of 'noise', 'content' or 'custom
    init_img_type: str = 'content'

    # weight parameters
    content_weight: int = 5e0
    style_weight: int = 1e6
    tv_weight: int = 1e-4

    # layer parameters
    content_layers: List[str] = field(default_factory=lambda: ["conv4_2"])
    style_layers: List[str] = field(default_factory=lambda:
                                    ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    style_layer_weights: List[int] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 8.0])

    def update(self, **kwargs):
        '''
        Convenient update function, due to the numerous arguments in StyleConfig.
        Please call Styler.update_config() if a Styler instance has already been initialized.
        '''
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Styler:
    '''
    Initialize with a StyleConfig instance.
    See style_single_image.py for an example of usage.
    '''

    # non modifiable configurations
    non_modifiable = ["use_avg_pool", "content_layers", "style_layers"]

    def __init__(self,
                 cfg: StyleConfig,
                 device):
        self._validate_config(cfg)
        self.cfg = cfg
        self.device = device
        self.loader = util.ImageLoader(device)

    def _validate_config(self, cfg: StyleConfig):
        assert len(cfg.style_layers) == len(cfg.style_layer_weights), (
            "style layer weights must correspond to style layers!")

    def _choose_init_img(self, content_img, init_img, dimsize):
        if self.cfg.init_img_type == "noise":
            return torch.randn(content_img.data.size(), device=self.device)
        elif self.cfg.init_img_type == "custom" and init_img != None:
            return self.loader.load(init_img, dimsize)
        elif self.cfg.init_img_type == "content":
            return self.loader.load(content_img, dimsize)
        else:
            raise Exception("Image initialization must be one of ['noise', 'custom', or 'content']."
                            + " You must provide an 'init_img' argument to 'style' if choosing the "
                            + "'custom' option.")

    def _style(self, content_img, style_img, init_img):
        '''
        accepts and outputs well-formatted pytorch tensors
        '''
        input_img = init_img

        print("Building the style transfer model..")
        model, style_losses, content_losses, tv_loss = build_model_and_losses(
            self.device,
            style_img,
            content_img,
            self.cfg.use_avg_pool,
            self.cfg.content_layers,
            self.cfg.style_layers,
        )
        optimizer = self.get_input_optimizer(input_img)

        print("Optimizing..")
        run = [0]
        while run[0] <= self.cfg.num_iters:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                tv_score = 0

                for sl, weight in zip(style_losses, self.cfg.style_layer_weights):
                    style_score += sl.loss * weight
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.cfg.style_weight
                content_score *= self.cfg.content_weight
                tv_score = self.cfg.tv_weight * tv_loss.loss

                loss = style_score + content_score + tv_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print(
                        "Style Loss : {:4f} Content Loss: {:4f} TV Loss: {:4f}".format(
                            style_score.item(), content_score.item(), tv_score.item()
                        )
                    )
                    print()

                return style_score + content_score + tv_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    def update_config(self, **kwargs):
        if any([key in non_modifiable for key, _ in kwargs]):
            raise Exception(f"Supplied one of following non_modifiable configurations: "
                            + "{non_modifiable}")
        self.cfg.update(kwargs)
        if "imsize" in kwargs.keys():
            self.loader = util.ImageLoader(self.cfg.imsize, self.device)

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def style(self, content_img, style_img, init_img=None):
        content, style = self.loader.load_content_style_imgs(content_img,
                                                             style_img,
                                                             self.cfg.max_dimsize)
        init = self._choose_init_img(content_img, init_img, self.cfg.max_dimsize)
        output = self._style(content, style, init)
        return self.loader.unload(output)

    def style_multiscale(self,
                         content_img,
                         style_img,
                         init_img=None,
                         octaves=3,
                         save_intermediate=False):
        '''
        Perform style transfer on images of iteratively greater size. Produces a more
        stable output for higher resolutions.
        '''

        for octave in range(octaves):
            dimsize = util.calc_octave_resolution(self.cfg.max_dimsize, octave, octaves)
            content, style = self.loader.load_content_style_imgs(content_img,
                                                                 style_img,
                                                                 dimsize)
            if octave == 0:
                init = self._choose_init_img(content_img, init_img, dimsize)
            else:
                _, _, h, w = content.size()
                init = F.interpolate(init, size=(h, w)).detach()
            init = self._style(content, style, init)
            if save_intermediate:
                util.imsave(self.loader.unload(init), f"multiscale_intermediate_{octave}")
        return self.loader.unload(init)
