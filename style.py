from __future__ import print_function

import torch
import torch.optim as optim
import torchvision.models as models

import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List
import copy
import os

import util
from model import build_model_and_losses

# TODO: add layer specific weights
# TODO: add command line flags (?)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu


CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
OUTPUT_NAME = "test"

@dataclass
class StyleConfig:
    num_iters: int = 250
    use_avg_pool: bool = True
    content_weight: int = 5e0
    style_weight: int = 1e6
    tv_weight: int = 5e5
    content_layers: List[str] = field(default_factory=lambda: ["conv4_2"])
    style_layers: List[str] = field(default_factory=lambda:
                                    ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])


class Styler:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    
    def style(self, content_img, style_img):
        input_img = content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        # input_img = torch.randn(content_img.data.size(), device=device)

        print("Building the style transfer model..")
        model, style_losses, content_losses, tv_loss = build_model_and_losses(
            device,
            style_img,
            content_img,
            self.cfg.use_avg_pool,
            self.cfg.content_layers,
            self.cfg.style_layers
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

                for sl in style_losses:
                    style_score += sl.loss
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


loader = util.ImageLoader(imsize, device)
content_img = loader.load_content_img(CONTENT_IMG_NAME)
style_img = loader.load_style_img(STYLE_IMG_NAME)

default_config = StyleConfig()
styler = Styler(default_config)

output = styler.style(
    content_img,
    style_img,
)

plt.figure()
loader.imsave(output, name=OUTPUT_NAME)
