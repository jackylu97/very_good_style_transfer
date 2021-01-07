from __future__ import print_function

import torch
import torch.optim as optim
import torchvision.models as models

from dataclasses import dataclass, field
from typing import List
import copy
import os

from model import build_model_and_losses

@dataclass
class StyleConfig:
    '''
    Style Transfer settings. One config per Styler instance.
    '''
    # pylint: disable=too-many-instance-attributes
    num_iters: int = 250
    use_avg_pool: bool = True
    content_weight: int = 5e0
    style_weight: int = 1e6
    tv_weight: int = 1e-4
    content_layers: List[str] = field(default_factory=lambda: ["conv4_2"])
    style_layers: List[str] = field(default_factory=lambda:
                                    ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    style_layer_weights: List[int] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 8.0])

    def update(self, **kwargs):
        '''
        Convenient update function, due to the numerous arguments in StyleConfig.
        '''
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Styler:
    '''
    Initialize with a StyleConfig instance.
    See style_single_image.py for an example of usage.
    '''
    def __init__(self, device, cfg):
        self._validate_config(cfg)
        self.cfg = cfg
        self.device = device

    def _validate_config(self, cfg):
        assert len(cfg.style_layers) == len(cfg.style_layer_weights), (
            "style layer weights must correspond to style layers!")

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def style(self, content_img, style_img, init_img=None):

        input_img = torch.randn(content_img.data.size(), device=self.device)
        if init_img != None:
            input_img = init_img.clone()
        # Comment next two lines to use random noise as input
        else:
            input_img = content_img.clone()

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
