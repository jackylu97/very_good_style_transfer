from __future__ import print_function

import torch
import torch.optim as optim


import matplotlib.pyplot as plt

import torchvision.models as models

import copy
import os
import util
# TODO: fix wildcard import later
from model import *

# TODO: add layer specific weights
# TODO: add command line flags (?)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu


CONTENT_IMG_NAME = "lion.jpg"
STYLE_IMG_NAME = "starry-night.jpg"
NUM_ITERS = 300
OUTPUT_NAME = "test"
USE_AVG_POOL = True
TV_LOSS = 5e5

loader = util.ImageLoader(imsize, device)
content_img = loader.load_content_img(CONTENT_IMG_NAME)
style_img = loader.load_style_img(STYLE_IMG_NAME)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# desired depth layers to compute style/content losses :
content_layers_default = ["conv4_2"]
style_layers_default = ['relu1_1', 'relu2_1',
                        'relu3_1', 'relu4_1', 'relu5_1']


def get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_img,
        content_img,
        content_layers=content_layers_default,
        style_layers=style_layers_default,
    ):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []
    tv_loss = None

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    tv_loss_module = TotalVariationLoss()
    model.add_module("total_variation_loss", tv_loss_module)
    tv_loss = tv_loss_module

    i = 1 # increment every time we see a conv
    j = 0 # increment every time we see a pool
    style_ind = 0
    content_ind = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            j += 1
            name = "conv{}_{}".format(i, j)
        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(i, j)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            if USE_AVG_POOL:
                layer = nn.AvgPool2d(layer.kernel_size)
            name = "pool{}".format(i)
            i += 1
            j = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn{}_{}".format(i,j)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(content_ind), content_loss)
            content_losses.append(content_loss)
            content_ind += 1

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(style_ind), style_loss)
            style_losses.append(style_loss)
            style_ind += 1

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses, tv_loss


input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(
        cnn,
        normalization_mean,
        normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        style_weight=1e6,
        content_weight=5e0,
        tv_weight=1e4
    ):
    """Run the style transfer."""
    print("Building the style transfer model..")
    model, style_losses, content_losses, tv_loss = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )
    optimizer = get_input_optimizer(input_img)

    print("Optimizing..")
    run = [0]
    while run[0] <= num_steps:

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

            style_score *= style_weight
            content_score *= content_weight
            tv_score = tv_weight * tv_loss.loss

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


output = run_style_transfer(
    cnn,
    cnn_normalization_mean,
    cnn_normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=NUM_ITERS,
    tv_weight=TV_LOSS
)

plt.figure()
loader.imsave(output, name=OUTPUT_NAME)
