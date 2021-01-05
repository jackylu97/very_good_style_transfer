import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# desired depth layers to compute style/content losses :
content_layers_default = ["conv4_2"]
style_layers_default = ['relu1_1', 'relu2_1',
                        'relu3_1', 'relu4_1', 'relu5_1']

class ContentLoss(nn.Module):
    def __init__(
            self, target,
        ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, input):
        b, c, h, w = input.size()
        w_variance = torch.sum(torch.pow(input[:, :, :, :-1] - input[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(input[:, :, :-1, :] - input[:, :, 1:, :], 2))
        self.loss = (w_variance + h_variance) / (b * c * h * w) ** 2
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def build_model_and_losses(
        device,
        style_img,
        content_img,
        use_avg_pool = True,
        content_layers=content_layers_default,
        style_layers=style_layers_default,
    ):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

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
            if use_avg_pool:
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
