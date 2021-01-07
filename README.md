## very_good_style_transfer

This is a pytorch implementation of the Neural Style algorithm. This implementation is meant to be lightweight and create favorable abstractions towards implementing more complex operations involving style transfer. 

Builds off of example code from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

### Quickstart 
Style an image in ~10 lines of code (see `style_single_image.py`):
```
# desired size of the output image
imsize = 512

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = util.ImageLoader(imsize, device)
content_img, style_img = loader.load_content_style_imgs("lion.jpg", "starry-night.jpg")

default_config = StyleConfig()
styler = Styler(device, default_config)

output = styler.style(
    content_img,
    style_img,
)

loader.imsave(output, name="styled")
```

TODO:
- add ability to use multiple style images with weights
- add option to normalize weights
- add option to normalize gradients
- add script that implements multi-scale rendering
- callback system to add intermediate computations between style transfer iterations
- basic image/canvas manipulation commands
- figure out looping zoom ? 