## very_good_style_transfer

This is a pytorch implementation of the Neural Style algorithm. This implementation is meant to be modular and enforce abstractions to easily programmatically perform style transfer with other operations. Other libraries I've used opt for command line tool implementations, which make it cumbersome to write scripts that involve multiple style transfers, for example.

Builds off of example code from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

### Quickstart 
Style an image in ~10 lines of code (see `style_single_image.py`):
```python
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
- improve PIL -> torch interface; fold that logic into Styler instead of calling Util directly
- figure out looping zoom ? 