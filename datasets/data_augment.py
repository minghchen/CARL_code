import math
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

def flip(x, dim=-1):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def resize(images, size):
    return torch.nn.functional.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

def uniform_crop(images, size, spatial_idx=1):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped

def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order RGB.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = images.clone()
    gray_channel = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0, img_hue=0, img_blur=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    order = np.random.permutation(np.arange(4))
    b_var = 1 + random.uniform(-img_brightness, img_brightness)
    c_var = 1 + random.uniform(-img_contrast, img_contrast)
    s_var = 1 + random.uniform(-img_saturation, img_saturation)
    h_var = random.uniform(-img_hue, img_hue)
    if img_blur == 0:
        g_apply = False
    else:
        g_var = random.uniform(0.1, 2.0)
        g_apply = random.uniform(0,1) < img_blur
    for i in range(len(images)):
        img = transforms.functional.to_pil_image(images[i])
        for idx in range(4):
            if order[idx] == 0:
                img = transforms.functional.adjust_brightness(img, b_var)
            elif order[idx] == 1:
                img = transforms.functional.adjust_contrast(img, c_var)
            elif order[idx] == 2:
                img = transforms.functional.adjust_saturation(img, s_var)
            elif order[idx] == 3:
                img = transforms.functional.adjust_hue(img, h_var)
            if g_apply:
                img = img.filter(ImageFilter.GaussianBlur(radius=g_var))
        images[i] = transforms.functional.to_tensor(img)
    return images


def brightness_jitter(images, var=0):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + random.uniform(-var, var)

    for i in range(len(images)):
        img = transforms.functional.to_pil_image(images[i])
        img = transforms.functional.adjust_brightness(img, alpha)
        images[i] = transforms.functional.to_tensor(img)

    return images


def contrast_jitter(images, var=0):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + random.uniform(-var, var)

    for i in range(len(images)):
        img = transforms.functional.to_pil_image(images[i])
        img = transforms.functional.adjust_contrast(img, alpha)
        images[i] = transforms.functional.to_tensor(img)

    return images


def saturation_jitter(images, var=0):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + random.uniform(-var, var)
    for i in range(len(images)):
        img = transforms.functional.to_pil_image(images[i])
        img = transforms.functional.adjust_saturation(img, alpha)
        images[i] = transforms.functional.to_tensor(img)

    return images

def hue_jitter(images, var=0):
    """
    Perfrom hue jittering on the input images. The channels of images
    should be in order RGB.
    Args:
        var (float): jitter ratio for hue.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = random.uniform(-var, var)
    for i in range(len(images)):
        img = transforms.functional.to_pil_image(images[i])
        img = transforms.functional.adjust_hue(img, alpha)
        images[i] = transforms.functional.to_tensor(img)

    return images


def color_normalization(images, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225]):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.
    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1], "channel mean not computed properly"
    assert (
        len(stddev) == images.shape[1]
    ), "channel stddev not computed properly"

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]

    return out_images


def _get_param_spatial_crop(scale, ratio, height, width):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(10):
        area = height * width
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.8 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.
    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    result = torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False)
    return result

class AugmentOp:
    """
    Apply for video.
    """
    def __init__(self, aug_fn, *args, **kwargs):
        self.aug_fn = aug_fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, images):
        return self.aug_fn(images, *self.args, **self.kwargs)

class RandomOp:
    """
    Apply for video.
    """
    def __init__(self, aug_fn, prob, *args, **kwargs):
        self.aug_fn = aug_fn
        self.prob = prob
        self.args = args
        self.kwargs = kwargs

    def __call__(self, images):
        if random.uniform(0,1) < self.prob:
            images = self.aug_fn(images, *self.args, **self.kwargs)
        return images

class ComposeOp:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, img_list):
        for op in self.ops:
            # start_time = time.time()
            img_list = op(img_list)
            # if torch.cuda.device_count() == 1:
            #     print(f"op time: {time.time()-start_time:.3f}")
        return img_list

def create_ssl_data_augment(cfg, augment):
    ops = []
    if augment:
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        s = cfg.AUGMENTATION.STRENGTH
        ops.append(AugmentOp(random_resized_crop, **{
            'target_height': cfg.IMAGE_SIZE,
            'target_width': cfg.IMAGE_SIZE
        }))
        ops.append(RandomOp(flip, 0.5))
        ops.append(RandomOp(color_jitter, 0.8, **{
            'img_brightness': 0.8*s,
            'img_contrast': 0.8*s,
            'img_saturation': 0.8*s,
            'img_hue': 0.2*s,
            'img_blur': 0.0 if cfg.DATASETS[0] == "finegym" else 0.5*s,
        }))
        ops.append(RandomOp(grayscale, 0.2))
    else:
        ops.append(AugmentOp(uniform_crop, **{
            'size': cfg.IMAGE_SIZE
        }))
    ops.append(AugmentOp(resize, **{
        'size': cfg.IMAGE_SIZE
    }))
    ops.append(AugmentOp(color_normalization, **{
        "mean" : [0.485, 0.456, 0.406],
        "stddev": [0.229, 0.224, 0.225]
    }))
    return ComposeOp(ops)

def create_data_augment(cfg, augment):
    ops = []
    if augment:
        if cfg.AUGMENTATION.BRIGHTNESS:
            ops.append(AugmentOp(brightness_jitter, **{
                'var': cfg.AUGMENTATION.BRIGHTNESS_MAX_DELTA,
            }))
        if cfg.AUGMENTATION.CONTRAST:
            ops.append(AugmentOp(contrast_jitter, **{
                'var': cfg.AUGMENTATION.CONTRAST_MAX_DELTA,
            }))
        if cfg.AUGMENTATION.HUE:
            ops.append(AugmentOp(hue_jitter, **{
                'var': cfg.AUGMENTATION.HUE_MAX_DELTA,
            }))
        if cfg.AUGMENTATION.SATURATION:
            ops.append(AugmentOp(saturation_jitter, **{
                'var': cfg.AUGMENTATION.SATURATION_MAX_DELTA,
            }))
        if cfg.AUGMENTATION.RANDOM_CROP:
            ops.append(AugmentOp(random_resized_crop, **{
                'target_height': cfg.IMAGE_SIZE,
                'target_width': cfg.IMAGE_SIZE
            }))
        if cfg.AUGMENTATION.RANDOM_FLIP:
            ops.append(RandomOp(flip, 0.5))
    else:
        if cfg.AUGMENTATION.RANDOM_CROP:
            ops.append(AugmentOp(uniform_crop, **{
                'size': cfg.IMAGE_SIZE
            }))
    ops.append(AugmentOp(resize, **{
        'size': cfg.IMAGE_SIZE
    }))
    ops.append(AugmentOp(color_normalization, **{
        "mean" : [0.485, 0.456, 0.406],
        "stddev": [0.229, 0.224, 0.225]
    }))
    return ComposeOp(ops)