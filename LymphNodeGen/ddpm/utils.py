import json
import numpy as np
from PIL import Image
from inspect import isfunction

import torch
from torchvision import transforms as T


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# trainer
def cycle(dl):
    while True:
        for data in dl:
            yield data


def identity(t, *args, **kwargs):
    return t


def log_function(dict, path):
    with open(path, "a+") as file:
        file.write(json.dumps(dict))
        file.write("\n")


CHANNELS_TO_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f"channels {channels} invalid"
    mode = CHANNELS_TO_MODE[channels]
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor (channels, frames, height, width) -> gif
def tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0  # to [0,1]
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


# gif -> tensor (channels, frames, height, width)
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


# diffusion
def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])