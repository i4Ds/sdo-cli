from matplotlib.colors import LinearSegmentedColormap
from sunpy.visualization.colormaps import cm
from munch import DefaultMunch
import json
import random
import warnings

import numpy as np
import torch
from torchvision.utils import save_image as tv_save_image, make_grid
import os
from pathlib import Path
from PIL import Image

import yaml


def get_vanilla_image_gradient(model, inpt, err_fn, abs=False):
    if isinstance(model, torch.nn.Module):
        model.zero_grad()
    inpt = inpt.detach()
    inpt.requires_grad = True

    # output = model(inpt)

    err = err_fn(inpt)
    err.backward()

    grad = inpt.grad.detach()

    if isinstance(model, torch.nn.Module):
        model.zero_grad()

    if abs:
        grad = torch.abs(grad)
    return grad.detach()


def get_guided_image_gradient(model: torch.nn.Module, inpt, err_fn, abs=False):
    def guided_relu_hook_function(module, grad_in, grad_out):
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            return (torch.clamp(grad_in[0], min=0.0),)

    model.zero_grad()

    # Apply hooks
    hook_ids = []
    for mod in model.modules():
        hook_id = mod.register_backward_hook(guided_relu_hook_function)
        hook_ids.append(hook_id)

    inpt = inpt.detach()
    inpt.requires_grad = True

    err = err_fn(inpt)
    err.backward()

    grad = inpt.grad.detach()

    model.zero_grad()
    for hooks in hook_ids:
        hooks.remove()

    if abs:
        grad = torch.abs(grad)
    return grad.detach()


def get_smooth_image_gradient(model, inpt, err_fn, abs=True, n_runs=20, eps=0.1,  grad_type="vanilla"):
    grads = []
    for i in range(n_runs):
        inpt = inpt + torch.randn(inpt.size()).to(inpt.device) * eps
        if grad_type == "vanilla":
            single_grad = get_vanilla_image_gradient(
                model, inpt, err_fn, abs=abs)
        elif grad_type == "guided":
            single_grad = get_guided_image_gradient(
                model, inpt, err_fn, abs=abs)
        else:
            warnings.warn("This grad_type is not implemented yet")
            single_grad = torch.zeros_like(inpt)
        grads.append(single_grad)

    grad = torch.mean(torch.stack(grads), dim=0)
    return grad.detach()


def get_input_gradient(model, inpt, err_fn, grad_type="vanilla", n_runs=20, eps=0.1,
                       abs=False, results_fn=lambda x, *y, **z: None):
    """
    Given a model creates calculates the error and backpropagates it to the image and saves it (saliency map).

    Args:
        model: The model to be evaluated
        inpt: Input to the model
        err_fn: The error function the evaluate the output of the model on
        grad_type: Gradient calculation method, currently supports (vanilla, vanilla-smooth, guided,
        guided-smooth) ( the guided backprob can lead to segfaults -.-)
        n_runs: Number of runs for the smooth variants
        eps: noise scaling to be applied on the input image (noise is drawn from N(0,1))
        abs (bool): Flag, if the gradient should be a absolute value
        results_fn: function which is called with the results/ return values. Expected f(grads)

    """
    model.zero_grad()

    if grad_type == "vanilla":
        grad = get_vanilla_image_gradient(model, inpt, err_fn, abs)
    elif grad_type == "guided":
        grad = get_guided_image_gradient(model, inpt, err_fn, abs)
    elif grad_type == "smooth-vanilla":
        grad = get_smooth_image_gradient(
            model, inpt, err_fn, abs, n_runs, eps, grad_type="vanilla")
    elif grad_type == "smooth-guided":
        grad = get_smooth_image_gradient(
            model, inpt, err_fn, abs, n_runs, eps, grad_type="guided")
    else:
        warnings.warn("This grad_type is not implemented yet")
        grad = torch.zeros_like(inpt)
    model.zero_grad()

    results_fn(grad)

    return grad


def update_model(original_model, update_dict, exclude_layers=(), do_warnings=True):
    # also allow loading of partially pretrained net
    model_dict = original_model.state_dict()

    # 1. Give warnings for unused update values
    unused = set(update_dict.keys()) - \
        set(exclude_layers) - set(model_dict.keys())
    not_updated = set(model_dict.keys()) - \
        set(exclude_layers) - set(update_dict.keys())
    if do_warnings:
        for item in unused:
            warnings.warn("Update layer {} not used.".format(item))
        for item in not_updated:
            warnings.warn("{} layer not updated.".format(item))

    # 2. filter out unnecessary keys
    update_dict = {k: v for k, v in update_dict.items() if
                   k in model_dict and k not in exclude_layers}

    # 3. overwrite entries in the existing state dict
    model_dict.update(update_dict)

    # 4. load the new state dict
    original_model.load_state_dict(model_dict)


def set_seed(seed):
    """Sets the seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model, model_file, exclude_layers=(), warnings=True):
    """
    Loads a pytorch model from a given directory (using pytorch)

    Args:
        model: The model to be loaded (whose parameters should be restored)
        model_file: The file from which the model parameters should be loaded
        exclude_layers: List of layer names which should be excluded from restoring
        warnings (bool): Flag which indicates if method should warn if not everything went perfectly

    """

    if os.path.exists(model_file):
        pretrained_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        update_model(model, pretrained_dict, exclude_layers, warnings)
        return model
    else:
        raise IOError("Model file does not exist!")


def tensor_to_image(tensor, **kwargs):
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


class PathJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)

        # Call the default method for other types
        return json.JSONEncoder.default(self, obj)


def save_image_grid(tensor, name, save_dir, n_iter=None, prefix=False, iter_format="{:05d}", image_args=None):
    """
    Saves images of a 4d- tensor (N, C, H, W) as a image grid into an image file in the image directory of the
    PytorchPlotFileLogger

    Args:
        tensor: 4d- tensor (N, C, H, W)
        name: file-name of the image file
        n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
        iter_format: The format string, which indicates how n_iter will be formated as a string
        prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
        image_args: Arguments for the tensorvision save image method

    """

    if image_args is None:
        image_args = {}

    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)

    if not (tensor.size(1) == 1 or tensor.size(1) == 3):
        warnings.warn("The 1. dimension (channel) has to be either 1 (gray) or 3 (rgb), taking the first "
                      "dimension now !!!")
        tensor = tensor[:, 0:1, ]

    if n_iter is not None:
        name = name_and_iter_to_filename(name=name, n_iter=n_iter, extension=".png", iter_format=iter_format,
                                         prefix=prefix)
    elif not name.endswith(".png"):
        name += ".png"

    img_file = save_dir / Path(name)

    if image_args is None:
        image_args = {}

    os.makedirs(os.path.dirname(img_file), exist_ok=True)

    tv_save_image(tensor, img_file, **image_args)
    return str(img_file)


def name_and_iter_to_filename(name, n_iter, extension, iter_format="{:05d}", prefix=False):
    iter_str = iter_format.format(n_iter)
    if prefix:
        name = iter_str + "_" + name + extension
    else:
        name = name + "_" + iter_str + extension

    return name


def save_model(model, name, model_dir, n_iter=None, iter_format="{:05d}", prefix=False):
    """
    Saves a pytorch model in the model directory of the experiment folder

    Args:
        model: The model to be stored
        name: The file name of the model file
        n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
        iter_format: The format string, which indicates how n_iter will be formated as a string
        prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix

    """

    if n_iter is not None:
        name = name_and_iter_to_filename(name,
                                         n_iter,
                                         ".pth",
                                         iter_format=iter_format,
                                         prefix=prefix)

    if not name.endswith(".pth"):
        name += ".pth"

    model_file = os.path.join(model_dir, name)
    torch.save(model.state_dict(), model_file)


def merge_config(user: dict, default: dict) -> dict:
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_config(user[k], v)
    return user


def read_config(config_file: Path, overrides: dict = None) -> dict:
    config_file = Path(os.path.expanduser(config_file))
    with open(config_file, "r") as f:
        user_config = yaml.safe_load(f)

    default_config_file = config_file.parent / Path("defaults.yaml")
    if os.path.exists(default_config_file):
        with open(default_config_file, "r") as f:
            default_config = yaml.safe_load(f)
            user_config = merge_config(user_config, default_config)

    if overrides:
        user_config = merge_config(overrides, user_config)

    # Hack to convert dict to object
    return DefaultMunch.fromDict(user_config)


# Channels that correspond to HMI Magnetograms
HMI_WL = ['Bx', 'By', 'Bz']

# A colormap for visualizing HMI
HMI_CM = LinearSegmentedColormap.from_list(
    "bwrblack", ["#0000ff", "#000000", "#ff0000"])


def channel_to_map(name):
    """Given channel name, return colormap"""
    return HMI_CM if name in HMI_WL else cm.cmlist.get('sdoaia%d' % int(name))


def colorize_sdo(X, channel=171):
    """Given image and channel, visualize results"""
    cm = channel_to_map(channel)
    Xcv = cm(X)
    return (Xcv[:, :, :3]*255).astype(np.uint8)
