from sdo.cli import pass_environment

import click

import cv2

import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import pprint
import os
from pathlib import Path
import pandas as pd
import traceback

from sdo.data_loader.image_param.IP_CONSTANTS.CONSTANTS import *
from sdo.data_loader.image_param.api_wrappers import api_wrappers as wrappers
from sdo.data_loader.image_param.objects.spatiotemporal_event import SpatioTemporalEvent
from sdo.data_loader.image_param.utils.coord_convertor import *
from sklearn.feature_extraction import image

date_format = '%Y-%m-%dT%H:%M:%S'


def image_patches(ctx, data_dir, patch_dir, patch_size=256, max_patches=64, aia_wave=AIA_WAVE.AIA_171, crop_size=None):
    """
    Loads a set of SDO images between start and end from the Georgia State University Data Lab API

    :param data_dir: the target directory, where images will be stored
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    """
    if os.path.exists(data_dir) and not os.path.isdir(data_dir):
        raise ValueError(data_dir + " is not a directory")

    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    images = list(Path(data_dir).rglob(f'*__{aia_wave}.jpeg'))
    ctx.log(f"converting {len(images)} images")

    for path in images:
        try:
            im = np.array(Image.open(path))
            w, h, c = im.shape
            if crop_size is not None and w > crop_size and h > crop_size:
                im = im[crop_size:w-crop_size, crop_size:h-crop_size]
            # TODO find a way to compute overlapping patches
            patches = image.extract_patches_2d(
                im, (patch_size, patch_size), random_state=0, max_patches=max_patches)
            for i, patch in enumerate(patches):
                # TODO add some logic for evaluating if the image patch is a valid patch
                path_patch = Path(patch_dir) / f"{path.stem}__{i}.jpeg"
                img_patch = Image.fromarray(patch)
                img_patch.save(path_patch)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return


@click.command("patch", short_help="Generates patches from a set of images")
@click.option("--path", default="./data", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--targetpath", default="./data_patches/train/aia", type=click.Path(file_okay=False, resolve_path=True))
@click.option("--size", default=256, type=int)
@click.option("--max-patches", default=64, type=int)
@click.option("--wavelength", default='*', type=str, help="Allows to filter the files by wavelength. One of ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']")
@click.option("--crop-size", default=None, type=int, help="allows cropping the input image by crop size on each side")
@pass_environment
def patch(ctx, path, targetpath, size, max_patches, wavelength, crop_size):
    """Loads a set of SDO images between start and end from the Georgia State University Data Lab API."""
    ctx.log("Starting to generate patches...")
    ctx.vlog(
        f"with options: source dir {path}, target dir {targetpath}, wavelength {wavelength}, crop size {crop_size}")
    image_patches(ctx, path, targetpath, size,
                  max_patches, wavelength, crop_size)
