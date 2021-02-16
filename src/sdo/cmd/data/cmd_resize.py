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


def image_resize(ctx, data_dir, out_dir, aia_wave=AIA_WAVE.AIA_171, size=32):
    if os.path.exists(data_dir) and not os.path.isdir(data_dir):
        raise ValueError(data_dir + " is not a directory")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images = list(Path(data_dir).rglob(f'*__{aia_wave}.jpeg'))
    ctx.log(f"converting {len(images)} images")

    for path in images:
        try:
            img = Image.open(path)
            img = img.resize((size, size), Image.ANTIALIAS)
            img.save(Path(out_dir) / path.name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return


@click.command("resize", short_help="Generates a set of resized images")
@click.option("--path", default="./data", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--targetpath", default="./data_resized/train/aia", type=click.Path(file_okay=False, resolve_path=True))
@click.option("--wavelength", default='*', type=str, help="Allows to filter the files by wavelength. One of ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']")
@click.option("--size", default=32, type=int, help="Size of the image, will be converzed to size x size")
@pass_environment
def resize(ctx, path, targetpath, wavelength, size):
    ctx.log("Starting to generate resized images...")
    ctx.vlog(
        f"with options: source dir {path}, target dir {targetpath}, wavelength {wavelength}")
    image_resize(ctx, path, targetpath, wavelength, size)
