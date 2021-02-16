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


date_format = '%Y-%m-%dT%H%M%S'


def load_data(ctx, data_dir, start, end, freq='60min', aia_wave=AIA_WAVE.AIA_171, image_size=IMAGE_SIZE.P2000):
    """
    Loads a set of SDO images between start and end from the Georgia State University Data Lab API

    :param data_dir: the target directory, where images will be stored
    :param start: start time corresponding to the first image.
    :param end: end time corresponding to the last image.
    :param freq: the frequency (min interval 6min), also refer to: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#custom-frequency-ranges
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :param image_size: size of the output image (heatmap). Either use the class
           `constants.IMAGE_SIZE` to provide a valid size, or pass in a string from the list: [
           '2k', '512', '256']
    """
    if os.path.exists(data_dir) and not os.path.isdir(data_dir):
        raise ValueError(data_dir + " is not a directory")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dates = pd.date_range(start=start, end=end,
                          freq=freq).to_pydatetime().tolist()

    existing = set(Path(data_dir).rglob(f'*__{aia_wave}.jpeg'))
    ctx.log(f"loading data for {len(dates)} images between {start} and {end}")
    for date in dates:
        try:
            path = Path(data_dir) / \
                f"{date.strftime(date_format)}__{aia_wave}.jpeg"
            if path not in existing:
                ctx.vlog(f"loading image for date {date}")
                img = wrappers.get_aia_image_jpeg(date, aia_wave, image_size)
                img.save(path)
                # header = wrappers.get_aia_imageheader_json(date, aia_wave)
                # pprint.pprint(header)
            else:
                ctx.vlog(f"image for date {date} already present")
        except Exception as e:
            ctx.log(e)
            traceback.print_exc()
            pass

# NOTE m is no longer a valid alias for minute: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases


@click.command("download", short_help="Loads a set of SDO images between start and end from the Georgia State University Data Lab API")
@click.option("--path", default="./data", type=click.Path(resolve_path=True))
@click.option("--start", default='2012-12-01T00:00:00', type=click.DateTime(), help="Start date")
@click.option("--end", default='2012-12-24T23:59:00', type=click.DateTime(), help="End date")
@click.option("--freq", default='60min', type=str, help="Frequency (any multiple of 6m)")
@click.option("--wavelength", default='171', type=str, help="Allows to filter the files by wavelength. One of ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']")
@pass_environment
def download(ctx, path, start, end, freq, wavelength):
    """Loads a set of SDO images between start and end from the Georgia State University Data Lab API."""
    ctx.log("Starting to download images...")
    ctx.vlog(
        f"with options: target dir {path}, between {start} and {end} with freq {freq}")
    load_data(ctx, path, start, end, freq, wavelength)
