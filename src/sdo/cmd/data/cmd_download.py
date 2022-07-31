import os
import traceback
from pathlib import Path

import click
from tqdm import tqdm
import pandas as pd
from sdo.cli import pass_environment
from sdo.data_loader.image_param.api_wrappers import api_wrappers as wrappers
from sdo.data_loader.image_param.IP_CONSTANTS.CONSTANTS import *
from sdo.data_loader.image_param.utils.coord_convertor import *
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
import csv
import requests

date_format = '%Y-%m-%dT%H%M%S'


def load_data(ctx, data_dir, start, end, freq='60min', metadata=False, aia_wavelengths=[AIA_WAVE.AIA_171], image_size=IMAGE_SIZE.P2000, max_flux=None, min_flux=None):
    """
    Loads a set of SDO images between start and end from the Georgia State University Data Lab API

    :param data_dir: the target directory, where images will be stored
    :param start: start time corresponding to the first image.
    :param end: end time corresponding to the last image.
    :param freq: the frequency (min interval 6min), also refer to: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#custom-frequency-ranges
    :param aia_wave: wavelength channel of the image. Either use the class `constants.AIA_WAVE`
                     to provide a valid wavelength, or pass in a string from this list: ['94',
                     '131', '171', '193', '211', '304', '335', '1600', '1700']
    :param metadata: whether to also download image metadata (Default: False)
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

    meta_path = Path(data_dir) / "meta.csv"
    if metadata:
        # TODO only override if not exists
        with open(meta_path, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['QUALITY', 'DSUN', 'X0', 'R_SUN', 'Y0', 'CDELT', 'FILE_NAME'])
            writer.writeheader()
            ctx.log(f"writing metadata to {meta_path}")

    existing = set(Path(data_dir).rglob(f'*.jpeg'))
    ctx.log(
        f"loading data for {len(dates)*len(aia_wavelengths)} images between {start} and {end} for wavelengths {aia_wavelengths}")

    for date in tqdm(dates):
        for aia_wavelenth in aia_wavelengths:
            try:
                file_name = f"{date.strftime(date_format)}__{aia_wavelenth}.jpeg"
                path = Path(data_dir) / file_name

                if path not in existing:
                    if max_flux != None:
                        peak_flux = get_peak_flux_at_time(date)
                        if peak_flux >= max_flux:
                            ctx.vlog(
                                f"peak flux {format(peak_flux, '.17f')} is above max flux {format(max_flux, '.17f')} at {date}, skipping")
                            continue
                    if min_flux != None:
                        peak_flux = get_peak_flux_at_time(date)
                        if peak_flux <= min_flux:
                            ctx.vlog(
                                f"peak flux {format(peak_flux, '.17f')} is below min flux {format(min_flux, '.17f')} at {date}, skipping")
                            continue

                    ctx.vlog(
                        f"loading image for date {date} or within 10 minutes of that time")
                    img = wrappers.get_aia_image_jpeg(
                        date, aia_wavelenth, image_size)
                    img.save(path)
                else:
                    ctx.vlog(f"image for date {date} already present")

                if metadata:
                    header = wrappers.get_aia_imageheader_json(
                        date, aia_wavelenth)
                    header["file_name"] = file_name

                    with open(meta_path, 'a', encoding='utf-8') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=['QUALITY', 'DSUN', 'X0', 'R_SUN', 'Y0', 'CDELT', 'file_name'])
                        writer.writerow(header)
            except requests.exceptions.Timeout as e:
                # TODO implement retries? https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
                ctx.log(e)
                raise e
            except Exception as e:
                ctx.log(e)
                traceback.print_exc()
                pass


def get_peak_flux_at_time(datetime):
    # https://github.com/sunpy/sunpy/blob/master/sunpy/timeseries/sources/goes.py
    # https://ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf
    goes_long_key = 'xrsb'
    search_result = Fido.search(a.Time(datetime, datetime), a.Instrument.xrs)
    download_result = Fido.fetch(search_result)

    goes_ts = ts.TimeSeries(download_result)

    if isinstance(goes_ts, list):
        frames = []
        for goes_ts_frm in goes_ts:
            frames.append(goes_ts_frm.to_dataframe())
        goes_ts_df = pd.concat(frames)
        #goes_ts_df = goes_ts_df.loc[~goes_ts_df.index.duplicated(keep='first')]
    else:
        goes_ts_df = goes_ts.to_dataframe()
    goes_at_time = goes_ts_df.iloc[goes_ts_df.index.get_loc(
        datetime, method='nearest')][goes_long_key]
    print("found peak flux at time", str(goes_at_time))
    return goes_at_time


# NOTE m is no longer a valid alias for minute: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases


@click.command("download", short_help="Loads a set of SDO images between start and end from the Georgia State University Data Lab API")
@click.option("--path", default="./data", type=click.Path(resolve_path=True))
@click.option("--start", default='2012-12-01T00:00:00', type=click.DateTime(), help="Start date")
@click.option("--end", default='2012-12-31T23:59:00', type=click.DateTime(), help="End date")
@click.option("--freq", default='60min', type=str, help="Frequency (any multiple of 6m)")
@click.option("--metadata", is_flag=True, default=False, type=bool, help="A file with image metadata will be downloaded as <datadir>/meta.csv")
@click.option("--wavelength", default=['171'], multiple=True, type=str, help="Allows to filter the files by wavelength. One of ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']")
@click.option("--max-flux", default=None, type=float, help="Allows to filter by GOES x-ray max flux")
@click.option("--min-flux", default=None, type=float, help="Allows to filter by GOES x-ray min flux")
@pass_environment
def download(ctx, path, start, end, freq, metadata, wavelength, max_flux, min_flux):
    """Loads a set of SDO images between start and end from the Georgia State University Data Lab API."""
    ctx.log("Starting to download images...")
    ctx.vlog(
        f"with options: target dir {path}, between {start} and {end} with freq {freq}")
    load_data(ctx, path, start, end, freq, metadata,
              aia_wavelengths=wavelength, max_flux=max_flux, min_flux=min_flux)

# TODO allow loading image parameters
