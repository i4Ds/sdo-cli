import logging
from pathlib import Path

import click
import pandas as pd
from sdo.cli import pass_environment
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

date_format = '%Y-%m-%d'


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GOES")


# retrying because of some unreliable results from the API, see https://github.com/jd/tenacity
# @retry(stop=stop_after_attempt(5), reraise=True, before=before_log(logger, logging.INFO), wait=wait_fixed(5))
def download_flux_search_results(search_result):
    download_result = Fido.fetch(search_result)
    goes_ts = ts.TimeSeries(download_result)

    if isinstance(goes_ts, list):
        frames = []
        for goes_ts_frm in goes_ts:
            frames.append(goes_ts_frm.to_dataframe())
        return pd.concat(frames)

    return goes_ts.to_dataframe()


def get_goes_flux(start, end):
    # https://github.com/sunpy/sunpy/blob/master/sunpy/timeseries/sources/goes.py
    # https://ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf
    # https://docs.sunpy.org/en/stable/generated/gallery/acquiring_data/goes_xrs_example.html#sphx-glr-generated-gallery-acquiring-data-goes-xrs-example-py

    search_result = Fido.search(a.Time(start, end), a.Instrument.xrs)
    return download_flux_search_results(search_result)


@click.command("download", short_help="Loads a the GOES X-Ray flux timeseries for a date range and stores it in a CSV")
@click.option("--out-dir", default=".", type=click.Path(resolve_path=True, exists=True))
@click.option("--start", default='2012-12-01T00:00:00', type=click.DateTime(), help="Start date")
@click.option("--end", default='2012-12-31T23:59:00', type=click.DateTime(), help="End date")
@pass_environment
def download(ctx, out_dir, start, end):
    ctx.log("Starting to download GOES timeseries...")
    ctx.vlog(
        f"with options: target dir {out_dir}, between {start} and {end}")
    goes_df = get_goes_flux(start, end)
    goes_df.index.name = "timestamp"
    goes_df.to_csv(
        out_dir / Path(f"goes_{start.strftime(date_format)}-{end.strftime(date_format)}.csv"))
