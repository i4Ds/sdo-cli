
import numpy as np
import datetime as dt
import os
from pathlib import Path

import pandas as pd
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
import logging

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GOES")


def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31).day
    last_ts = date.replace(month=date.month+1, day=1) - dt.timedelta(days=1)
    return last_ts.day


def get_date_ranges(start, end):
    # retrieve monthly ranges between start and end to not overload the HEK API
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    adjusted_start = dt.datetime(start.year, start.month, 1, 0, 0, 0)
    adjusted_end = dt.datetime(
        end.year, end.month, last_day_of_month(end), 23, 59, 59)

    start_dates = pd.date_range(start=adjusted_start, end=adjusted_end,
                                freq="MS").to_pydatetime().tolist()

    end_dates = pd.date_range(start=adjusted_start, end=adjusted_end,
                              freq="M").to_pydatetime().tolist()

    if len(start_dates) == 0:
        return [(adjusted_start, adjusted_end)]

    ranges = []
    for s, e in zip(start_dates, end_dates):
        ranges.append((s, e))

    if len(ranges) == 0:
        return [(adjusted_start, adjusted_end)]

    return ranges


def fetch_goes_metadata(start, end, path):
    # NOTE it can happen that sunpy will download invalid files during rate limiting, find them using grep -r "Too many" ~/sunpy/data/*
    # https://docs.sunpy.org/en/v3.0.0/generated/gallery/acquiring_data/goes_xrs_example.html
    # https://www.ngdc.noaa.gov/stp/satellite/goes-r.html
    # https://www.ngdc.noaa.gov/stp/satellite/goes/index.html

    # Since March 2020, data prior to GOES 15 (incl) is no longer supported by NOAA and GOES 16 and 17 data is now provided.
    # GOES 16 and 17 are part of the GOES-R series and provide XRS data at a better time resolution (1s).
    # GOES 16 has been taking observations from 2017, and GOES 17 since 2018
    for t_start, t_end in get_date_ranges(start, end):
        goes_dir = Path(path) / Path("goes_cache") / Path(str(t_start.year)) /\
            Path(str(t_start.month))
        goes_path = goes_dir / Path('goes_ts.csv')
        if not Path.exists(goes_path):
            try:
                logger.info(f"retrieving goes data for {goes_path}")
                os.makedirs(goes_dir, exist_ok=True)
                search_result = Fido.search(
                    a.Time(t_start, t_end), a.Instrument.xrs)

                logger.info(f"found {len(search_result)} search results")
                for result in search_result:
                    # https://docs.astropy.org/en/stable/table/operations.html#grouped-operations
                    grouped = result.group_by("Start Time")
                    rows_to_remove = np.array([], dtype=int)
                    for group in grouped.groups:
                        if len(group) > 1:
                            # only retain the observations for the newest known satellite
                            satellite_number = np.max(group["SatelliteNumber"])
                            remove_mask = group["SatelliteNumber"] != satellite_number
                            filtered_group = group[remove_mask]

                            for row in filtered_group:
                                indices_to_remove = np.where((result["Start Time"] == row["Start Time"]) & (
                                    result["SatelliteNumber"] == row["SatelliteNumber"]))
                                rows_to_remove = np.concatenate(
                                    (rows_to_remove, indices_to_remove), axis=None, dtype=int)

                    logger.info(
                        f"removing {len(rows_to_remove)} duplicate search results for older satellites")
                    result.remove_rows(rows_to_remove)

                download_result = Fido.fetch(search_result)
                goes_ts = ts.TimeSeries(download_result, concatenate=True)
                if isinstance(goes_ts, list) and len(goes_ts) > 0:
                    frames = []
                    for goes_ts_frm in goes_ts:
                        frames.append(goes_ts_frm.to_dataframe())
                    goes_ts_df = pd.concat(frames)
                else:
                    goes_ts_df = goes_ts.to_dataframe()
                goes_ts_df.index.name = 'timestamp'
                goes_ts_df.to_csv(goes_path, index=True,
                                  date_format=ISO_FORMAT)

                # TODO fix is within a leap second but datetime does not support leap seconds'),)  /Users/mariusgiger/repos/master/sdo-cli/tmp/goes_cache/2012/6/goes_ts.csv'
            except Exception as e:
                # it can happen that the downloaded hdf5 files are invalid. In this case run the ./check.sh command and remove the
                # corrupt files.
                # Usually the following error would appear: `Can't read data (wrong B-tree signature)`
                # also refer to the source code: sunpy/timeseries/sources/goes.py::_parse_netcdf

                logger.error(
                    f"could not download GOES series for {goes_path}", e)
                #raise e


def get_goes_at(at, cache_dir, max_diff=60):
    # https://github.com/sunpy/sunpy/blob/master/sunpy/timeseries/sources/goes.py
    # https://ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf

    goes_path = Path(cache_dir) / Path("goes_cache") / Path(str(at.year)) /\
        Path(str(at.month)) / Path('goes_ts.csv')

    if not os.path.exists(goes_path):
        raise Exception("please first cache GOES data before accessing it")

    goes_ts_df = pd.read_csv(goes_path)
    goes_ts_df["timestamp"] = pd.to_datetime(
        goes_ts_df['timestamp'], format=ISO_FORMAT)
    goes_ts_df = goes_ts_df.set_index('timestamp', drop=False)
    goes_ts_df = goes_ts_df.sort_index()
    goes_at_time = goes_ts_df.iloc[goes_ts_df.index.get_loc(
        at, method='nearest')]

    # NOAA have recently re-processed the GOES 13, 14 and 15 XRS science quality data,
    # such that the SWPC scaling factor has been removed. This means that no post processing is necessary anymore.
    # The sunpy GOES XRS client for Fido now provides this new re-processed data.
    # https://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/GOES_13-15_XRS_Science-Quality_Data_Readme.pdf
    # https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/docs/GOES-R_EXIS_XRS_L1b_Science-Quality_Data_ReadMe.pdf

    ts = goes_at_time["timestamp"]
    diff = (ts - at).total_seconds()
    # if diff is bigger than max_diff, raise an exception..
    if diff >= max_diff:
        raise Exception(
            f"goes diff too large, goes at {ts} wanted {at}, diff {diff}")

    return goes_at_time.to_dict()
