
from datetime import timedelta
from multiprocessing.sharedctypes import Value
from dateutil import parser
import dask.dataframe as dd
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


class GOESCache:
    def __init__(
            self,
            storage_path: str):

        self.cache_dir = Path(storage_path) / Path("goes_cache")
        self.parquet_dir = self.cache_dir / 'goes_ts.parquet'
        self.ddf_cached = None

    def fetch_goes_metadata(self, start, end, path):
        # NOTE it can happen that sunpy will download invalid files during rate limiting, find them using grep -r "Too many" ~/sunpy/data/*
        # https://docs.sunpy.org/en/v3.0.0/generated/gallery/acquiring_data/goes_xrs_example.html
        # https://www.ngdc.noaa.gov/stp/satellite/goes-r.html
        # https://www.ngdc.noaa.gov/stp/satellite/goes/index.html

        # Since March 2020, data prior to GOES 15 (incl) is no longer supported by NOAA and GOES 16 and 17 data is now provided.
        # GOES 16 and 17 are part of the GOES-R series and provide XRS data at a better time resolution (1s).
        # GOES 16 has been taking observations from 2017, and GOES 17 since 2018

        for t_start, t_end in get_date_ranges(start, end):
            goes_dir = self.cache_dir / Path(str(t_start.year)) /\
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
                                satellite_number = np.max(
                                    group["SatelliteNumber"])
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

                    # NOTE sunpy is not dropping data with leap seconds therefore the following error can occur
                    # "XY is within a leap second but datetime does not support leap seconds"
                    # this can be fixed by changing the following code in
                    # https://github.com/sunpy/sunpy/blob/6f586392f9799383017e0566d4303928183c06be/sunpy/timeseries/sources/goes.py#L288
                    #
                    # not_leap_seconds = np.char.find(times.iso, ":60.") == -1
                    # times = times[not_leap_seconds]
                    # data = DataFrame({"xrsa": xrsa[not_leap_seconds], "xrsb": xrsb[not_leap_seconds]}, index=times.datetime)
                    #
                    # also refer to: https://github.com/sunpy/sunpy/issues/5422

                except Exception as e:
                    # it can happen that the downloaded hdf5 files are invalid. In this case run the ./check.sh command and remove the
                    # corrupt files.
                    # Usually the following error would appear: `Can't read data (wrong B-tree signature)`
                    # also refer to the source code: sunpy/timeseries/sources/goes.py::_parse_netcdf

                    logger.error(
                        f"could not download GOES series for {goes_path}", e)
                    #raise e
            else:
                logger.debug(
                    f"skipping downloading data for {str(goes_path)} as it is already there..")

        # convert to parquet format to read with dask
        # https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.repartition.html
        # https://docs.dask.org/en/stable/dataframe-best-practices.html
        if not os.path.exists(self.parquet_dir):
            ddf = dd.read_csv(str(self.cache_dir) + '/**/*.csv')
            ddf["timestamp"] = dd.to_datetime(
                ddf['timestamp'], format=ISO_FORMAT)
            ddf = ddf.set_index('timestamp', drop=True)
            ddf = ddf.repartition(freq='7d')
            ddf.to_parquet(self.parquet_dir, overwrite=True)
        else:
            logger.info(
                "skipping conversion to parquet as it is already converted")

    def get_goes_at(self, at, max_diff=60):
        # NOAA have recently re-processed the GOES 13, 14 and 15 XRS science quality data,
        # such that the SWPC scaling factor has been removed. This means that no post processing is necessary anymore.
        # The sunpy GOES XRS client for Fido now provides this new re-processed data.
        # https://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/GOES_13-15_XRS_Science-Quality_Data_Readme.pdf
        # https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/docs/GOES-R_EXIS_XRS_L1b_Science-Quality_Data_ReadMe.pdf
        # https://github.com/sunpy/sunpy/blob/master/sunpy/timeseries/sources/goes.py
        # https://ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf

        if not os.path.exists(self.parquet_dir):
            raise Exception(
                f"please first cache GOES data before accessing it {self.parquet_dir}")
        if self.ddf_cached is None:
            ddf = dd.read_parquet(self.parquet_dir, engine="pyarrow",
                                  calculate_divisions=True)
            ddf = ddf[(ddf["quality_xrsb"] == 0)]
            self.ddf_cached = ddf
        else:
            ddf = self.ddf_cached

        if type(at) is str:
            at = parser.parse(at)
        search_dt_start = at - timedelta(seconds=max_diff)
        search_dt_end = at + timedelta(seconds=max_diff)

        result = ddf.loc[search_dt_start:search_dt_end].compute()
        if result.size == 0:
            raise ValueError(
                f"no goes value found at {at} within +/- {max_diff}")
        result.reset_index(inplace=True)
        idx = result.timestamp.subtract(at).abs().idxmin()
        goes_at_time = result.iloc[idx]

        return goes_at_time.to_dict()
