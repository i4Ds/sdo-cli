import logging
from pathlib import Path

import click
from sdo.cli import pass_environment

from sdo.data_loader.goes.cache import GOESCache


@click.command("download", short_help="Loads a the GOES X-Ray flux timeseries for a date range and stores it partitioned by year and month in a CSV")
@click.option("--output", default=".", type=click.Path(resolve_path=True, exists=False, path_type=Path))
@click.option("--start", default='2012-12-01T00:00:00', type=click.DateTime(), help="Start date")
@click.option("--end", default='2012-12-31T23:59:00', type=click.DateTime(), help="End date")
@pass_environment
def download(ctx, output, start, end):
    ctx.log("Starting to download GOES timeseries...")
    ctx.vlog(
        f"with options: target dir {output}, between {start} and {end}")
    goes_cache = GOESCache(output)
    goes_cache.fetch_goes_metadata(start, end, output)
