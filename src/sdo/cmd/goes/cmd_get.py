import logging
from pathlib import Path

import click
from sdo.cli import pass_environment

from sdo.data_loader.goes.cache import GOESCache


@click.command("get", short_help="Gets a the GOES flux at a point in time")
@click.option("--cache-dir", default=".", type=click.Path(resolve_path=True, exists=False, path_type=Path))
@click.option("--timestamp", default='2012-12-01T00:00:00', type=click.DateTime(), help="Start date")
@pass_environment
def get(ctx, cache_dir, timestamp):
    ctx.log(f"getting goes flux at: {timestamp}")
    ctx.vlog(
        f"with options: cache dir {cache_dir}")
    goes_cache = GOESCache(cache_dir)
    goes = goes_cache.get_goes_at(timestamp)
    ctx.log(f"found goes {goes}")
