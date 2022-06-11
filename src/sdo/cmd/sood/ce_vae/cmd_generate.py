from sdo.cli import pass_environment
from sdo.sood.algorithms.ce_vae import main
from pathlib import Path

import click


@click.command("generate", short_help="Generate a set of images with the CE-VAE model (requires a pretrained model)")
@click.option("-c", "--config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True, default=Path("./config/defaults.yaml"))
@pass_environment
def generate(ctx, config_file):

    main(
        run="generate",
        config_file=config_file
    )
