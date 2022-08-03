from sdo.cli import pass_environment
from sdo.sood.algorithms.ce_vae import main
from pathlib import Path
import click


@click.command("reconstruct", short_help="Reconstructs input images (requires a pretrained model)")
@click.option("-c", "--config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True, default=Path("./config/defaults.yaml"))
@pass_environment
def reconstruct(ctx, config_file):

    main(
        run="reconstruct",
        config_file=config_file
    )
