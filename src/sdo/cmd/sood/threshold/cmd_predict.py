from sdo.cli import pass_environment
from sdo.sood.algorithms.threshold import main
from pathlib import Path
import click


@click.command("predict", short_help="Predicts anomaly scores using a threshold-based model")
@click.option("-m", "--predict-mode", default=None, type=click.Choice(["pixel", "sample"], case_sensitive=False))
@click.option("-c", "--config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True, default=Path("./config/defaults.yaml"))
@pass_environment
def predict(ctx, predict_mode, config_file):

    main(
        run="predict",
        predict_mode=predict_mode,
        config_file=config_file
    )
